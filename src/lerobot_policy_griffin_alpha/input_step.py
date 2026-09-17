"""The prompt-building input processor step shared by both heads.

Both registered input steps (``griffinlabs/griffin_alpha_input`` for the flow head,
``griffinlabs/griffin_alpha_fast_input`` for FAST) render the same user turn:

    [embodiment: {e}; arm control mode: {m}] {proprio tokens}  <image> x N
    {task}
    predict subtask: {true|false}

and differ only in what follows it -- FAST appends an assistant turn holding the tokenized action
chunk at training time; the flow head never emits an assistant turn and instead attaches the
continuous chunk to the batch. Those two behaviours are the leaf hooks ``_render_prompt`` and
``_finish_batch``.

Qwen-specific pinning, all of it load-bearing:

- **Visual-token band.** Qwen3-VL is dynamic-resolution: the per-image token count is steered by a
  pair of aspect-preserving *pixel* bounds (``shortest_edge``/``longest_edge`` are Qwen's
  ``min_pixels``/``max_pixels``; each merged visual token spans a 32x32-px cell). ``smart_resize``
  either leaves a frame alone, floors it down (above the max) or **ceils it up** (below the min), so
  the two bounds must not be collapsed into one: with ``min == max`` every frame takes the ceil
  branch and undersized ones overshoot the cap. Keeping ``min`` strictly below ``max`` holds the count
  in a 60-72 band that depends only on aspect ratio, not capture resolution.
- **Vocab alignment.** The raw Qwen tokenizer assigns ids only up to 151668 while the model's
  embedding table has 151936 rows. The gap is padded with placeholder tokens so the appended
  action/proprio tokens start exactly at the model's base vocab; skipping this would put them on
  untrained rows with no crash and no warning. Asserted in ``__post_init__``.
- **do_rescale.** Images arrive as float tensors in [0, 1] (lerobot's decode convention), i.e.
  already rescaled, so the image processor's own 1/255 rescale is disabled.
- **Label masking** keys on ``<|im_start|>`` (ChatML).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoProcessor

from lerobot.configs.types import PipelineFeatureType
from lerobot.processor import ProcessorStep, create_transition
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .backbone import QWEN_PATCH_MERGE, GriffinAlphaBackboneConfig
from .prompt_utils import (
    GriffinAlphaImageTransform,
    index_optional_list,
    make_proprio_state_tokens,
    map_normalized_state_to_vlm_proprio,
)

logger = logging.getLogger(__name__)


@dataclass
class SampleContext:
    """Everything a leaf needs to finish one sample's prompt."""

    index: int
    task: str
    subtask: str
    n_action_dims: int | None
    action: torch.Tensor | None  # (chunk, dim) for this sample, or None at inference


@dataclass
class GriffinAlphaPromptStep(ProcessorStep):
    """Base input step. Not registered; see the two leaves."""

    base_vlm_processor_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_tokens_per_image: int = 72
    min_tokens_per_image: int = 60
    max_sequence_length: int = 500
    proprio_vocab_size: int = 256
    action_vocab_size: int = 2048
    # The model's base vocab size (embedding rows). The raw tokenizer is padded up to this so the
    # appended action/proprio token ids line up with the model's ``action_token_min``.
    tokenizer_vocab_pad_to: int = 151936
    apply_image_augmentation: bool = True
    apply_inference_center_crop: bool = False
    include_proprio: bool = True
    # Empty means "every observation.images.* key present, in order". Order matters (see the config).
    image_keys: tuple[str, ...] = ()
    embodiment_prompt: str | None = None
    arm_control_mode: str | None = None
    predict_subtask: bool | None = None

    def __post_init__(self) -> None:
        # JSON round-trips turn the tuple into a list; normalize so equality stays stable after save/load.
        self.image_keys = tuple(self.image_keys)
        self._vla_processor = self._make_vla_processor()

        tokenizer = self._vla_processor.tokenizer
        action_min = tokenizer.convert_tokens_to_ids("<robot_action_0>")
        action_max = tokenizer.convert_tokens_to_ids(f"<robot_action_{self.action_vocab_size - 1}>")
        if action_min != self.tokenizer_vocab_pad_to or (
            action_max != self.tokenizer_vocab_pad_to + self.action_vocab_size - 1
        ):
            raise ValueError(
                f"Action token ids [{action_min}, {action_max}] do not start at the model base "
                f"vocab ({self.tokenizer_vocab_pad_to}); tokenizer/model vocab misalignment. The "
                "action tokens would index untrained embedding rows."
            )
        proprio_last = tokenizer.convert_tokens_to_ids(f"<proprio_state_{self.proprio_vocab_size - 1}>")
        if proprio_last != action_max + self.proprio_vocab_size:
            raise ValueError("Proprio state token ids are not contiguous after the action tokens.")

        self._turn_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        if self._turn_start_token_id is None:
            raise ValueError("Could not resolve <|im_start|>; label masking would mask everything.")
        self._turn_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        self._image_transform = GriffinAlphaImageTransform()

    def _make_vla_processor(self) -> AutoProcessor:
        processor = AutoProcessor.from_pretrained(self.base_vlm_processor_name, trust_remote_code=True)

        if not 0 < self.min_tokens_per_image < self.max_tokens_per_image:
            raise ValueError(
                f"Require 0 < min_tokens_per_image ({self.min_tokens_per_image}) < "
                f"max_tokens_per_image ({self.max_tokens_per_image})."
            )
        processor.image_processor.size = {
            "shortest_edge": self.min_tokens_per_image * QWEN_PATCH_MERGE**2,
            "longest_edge": self.max_tokens_per_image * QWEN_PATCH_MERGE**2,
        }
        # lerobot images are float tensors in [0, 1] -- already rescaled.
        processor.image_processor.do_rescale = False

        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

        n_pad = self.tokenizer_vocab_pad_to - len(processor.tokenizer)
        if n_pad < 0:
            raise ValueError(
                f"Tokenizer already has {len(processor.tokenizer)} tokens > pad target "
                f"{self.tokenizer_vocab_pad_to}."
            )
        if n_pad > 0:
            processor.tokenizer.add_tokens([f"<|vla_unused_{i}|>" for i in range(n_pad)], special_tokens=True)

        action_tokens = [f"<robot_action_{i}>" for i in range(self.action_vocab_size)]
        proprio_tokens = make_proprio_state_tokens(self.proprio_vocab_size)
        processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)
        return processor

    # -- leaf hooks --------------------------------------------------------------------------------
    def _render_prompt(self, content: list[dict], ctx: SampleContext) -> str:
        """Render the chat prompt for one sample from its user-turn ``content``."""
        raise NotImplementedError

    def _finish_batch(self, batch_input: dict[str, Any], transition: EnvTransition, ctxs: list[SampleContext]) -> None:
        """Attach head-specific tensors (labels, the continuous chunk, ...) to the tokenized batch."""

    def _generation_prompt(self, content: list[dict]) -> str:
        messages = [{"role": "user", "content": content}]
        return self._vla_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # -- the step ----------------------------------------------------------------------------------
    def _resolve_image_keys(self, observation: dict) -> tuple[str, ...]:
        if self.image_keys:
            return self.image_keys
        return tuple(k for k in observation if k.startswith(f"{OBS_IMAGES}."))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionKey.OBSERVATION]
        proprio_state = observation[OBS_STATE]
        batch_size = proprio_state.shape[0]
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        info = transition[TransitionKey.INFO]
        action = transition.get(TransitionKey.ACTION)
        has_action_labels = isinstance(action, torch.Tensor)
        # A training sample is one that carries an action chunk: augment it (when grad is enabled).
        apply_augmentation = self.apply_image_augmentation and has_action_labels and torch.is_grad_enabled()
        apply_center_crop = self.apply_inference_center_crop and not apply_augmentation
        image_keys = self._resolve_image_keys(observation)

        text_prompts = []
        batch_images = []
        ctxs: list[SampleContext] = []

        for i in range(batch_size):
            imgs_for_this_sample = []
            for image_key in image_keys:
                img_tensor = observation[image_key][i]
                if img_tensor is not None:
                    if apply_augmentation:
                        img_tensor = self._image_transform(img_tensor)
                    elif apply_center_crop:
                        img_tensor = self._image_transform.center_crop(img_tensor)
                    imgs_for_this_sample.append(img_tensor)
            batch_images.append(imgs_for_this_sample)

            n_action_dims = index_optional_list(info.get("n_action_dims"), i)
            if self.include_proprio:
                vlm_proprio = map_normalized_state_to_vlm_proprio(
                    proprio_state[i][:n_action_dims], self.proprio_vocab_size
                )
            else:
                vlm_proprio = ""

            task = index_optional_list(complementary_data.get("task"), i) or ""
            subtask = index_optional_list(complementary_data.get("subtask"), i) or ""
            predict_subtask = (
                index_optional_list(info.get("predict_subtask"), i) or self.predict_subtask or bool(subtask)
            )
            embodiment = index_optional_list(info.get("embodiment_prompt"), i) or self.embodiment_prompt or ""
            arm_control_mode = index_optional_list(info.get("arm_control_mode"), i) or self.arm_control_mode
            if arm_control_mode is None:
                raise ValueError(
                    "arm_control_mode must be provided via transition info or the policy config's "
                    "arm_control_mode, but both are missing."
                )

            content: list[dict] = [
                {
                    "type": "text",
                    "text": f"[embodiment: {embodiment}; arm control mode: {arm_control_mode}] {vlm_proprio}",
                }
            ]
            for _ in imgs_for_this_sample:
                content.append({"type": "image"})
            content.append(
                {"type": "text", "text": f"{task}\npredict subtask: {'true' if predict_subtask else 'false'}"}
            )

            ctx = SampleContext(
                index=i,
                task=task,
                subtask=subtask,
                n_action_dims=n_action_dims,
                action=action[i] if has_action_labels else None,
            )
            ctxs.append(ctx)
            text_prompts.append(self._render_prompt(content, ctx))

        batch_input = self._vla_processor(
            text=text_prompts,
            images=batch_images,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )
        batch_input["n_action_dims"] = info.get("n_action_dims")
        self._finish_batch(batch_input, transition, ctxs)
        return create_transition(complementary_data=batch_input)

    def transform_features(self, features):
        return {PipelineFeatureType.ACTION: {}, PipelineFeatureType.OBSERVATION: {}}

    def get_config(self) -> dict[str, Any]:
        return {
            "base_vlm_processor_name": self.base_vlm_processor_name,
            "max_tokens_per_image": self.max_tokens_per_image,
            "min_tokens_per_image": self.min_tokens_per_image,
            "max_sequence_length": self.max_sequence_length,
            "proprio_vocab_size": self.proprio_vocab_size,
            "action_vocab_size": self.action_vocab_size,
            "tokenizer_vocab_pad_to": self.tokenizer_vocab_pad_to,
            "apply_image_augmentation": self.apply_image_augmentation,
            "apply_inference_center_crop": self.apply_inference_center_crop,
            "include_proprio": self.include_proprio,
            "image_keys": self.image_keys,
            "embodiment_prompt": self.embodiment_prompt,
            "arm_control_mode": self.arm_control_mode,
            "predict_subtask": self.predict_subtask,
        }

    @classmethod
    def _base_kwargs_from_config(cls, config: GriffinAlphaBackboneConfig) -> dict[str, Any]:
        return dict(
            base_vlm_processor_name=config.base_vlm_processor_name,
            max_tokens_per_image=config.max_tokens_per_image,
            min_tokens_per_image=config.min_tokens_per_image,
            max_sequence_length=config.max_sequence_length,
            proprio_vocab_size=config.proprio_vocab_size,
            action_vocab_size=config.action_vocab_size,
            # action_token_min IS the model's base vocab size -- one source of truth.
            tokenizer_vocab_pad_to=config.action_token_min,
            apply_image_augmentation=config.apply_image_augmentation,
            apply_inference_center_crop=config.apply_inference_center_crop,
            include_proprio=config.include_proprio,
            image_keys=config.resolved_image_keys,
            embodiment_prompt=config.embodiment_prompt,
            arm_control_mode=config.arm_control_mode,
            predict_subtask=config.predict_subtask,
        )
