"""The Qwen3-VL-4B backbone shared by both Griffin Alpha-S heads.

``GriffinAlphaBackboneConfig`` holds every field that describes the backbone, the prompt, the image
token budget and the action-space transform; ``GriffinAlphaBackbonePolicy`` builds and owns the
``Qwen3VLForConditionalGeneration`` and implements the lerobot ``PreTrainedPolicy`` plumbing that
does not depend on the head (loading, saving, freezing, the action queue). Neither is registered
with lerobot: the two registered leaves are ``griffin_alpha`` (flow-matching expert,
``configuration_griffin_alpha.py``) and ``griffin_alpha_fast`` (FAST tokens,
``configuration_griffin_alpha_fast.py``).

Vocabulary layout, which both heads depend on even though only FAST emits action tokens: the base
Qwen3-VL-4B text vocabulary has 151936 embedding rows while its tokenizer only assigns ids up to
151668, so the processor step pads the tokenizer up to ``action_token_min`` before appending the
2048 ``<robot_action_i>`` and 256 ``<proprio_state_i>`` tokens. The proprio tokens are part of the
prompt for both heads; getting the pad wrong would put them on untrained rows.
"""

from __future__ import annotations

import builtins
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import draccus
import torch
from torch import Tensor
from transformers import AutoConfig, GenerationConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl import Qwen3VLConfig
from transformers.utils import is_flash_attn_2_available

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .action_steps import ResampleActionProcessorStep
from .model_utils import DEFAULT_MAX_NEW_TOKENS, no_random_init, resolve_dtype, save_model_safetensors

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="GriffinAlphaBackbonePolicy")


# ``Qwen3VLConfig`` is a transformers config whose annotations contain forward references that
# draccus cannot resolve when it recurses into the type. Register leaf encode/decode hooks so
# draccus serializes it as a plain dict instead of introspecting its fields.
@draccus.encode.register
def _encode_qwen3vl_config(cfg: Qwen3VLConfig) -> dict:
    return cfg.to_dict()


draccus.decode.register(Qwen3VLConfig, lambda value: Qwen3VLConfig.from_dict(value))


# Qwen3-VL-4B-Instruct's text vocab size = the model's embedding table row count. See the module
# docstring for why the tokenizer has to be padded up to it.
DEFAULT_ACTION_TOKEN_MIN = 151936
DEFAULT_ACTION_VOCAB_SIZE = 2048
# The canonical action width the bases were pre-trained with: every embodiment's actions occupy
# the leading slots of a 32-wide vector, padded with zeros, and the real per-sample width travels
# as ``n_action_dims``.
MAX_ACTION_DIM = 32
ACTION_HORIZON = 50

# Qwen3-VL patch_size (16) x spatial_merge_size (2): one merged visual token per 32x32-px cell.
QWEN_PATCH_MERGE = 16 * 2

FLASH_ATTENTION_2 = "flash_attention_2"
ATTN_FALLBACK = "sdpa"


def resolve_attn_implementation(requested: str, *, require: bool = False) -> str:
    """Resolve the attention backend, degrading FlashAttention 2 to SDPA when unavailable.

    Both Qwen3-VL's text and vision stacks support FA2, so it covers the whole model. But transformers
    raises ``ImportError`` at construction time if FA2 is requested and ``flash_attn`` is missing,
    which would make the plugin unusable on any machine without it. So the default asks for FA2 and
    falls back -- with a warning, never silently. Pass ``require=True``
    (``require_attn_implementation`` on the config) on a training job so a missing flash-attn fails
    the launch instead of quietly costing throughput.
    """
    if requested != FLASH_ATTENTION_2:
        return requested
    if is_flash_attn_2_available():
        return FLASH_ATTENTION_2
    if require:
        raise RuntimeError(
            "attn_implementation='flash_attention_2' was required but flash-attn is not available. "
            "Install it (see the [flash] extra in pyproject.toml) or set "
            "require_attn_implementation=False to fall back to SDPA."
        )
    logger.warning(
        "FlashAttention 2 requested but unavailable (flash-attn not installed or unsupported GPU); "
        "falling back to %r. Throughput will be lower than an FA2 run.",
        ATTN_FALLBACK,
    )
    return ATTN_FALLBACK


# Keys accepted by ``Qwen3VLForConditionalGeneration.forward``, intersected with what the Qwen3-VL
# processor emits ({input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids})
# plus the FAST head's labels. Video inputs are never used.
MODEL_INPUT_KEYS = frozenset(
    {
        "input_ids",
        "attention_mask",
        "labels",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    }
)


def _patch_vision_conv3d_fp32(model: Qwen3VLForConditionalGeneration) -> str | None:
    """Run Qwen3-VL's vision patch-embed ``Conv3d`` in fp32; return the module name it patched.

    Two reasons:

    1. **Throughput on affected torch builds.** torch 2.9.x has no fast bf16 ``Conv3d`` kernel for this
       shape on recent cuDNN/CUDA (pytorch#166122, pytorch#174051): bf16 falls back to a slow dilated
       conv that dominated the whole forward. The regression is absent in 2.8 and fixed in 2.10.
    2. **Parity with pre-training.** Every Griffin Alpha-S checkpoint was pre-trained with this wrapper
       in place, so its vision tower is adapted to an fp32 patch embedding. Keeping it on at fine-tune
       and inference time keeps numerics identical to training.

    The *parameter* stays in its original dtype (bf16); only the op runs in fp32, and gradients flow
    back through the cast. Set ``vision_patch_embed_fp32=False`` to drop it, accepting point 2.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):

            def _fp32_conv3d(x, _c=module):
                return torch.nn.functional.conv3d(
                    x.float(),
                    _c.weight.float(),
                    None if _c.bias is None else _c.bias.float(),
                    _c.stride,
                    _c.padding,
                    _c.dilation,
                    _c.groups,
                ).to(x.dtype)

            module.forward = _fp32_conv3d
            return name
    return None


@dataclass
class GriffinAlphaBackboneConfig(PreTrainedConfig):
    """Fields shared by both heads. Not registered as a policy type; see the two leaf configs."""

    output_features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(MAX_ACTION_DIM,)),
    })

    horizon: int = ACTION_HORIZON
    n_action_steps: int = ACTION_HORIZON

    qwen3vl_config: Qwen3VLConfig = field(default_factory=Qwen3VLConfig)

    # Vocabulary layout (see the module docstring). ``action_token_min`` doubles as the tokenizer pad
    # target, so both heads carry these even though only FAST emits action tokens.
    action_vocab_size: int = DEFAULT_ACTION_VOCAB_SIZE
    action_token_min: int = DEFAULT_ACTION_TOKEN_MIN
    action_token_max: int = DEFAULT_ACTION_TOKEN_MIN + DEFAULT_ACTION_VOCAB_SIZE - 1
    proprio_vocab_size: int = 256

    # Per-image visual-token band. Qwen3-VL is dynamic-resolution, so the token count is steered by
    # a pair of aspect-preserving *pixel* bounds (min/max_tokens_per_image * QWEN_PATCH_MERGE**2)
    # rather than set directly. Together they hold every aspect ratio in a 60-72 band, independent
    # of capture resolution:
    #
    #     1:1 -> 64    4:3 / 5:4 / 3:4 -> 63    16:9 / 9:16 -> 66    3:2 / 16:10 / 21:9 -> 60
    #     2:1 -> 72
    #
    # The max is 72 rather than 70 purely to cover 2:1: on an integer 32-px-cell grid its
    # aspect-preserving options are 5x11=55 or 6x12=72, so a 70 cap dropped it to 55. Both bounds
    # matter and min must stay strictly below max: Qwen's smart_resize rounds frames BELOW the
    # minimum up, so a collapsed band makes undersized frames overshoot the cap.
    max_tokens_per_image: int = 72
    min_tokens_per_image: int = 60
    # Max total token length per sample; longer sequences are right-truncated. The FAST head's action
    # tokens sit at the END of the sequence, so an over-budget sample loses (part of) its labels; the
    # FAST input step warns when that happens. ~310 tokens for a 3-camera prompt at 14 action dims,
    # up to ~570 at 32 dims with high-motion chunks.
    max_sequence_length: int = 500

    # Attention backend. The default *requests* FlashAttention 2 and degrades to SDPA with a warning
    # when flash-attn is missing; after construction the policy rewrites this field to what is
    # actually in use, so a saved checkpoint never claims FA2 for a run that fell back.
    attn_implementation: str = "flash_attention_2"
    # Set True on a training job so a missing flash-attn fails the launch instead of costing
    # throughput for the whole run.
    require_attn_implementation: bool = False
    # Run the vision patch-embed Conv3d in fp32. See ``_patch_vision_conv3d_fp32``.
    vision_patch_embed_fp32: bool = True

    base_vlm_processor_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    # Training-time random crop (0.9 area) + colour jitter, applied only when a sample carries an
    # action chunk and grad is enabled. ``apply_inference_center_crop`` applies the matching 0.9
    # centre crop at inference so the field of view matches training.
    apply_image_augmentation: bool = True
    apply_inference_center_crop: bool = False
    gradient_checkpointing: bool = True
    # Include the discretized proprio-state tokens in the prompt. Disabling forces a
    # vision+language-only policy; with few deterministic demos, prompt proprio can invite a
    # state->trajectory shortcut that never learns to look.
    include_proprio: bool = True
    # Freeze the whole visual stack (Qwen3-VL's vision->LM merger lives inside ``visual`` and the
    # encoder injects features at several text layers, so this is one coarse cut).
    freeze_vision_tower: bool = False
    # Freeze embed_tokens. Qwen3-VL ties lm_head to embed_tokens, so this also freezes the output
    # projection.
    freeze_embeddings: bool = False

    # Camera keys, in prompt order. ORDER MATTERS: the model was pre-trained with the primary
    # (exocentric) camera first, then wrist cameras. The empty default means "every VISUAL input
    # feature, in the order the dataset/env lists them" (see ``resolved_image_keys``); set it
    # explicitly when that order is not the one you want.
    image_keys: tuple[str, ...] = ()
    # Policy-wide values for conditioning fields that a dataset's per-sample ``info`` may override.
    # ``arm_control_mode`` must be resolvable from one source or the other (no silent default);
    # the released checkpoints use "eef_pose" and "joint".
    embodiment_prompt: str | None = None
    arm_control_mode: str | None = None
    predict_subtask: bool | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )
    state_key: str = OBS_STATE

    # ---- Action-space transform -----------------------------------------------------------------
    # Default path: lerobot's built-in relative-action steps. Actions are predicted relative to the
    # current ``observation.state`` on every dimension EXCEPT those whose dataset feature name
    # matches an entry of ``relative_exclude_joints`` (grippers stay absolute). This is how the
    # released bases were pre-trained. ``action_feature_names`` is filled in by lerobot's
    # ``make_policy`` from the dataset's ``action.names`` -- without names, EVERY dimension becomes
    # relative, gripper included, so give your dataset action names.
    use_relative_actions: bool = True
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None
    # Opt-in SE(3) path for absolute-pose action spaces that embed 4x4 homogeneous matrices in the
    # action/state vectors: ``se3_segment_start_idxs`` lists where each 16-D flattened matrix starts,
    # ``relative_action_mask`` marks which dims are made relative to the chunk-start state (matrix
    # segments via ``state^-1 @ action``, real dims by subtraction). Requires
    # ``use_relative_actions=False``. See ``action_steps`` and ``reconnect_se3_steps``.
    relative_action_mask: list[bool] | None = None
    se3_segment_start_idxs: list[int] | None = None
    # Cubic-spline resample of the dataset's action chunk to ``horizon`` steps. Not combinable with
    # the SE(3) path (interpolating matrices is not implemented).
    resample_action_chunk_size: int | None = None

    # ---- Fine-tuning presets ------------------------------------------------------------------
    optimizer_lr: float = 3e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 50.0

    # lerobot's cosine schedule auto-scales BOTH numbers when a run is shorter than
    # scheduler_decay_steps, so for runs under 30k steps these act as a 6.7% warmup fraction. For
    # longer runs the decay reaches scheduler_decay_lr at 30k and stays there: set
    # scheduler_decay_steps to the run length.
    scheduler_warmup_steps: int = 2_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than horizon ({self.horizon})"
            )
        if not 0 < self.min_tokens_per_image < self.max_tokens_per_image:
            raise ValueError(
                f"Require 0 < min_tokens_per_image ({self.min_tokens_per_image}) < "
                f"max_tokens_per_image ({self.max_tokens_per_image})."
            )
        self.image_keys = tuple(self.image_keys)
        if not self.se3_segment_start_idxs:  # [] and None both mean "no SE(3) segments"
            self.se3_segment_start_idxs = None
        if self.relative_action_mask is not None and self.se3_segment_start_idxs is None:
            raise ValueError(
                "relative_action_mask belongs to the SE(3) path and needs se3_segment_start_idxs. For "
                "plain relative actions use use_relative_actions / relative_exclude_joints instead."
            )
        if self.se3_segment_start_idxs is not None and self.use_relative_actions:
            raise ValueError(
                "se3_segment_start_idxs is set but use_relative_actions is True; the SE(3) path has its "
                "own relative step, so set use_relative_actions=False (and relative_action_mask)."
            )
        if self.se3_segment_start_idxs is not None and self.resample_action_chunk_size is not None:
            raise ValueError(
                "Resampling and SE(3) matrices cannot be used at the same time, as correct "
                "interpolation of SE(3) values is not implemented."
            )

    def validate_features(self) -> None:
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(MAX_ACTION_DIM,))

    @property
    def resolved_image_keys(self) -> tuple[str, ...]:
        """``image_keys`` when set, else every VISUAL input feature in order."""
        if self.image_keys:
            return self.image_keys
        return tuple(k for k, ft in (self.input_features or {}).items() if ft.type is FeatureType.VISUAL)

    @property
    def image_pixel_budget(self) -> int:
        """Upper aspect-preserving pixel bound (Qwen's ``max_pixels`` / ``longest_edge``)."""
        return self.max_tokens_per_image * QWEN_PATCH_MERGE**2

    @property
    def image_pixel_floor(self) -> int:
        """Lower aspect-preserving pixel bound (Qwen's ``min_pixels`` / ``shortest_edge``)."""
        return self.min_tokens_per_image * QWEN_PATCH_MERGE**2

    @property
    def image_processor_size(self) -> dict[str, int]:
        """The ``image_processor.size`` dict that pins the visual-token band."""
        return {"shortest_edge": self.image_pixel_floor, "longest_edge": self.image_pixel_budget}

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def max_action_dim(self) -> int:
        return self.output_features[ACTION].shape[-1]


class GriffinAlphaBackbonePolicy(PreTrainedPolicy):
    """Owns the Qwen3-VL model and the head-independent lerobot plumbing. Subclass per head."""

    config_class = GriffinAlphaBackboneConfig
    # lerobot requires every PreTrainedPolicy subclass to carry a name; this one is never registered as
    # a policy type (only the two leaves are), so the value is informational.
    name = "griffin_alpha_backbone"
    config: GriffinAlphaBackboneConfig

    def __init__(
        self,
        config: GriffinAlphaBackboneConfig,
        qwen3vl_model: Qwen3VLForConditionalGeneration | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)
        config.validate_features()

        # Resolve once, here, so every entry point (reload, from_qwen3vl_model, from_qwen3vl_pretrained)
        # ends up with the same backend and the same truthful record of it.
        attn_implementation = resolve_attn_implementation(
            config.attn_implementation, require=config.require_attn_implementation
        )

        if qwen3vl_model is not None:
            self.model = qwen3vl_model
            current = getattr(self.model.config, "_attn_implementation", None)
            if current != attn_implementation:
                try:
                    self.model.set_attn_implementation(attn_implementation)
                except Exception as exc:  # not settable post-hoc on this class/version
                    logger.warning(
                        "Could not switch attention implementation from %r to %r: %s. Keeping %r.",
                        current, attn_implementation, exc, current,
                    )
                    attn_implementation = current
        else:
            # Fresh construction happens only on the reload path (from_pretrained), where every weight
            # is immediately overwritten by the checkpoint -- so skip the slow random init.
            config.qwen3vl_config._attn_implementation = attn_implementation
            with no_random_init(Qwen3VLForConditionalGeneration):
                self.model = Qwen3VLForConditionalGeneration(config.qwen3vl_config)

        # Record what is actually in use, not what was asked for.
        self.config.attn_implementation = attn_implementation
        logger.info("Attention implementation: %s", attn_implementation)

        self.config.qwen3vl_config = self.model.config

        # Reload builds the model with the plain constructor, which (unlike HF from_pretrained) does
        # not build under the config dtype, and load_state_dict keeps the destination dtype -- so a
        # bfloat16 checkpoint would silently reload as (mixed) float32. Cast explicitly.
        model_dtype = resolve_dtype(getattr(self.config.qwen3vl_config, "dtype", None))
        if model_dtype is not None:
            self.model = self.model.to(model_dtype)

        if config.vision_patch_embed_fp32:
            name = _patch_vision_conv3d_fp32(self.model)
            logger.info(
                "Vision patch-embed Conv3d -> fp32 forward: %s",
                name or "NOT FOUND (check the model architecture!)",
            )

        if config.freeze_vision_tower:
            self.model.model.visual.requires_grad_(False)
        if config.freeze_embeddings:
            # lm_head is weight-tied to embed_tokens, so this freezes the output projection too.
            self.model.model.language_model.embed_tokens.requires_grad_(False)
            self.model.lm_head.requires_grad_(False)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False

        self._action_resampler = (
            ResampleActionProcessorStep(target_chunk_size=config.resample_action_chunk_size)
            if config.resample_action_chunk_size
            else None
        )
        self.reset()

    # -- lerobot plumbing ----------------------------------------------------------------------
    def reset(self) -> None:
        self._action_queue: deque[Tensor] = deque()

    def get_optim_params(self) -> list[Tensor]:
        # lerobot's optimizer factory passes this straight to torch.optim.AdamW, which expects an
        # iterable of tensors (or param-group dicts).
        return [p for p in self.parameters() if p.requires_grad]

    def _save_pretrained(self, save_directory: Path, state_dict: dict[str, Tensor] | None = None) -> None:
        # lerobot passes a gathered ``state_dict`` under FSDP; otherwise ours is used.
        self.config._save_pretrained(save_directory)
        save_model_safetensors(self, save_directory / "model.safetensors", state_dict=state_dict)
        self.model.generation_config.save_pretrained(save_directory)

    @staticmethod
    def _filter_model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {key: value for key, value in batch.items() if key in MODEL_INPUT_KEYS and value is not None}

    def _resample_chunk(self, chunk: Tensor) -> Tensor:
        """Inverse of the input pipeline's ``ResampleActionProcessorStep`` (a no-op unless configured)."""
        if self._action_resampler is None:
            return chunk
        return self._action_resampler({ACTION: chunk})[ACTION]

    # -- head interface --------------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:  # pragma: no cover
        """``(B, horizon, action_dim)`` in the policy's (normalized, transformed) action space."""
        raise NotImplementedError

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # -- constructors ------------------------------------------------------------------------------
    @classmethod
    def from_qwen3vl_model(
        cls: builtins.type[T],
        qwen3vl_model: Qwen3VLForConditionalGeneration,
        config: GriffinAlphaBackboneConfig | None = None,
        *,
        resize_embeddings: bool = True,
        **kwargs,
    ) -> T:
        """Wrap a Qwen3-VL transformers model as a Griffin Alpha-S policy.

        With ``resize_embeddings=True`` (a fresh base model) the token embeddings are grown to hold
        the action and proprio tokens; Qwen3-VL ties ``lm_head`` to the embedding table, so
        ``resize_token_embeddings`` covers everything. Pass ``resize_embeddings=False`` for a model
        whose embeddings were already resized; the action range is then derived from the enlarged
        vocab.
        """
        base_config = qwen3vl_model.config
        config = config or cls.config_class(qwen3vl_config=base_config)
        if resize_embeddings:
            base_vocab = base_config.text_config.vocab_size
            new_vocab_size = base_vocab + config.action_vocab_size + config.proprio_vocab_size
            qwen3vl_model.resize_token_embeddings(new_vocab_size)
        else:
            final_vocab = qwen3vl_model.config.text_config.vocab_size
            base_vocab = final_vocab - config.action_vocab_size - config.proprio_vocab_size
            if base_vocab <= 0:
                raise ValueError(
                    f"Model vocab size ({final_vocab}) is too small to contain "
                    f"{config.action_vocab_size} action and {config.proprio_vocab_size} proprio "
                    "tokens; the model does not appear to be an adapted Griffin Alpha-S checkpoint. "
                    "Pass resize_embeddings=True to adapt a base model."
                )
        # Derive the action token range from the base vocab in both cases so it never relies on the
        # action_token_min default matching this particular model's vocab size.
        config.action_token_min = base_vocab
        config.action_token_max = base_vocab + config.action_vocab_size - 1
        config.qwen3vl_config = qwen3vl_model.config
        qwen3vl_model.generation_config.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        qwen3vl_model.generation_config.do_sample = False
        # The base Qwen3-VL generation config ships sampling params (top_p/top_k/temperature). With
        # do_sample=False these are unused and make GenerationConfig.save_pretrained fail its strict
        # validation, so clear them for greedy decoding.
        qwen3vl_model.generation_config.top_p = None
        qwen3vl_model.generation_config.top_k = None
        qwen3vl_model.generation_config.temperature = None
        policy = cls(config, qwen3vl_model, **kwargs)
        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def from_pretrained(cls: builtins.type[T], pretrained_name_or_path: str | Path, **kwargs) -> T:
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        download_keys = (
            "cache_dir", "force_download", "resume_download", "proxies", "token", "revision",
            "local_files_only",
        )
        gen_kwargs = {key: kwargs[key] for key in download_keys if key in kwargs}
        try:
            policy.model.generation_config = GenerationConfig.from_pretrained(
                pretrained_name_or_path, **gen_kwargs
            )
        except OSError:
            logger.info("No generation_config found for %s; keeping defaults.", pretrained_name_or_path)
        return policy

    @classmethod
    def from_qwen3vl_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: Qwen3VLConfig | None = None,
        resize_embeddings: bool = True,
        policy_config: GriffinAlphaBackboneConfig | None = None,
        **load_kwargs,
    ) -> T:
        """Load a Qwen3-VL checkpoint from the Hub or disk and wrap it as a Griffin Alpha-S policy."""
        policy_config = policy_config or cls.config_class()
        attn_implementation = resolve_attn_implementation(
            policy_config.attn_implementation, require=policy_config.require_attn_implementation
        )
        qwen3vl_config = config or AutoConfig.from_pretrained(pretrained_name_or_path, **load_kwargs)
        qwen3vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_name_or_path,
            config=qwen3vl_config,
            attn_implementation=attn_implementation,
            **load_kwargs,
        )
        return cls.from_qwen3vl_model(qwen3vl_model, policy_config, resize_embeddings=resize_embeddings)
