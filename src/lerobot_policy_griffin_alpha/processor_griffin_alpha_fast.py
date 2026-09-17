"""Input processor step + pre/post pipeline factory for ``griffin_alpha_fast``.

At training time the user turn (see ``input_step``) is followed by an assistant turn
``[subtask: <text>\\n]action: <robot_action_i>...`` holding the FAST-tokenized action chunk, and
``labels`` mask everything before that turn. At inference the prompt ends at the generation header
and the policy generates the action tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoProcessor

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.processor import EnvTransition

from .configuration_griffin_alpha_fast import GriffinAlphaFASTConfig
from .input_step import GriffinAlphaPromptStep, SampleContext
from .pipeline_common import build_pre_post_processors
from .prompt_utils import map_fast_token_to_vlm_action

logger = logging.getLogger(__name__)

_WARNED_TRUNCATION = False


@dataclass
@ProcessorStepRegistry.register("griffinlabs/griffin_alpha_fast_input")
class GriffinAlphaFASTInputProcessorStep(GriffinAlphaPromptStep):
    fast_action_tokenizer_name: str = "lerobot/fast-action-tokenizer"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._fast_tokenizer = AutoProcessor.from_pretrained(self.fast_action_tokenizer_name, trust_remote_code=True)

    def _render_prompt(self, content: list[dict], ctx: SampleContext) -> str:
        if ctx.action is None:
            return self._generation_prompt(content)
        sample_action = ctx.action[:, : ctx.n_action_dims]
        fast_tokens = self._fast_tokenizer(sample_action.cpu())[0]
        vlm_action = map_fast_token_to_vlm_action(fast_tokens)
        subtask_segment = f"subtask: {ctx.subtask}\n" if ctx.subtask else ""
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": f"{subtask_segment}action: {vlm_action}"}]},
        ]
        return self._vla_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def _finish_batch(self, batch_input: dict[str, Any], transition: EnvTransition, ctxs: list[SampleContext]) -> None:
        if not ctxs or ctxs[0].action is None:
            return  # inference: nothing to label
        labels = batch_input["input_ids"].clone()
        # Mask everything before the last turn start (the assistant turn's <|im_start|>).
        turn_start_id = self._turn_start_token_id
        truncated = 0
        for i in range(labels.size(0)):
            seq = labels[i]
            turn_indices = (seq == turn_start_id).nonzero(as_tuple=False)
            if turn_indices.numel() > 0:
                last_turn_index = turn_indices[-1].item()
                seq[:last_turn_index] = -100
                # A complete assistant turn ends with <|im_end|>. If none follows the turn start,
                # right-truncation cut the action chunk short and the labels are a partial FAST sequence.
                if not (seq[last_turn_index:] == self._turn_end_token_id).any():
                    truncated += 1
            else:
                seq[:] = -100  # no assistant turn survived truncation: no label at all
                truncated += 1

        pad_token_id = self._vla_processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch_input["labels"] = labels

        if truncated:
            global _WARNED_TRUNCATION
            if not _WARNED_TRUNCATION:
                _WARNED_TRUNCATION = True
                logger.warning(
                    "%d sample(s) in this batch lost part of their FAST action tokens to "
                    "max_sequence_length=%d right-truncation (further occurrences are not logged). "
                    "Raise max_sequence_length; Qwen3-VL has ample context.",
                    truncated, self.max_sequence_length,
                )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["fast_action_tokenizer_name"] = self.fast_action_tokenizer_name
        return config

    @classmethod
    def from_config(cls, config: GriffinAlphaFASTConfig) -> "GriffinAlphaFASTInputProcessorStep":
        return cls(fast_action_tokenizer_name=config.fast_action_tokenizer_name, **cls._base_kwargs_from_config(config))


def make_griffin_alpha_fast_pre_post_processors(
    config: GriffinAlphaFASTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[EnvTransition, EnvTransition],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pre/post pipelines for GriffinAlphaFASTPolicy (name derived by lerobot from the type string)."""
    return build_pre_post_processors(config, dataset_stats, GriffinAlphaFASTInputProcessorStep.from_config(config))
