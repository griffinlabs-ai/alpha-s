"""Input processor step + pre/post pipeline factory for ``griffin_alpha`` (the flow-matching head).

The flow head regresses the normalized action chunk directly, so the prompt is built in inference
form at training time too: no assistant turn holding action tokens, no ``labels``. Train and
inference therefore see byte-identical prefixes and the expert's positions need no correction. The
step attaches the continuous chunk (padded to the head width) to the batch instead, together with
the per-sample real width and the chunk pad mask the loss masks on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline, ProcessorStepRegistry
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION

from .configuration_griffin_alpha import CANONICAL_ACTION_DIMS, GriffinAlphaConfig
from .input_step import GriffinAlphaPromptStep, SampleContext
from .pipeline_common import build_pre_post_processors

logger = logging.getLogger(__name__)

# One notice per process for the subtask inference gap (see ``condition_on_subtask``). A module
# global rather than an instance attribute because the pipeline is rebuilt per DataLoader worker and
# per reload, and the point is to say it once.
_WARNED_SUBTASK_UNAVAILABLE = False


@dataclass
@ProcessorStepRegistry.register("griffinlabs/griffin_alpha_input")
class GriffinAlphaInputProcessorStep(GriffinAlphaPromptStep):
    """The shared prompt, plus the continuous action target the flow head regresses.

    ``max_action_dim`` is the expert's head width (pinned by the config's ``action_dim_override``,
    not derived from the dataset), so the emitted chunk is padded to it and the per-sample real width
    travels separately in ``n_action_dims``.
    """

    max_action_dim: int = CANONICAL_ACTION_DIMS
    # End the prompt with a truncated assistant turn (``[subtask: <text>\n]action: ``) so the expert
    # conditions on the ground-truth subtask when the dataset has one. See the config field.
    condition_on_subtask: bool = True

    def _render_prompt(self, content: list[dict], ctx: SampleContext) -> str:
        prompt = self._generation_prompt(content)
        if not self.condition_on_subtask:
            return prompt
        if ctx.action is None:
            global _WARNED_SUBTASK_UNAVAILABLE
            if not _WARNED_SUBTASK_UNAVAILABLE:
                _WARNED_SUBTASK_UNAVAILABLE = True
                logger.warning(
                    "condition_on_subtask=True: training prefixes carried a ground-truth `subtask:` line "
                    "when the dataset had one, but an inference batch has no subtask, so every prompt from "
                    "here on takes the no-subtask branch (assistant -> `action: `). That branch is a trained "
                    "form, but a minority one in pre-training."
                )
        # Appended to the RENDERED string, not passed as an assistant message: the template would close
        # the turn with <|im_end|>, and pre-training's <|im_end|> comes after the action. The subtask
        # line is omitted entirely when the frame has none, exactly as pre-training omits it.
        if ctx.subtask:
            prompt += f"subtask: {ctx.subtask}\n"
        return prompt + "action: "

    def _finish_batch(self, batch_input: dict[str, Any], transition: EnvTransition, ctxs: list[SampleContext]) -> None:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, torch.Tensor):
            return  # inference: no chunk to attach

        chunk = action if action.ndim == 3 else action[None]
        real_width = chunk.shape[-1]
        if real_width < self.max_action_dim:
            chunk = torch.nn.functional.pad(chunk, (0, self.max_action_dim - real_width))
        batch_input[ACTION] = chunk

        # The loss masks padded dims via n_action_dims. Without this a dataset that supplies none would
        # train the zero padding as signal: the loss divides by the kept count (a 14-D embodiment in a
        # 32-wide head would carry 14/32 of the gradient), and a padded dim has a=0, so the expert is
        # asked to recover eps from t*eps -- a 1/t gain at t as low as 0.001. Set from the pre-pad
        # width, but never over a per-row value the dataset did supply.
        if batch_input.get("n_action_dims") is None:
            batch_input["n_action_dims"] = [real_width] * chunk.shape[0]

        # Chunk steps past the episode end are dataset padding; lerobot marks them with
        # ``action_is_pad`` and the loss drops them when the mask is present.
        complementary = transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        for key in (f"{ACTION}_is_pad", "action_is_pad"):
            candidate = complementary.get(key)
            if isinstance(candidate, torch.Tensor):
                batch_input[f"{ACTION}_is_pad"] = candidate if candidate.ndim == 2 else candidate[None]
                break

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["max_action_dim"] = self.max_action_dim
        config["condition_on_subtask"] = self.condition_on_subtask
        return config

    @classmethod
    def from_config(cls, config: GriffinAlphaConfig) -> "GriffinAlphaInputProcessorStep":
        return cls(
            max_action_dim=config.max_action_dim,
            condition_on_subtask=config.condition_on_subtask,
            **cls._base_kwargs_from_config(config),
        )


def make_griffin_alpha_pre_post_processors(
    config: GriffinAlphaConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[EnvTransition, EnvTransition],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Pre/post pipelines for GriffinAlphaPolicy (name derived by lerobot from the type string)."""
    return build_pre_post_processors(config, dataset_stats, GriffinAlphaInputProcessorStep.from_config(config))
