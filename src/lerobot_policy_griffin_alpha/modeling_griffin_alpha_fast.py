"""``GriffinAlphaFASTPolicy``: Griffin Alpha-S with the FAST action-token head.

The action chunk is encoded by the FAST tokenizer (DCT + BPE) into tokens appended to the text
vocabulary, trained with the backbone's own next-token cross-entropy, and decoded at inference by
greedy generation followed by a relaxed inverse DCT (``fast_decoding``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .backbone import GriffinAlphaBackbonePolicy
from .configuration_griffin_alpha_fast import GriffinAlphaFASTConfig
from .fast_decoding import decode_action_tokens, relaxed_fast_decode


class GriffinAlphaFASTPolicy(GriffinAlphaBackbonePolicy):
    config_class = GriffinAlphaFASTConfig
    name = "griffin_alpha_fast"

    config: GriffinAlphaFASTConfig

    def __init__(
        self,
        config: GriffinAlphaFASTConfig,
        qwen3vl_model: Qwen3VLForConditionalGeneration | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(config, qwen3vl_model, *args, **kwargs)
        self.fast_tokenizer = AutoProcessor.from_pretrained(config.fast_action_tokenizer_name, trust_remote_code=True)
        self.fast_tokenizer.action_dim = config.max_action_dim
        self.fast_tokenizer.time_horizon = config.horizon

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        model_inputs = self._filter_model_inputs(batch)
        outputs = self.model(**model_inputs)
        loss = outputs.loss
        return loss, {"loss": loss.item()}

    def _relaxed_fast_decode(self, fast_ids: list[int], time_horizon: int, action_dim: int):
        return relaxed_fast_decode(self.fast_tokenizer, fast_ids, time_horizon, action_dim)

    def _decode_action_tokens(self, generated_ids: Tensor, n_action_dims: Sequence[int] | None = None) -> Tensor:
        return decode_action_tokens(
            generated_ids,
            n_action_dims,
            fast_tokenizer=self.fast_tokenizer,
            action_token_min=self.config.action_token_min,
            action_token_max=self.config.action_token_max,
            horizon=self.config.horizon,
            max_action_dim=self.config.max_action_dim,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        model_inputs = self._filter_model_inputs(batch)
        model_inputs.pop("labels", None)
        generated_ids = self.model.generate(**model_inputs)
        chunk = self._decode_action_tokens(generated_ids, batch.get("n_action_dims"))
        return self._resample_chunk(chunk)
