"""FAST action-token decoding for the ``griffin_alpha_fast`` head.

Both functions are pure: they take the FAST tokenizer and the vocabulary layout explicitly, so the
policy class that owns them stays thin and the behaviour is testable without a backbone.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch
from scipy.fft import idct
from torch import Tensor

logger = logging.getLogger(__name__)


def relaxed_fast_decode(fast_tokenizer, fast_ids: list[int], time_horizon: int, action_dim: int) -> np.ndarray:
    """Decode FAST action tokens into a ``(time_horizon, action_dim)`` array with *relaxed* length handling.

    The ``lerobot/fast-action-tokenizer`` (``UniversalActionProcessor.decode``) reshapes the BPE-decoded
    DCT coefficients straight into ``(time_horizon, action_dim)`` and, on any size mismatch, silently
    returns all-zeros. Autoregressive generation routinely emits token sequences that decode to a few
    too many / too few coefficients (e.g. 496 or 501 against an expected 500), so that brittle path
    yields a dead (zeros) action chunk on most steps. Pad or truncate the coefficients to the expected
    length before the reshape instead, matching lerobot pi0-FAST's ``relaxed_decoding=True``.
    """
    expected = time_horizon * action_dim
    try:
        decoded_tokens = fast_tokenizer.bpe_tokenizer.decode(fast_ids)
        coeff = np.array(list(map(ord, decoded_tokens))) + fast_tokenizer.min_token
        if coeff.shape[0] > expected:
            coeff = coeff[:expected]  # truncate on the right
        elif coeff.shape[0] < expected:
            coeff = np.pad(coeff, (0, expected - coeff.shape[0]), mode="constant", constant_values=0)
        coeff = coeff.reshape(time_horizon, action_dim)
    except Exception as e:  # genuinely unparseable tokens -> zeros (rare)
        logger.warning(f"FAST relaxed decode failed, using zero action: {e}")
        coeff = np.zeros((time_horizon, action_dim))
    return idct(coeff / fast_tokenizer.scale, axis=0, norm="ortho")


def decode_action_tokens(
    generated_ids: Tensor,
    n_action_dims: Sequence[int] | None,
    *,
    fast_tokenizer,
    action_token_min: int,
    action_token_max: int,
    horizon: int,
    max_action_dim: int,
) -> Tensor:
    """Turn a batch of generated token ids into a ``(B, horizon, max_action_dim)`` float32 chunk.

    Only ids inside ``[action_token_min, action_token_max]`` are action tokens; everything else the
    model emitted (a subtask line, the end-of-turn marker) is skipped. A sample whose real width
    ``n_action_dims[i]`` is below ``max_action_dim`` is decoded at its real width and zero-padded, so
    the returned tensor is rectangular. A sample with no action tokens at all decodes to zeros.
    """
    batch_size = generated_ids.shape[0]
    decoded_actions = []

    for i in range(batch_size):
        seq = generated_ids[i]
        action_mask = (seq >= action_token_min) & (seq <= action_token_max)
        action_token_ids = seq[action_mask]
        if action_token_ids.numel() == 0:
            decoded_actions.append(
                torch.zeros(horizon, max_action_dim, dtype=torch.float32, device=generated_ids.device)
            )
            continue

        fast_ids = (action_token_ids - action_token_min).tolist()
        action_dim = n_action_dims[i] if n_action_dims is not None else max_action_dim
        action = relaxed_fast_decode(fast_tokenizer, fast_ids, horizon, action_dim)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=generated_ids.device)
        if action_tensor.shape[-1] < max_action_dim:
            pad = torch.zeros(
                horizon,
                max_action_dim - action_tensor.shape[-1],
                dtype=action_tensor.dtype,
                device=action_tensor.device,
            )
            action_tensor = torch.cat([action_tensor, pad], dim=-1)
        decoded_actions.append(action_tensor)

    return torch.stack(decoded_actions, dim=0)
