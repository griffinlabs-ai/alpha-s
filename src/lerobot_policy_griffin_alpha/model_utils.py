"""Model-construction and checkpoint-saving helpers shared by both policy types."""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn.init as _nn_init
from torch import nn

DEFAULT_MAX_NEW_TOKENS = 512

_INIT_FN_NAMES = (
    "uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_",
    "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_",
)


@contextlib.contextmanager
def no_random_init(model_cls: type):
    """Skip random weight initialization while constructing ``model_cls`` on the reload path.

    ``PreTrainedPolicy.from_pretrained`` builds the policy (constructing a fresh transformers model)
    and then overwrites every parameter from the checkpoint, so the random fill -- minutes for a
    4B-parameter model -- is pure wasted work. Tensors are still allocated, just left unfilled until
    the checkpoint lands. Both ``torch.nn.init`` (used by each submodule's ``reset_parameters``) and
    the transformers class's ``_init_weights`` are no-op'd for the duration.
    """
    noop = lambda tensor, *args, **kwargs: tensor  # noqa: E731 (in-place init fns return the tensor)
    saved = {name: getattr(_nn_init, name) for name in _INIT_FN_NAMES if hasattr(_nn_init, name)}
    saved_init_weights = model_cls._init_weights
    try:
        for name in saved:
            setattr(_nn_init, name, noop)
        model_cls._init_weights = lambda self, module: None
        yield
    finally:
        for name, fn in saved.items():
            setattr(_nn_init, name, fn)
        model_cls._init_weights = saved_init_weights


def save_model_safetensors(module: nn.Module, path: Path, state_dict: dict | None = None) -> None:
    """Save ``module.state_dict()`` (or a caller-supplied ``state_dict``) as safetensors, deduplicating tied weights by storage address.

    ``safetensors.torch.save_model`` resolves the tied ``embed_tokens``/``lm_head`` pair by keeping
    the tensor that covers their shared storage, which fails whenever parameters are views into a
    larger flat buffer (as under some distributed optimizers). Rebuilding a plain CPU state dict
    instead is robust on every path: aliases are detected by ``(data_ptr, shape, dtype)``, the
    ``lm_head``-style name is kept (matching ``save_model``'s pick, so the checkpoint layout is
    unchanged), and each kept tensor is cloned into its own storage.
    """
    from safetensors.torch import save_file

    sd = module.state_dict() if state_dict is None else state_dict
    keep_by_addr: dict[tuple, str] = {}
    for name, t in sd.items():
        addr = (t.data_ptr(), tuple(t.shape), t.dtype)
        prev = keep_by_addr.get(addr)
        if prev is None or ("lm_head" in name and "lm_head" not in prev):
            keep_by_addr[addr] = name

    def _materialize(copy: bool) -> dict:
        return {
            name: (sd[name].detach().to("cpu", copy=True).contiguous() if copy
                   else sd[name].detach().cpu().contiguous())
            for name in keep_by_addr.values()
        }

    try:
        # No copy first: after dedup the kept tensors are distinct, and a CPU-resident checkpoint
        # (e.g. a re-save on a machine without a GPU) would otherwise need twice its size in RAM.
        save_file(_materialize(copy=False), str(path), metadata={"format": "pt"})
    except RuntimeError:
        # safetensors refuses tensors that still share storage (views into one flat buffer): clone
        # each survivor into its own storage and retry.
        save_file(_materialize(copy=True), str(path), metadata={"format": "pt"})


def resolve_dtype(dtype) -> torch.dtype | None:
    """A config ``dtype`` round-trips through JSON as a string (e.g. ``"bfloat16"``)."""
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        resolved = getattr(torch, dtype, None)
        return resolved if isinstance(resolved, torch.dtype) else None
    return None
