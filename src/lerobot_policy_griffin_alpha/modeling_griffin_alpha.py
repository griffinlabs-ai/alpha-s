"""``GriffinAlphaPolicy``: a flow-matching action expert beside the Qwen3-VL-4B backbone.

One backbone forward per step, and the only thing the expert takes from it is the per-layer K/V.
There is one objective -- the flow-matching regression -- and no FAST cross-entropy: ``_run_prefix``
strips ``labels`` and passes ``logits_to_keep=1`` so the lm_head projection never runs over the
sequence.

Layer-by-layer, per training step:

1. backbone layer *i* runs self-attention; the exact ``(K_i, V_i)`` it used are captured
   **post-RoPE, post-QK-norm, post-KV-share**, shape ``(B, 8, P, 128)``. Capture works by swapping
   the text config's attention implementation for a recorder, so none of Qwen's KV machinery is
   reimplemented here.
2. with ``freeze_backbone=True`` the backbone forward runs under ``no_grad`` and the expert is the
   only thing trained; with the default ``False`` the expert's gradient reaches the backbone through
   the captured K/V (joint training, two learning rates).
3. expert layer *i*: ``input_layernorm`` (adaptively modulated by the flow time) -> q_proj
   (1024 -> 2048), k/v_proj (1024 -> 1024) -> q_norm/k_norm on head_dim 128 -> mRoPE at the suffix
   positions.
4. **the mixture-of-transformers mixing happens here and only here**: ``K = cat(K_i, K_expert)``
   along the sequence axis (same for V); queries are only the expert's own 50. The attention matrix
   is ``(chunk, P + chunk)``. The two residual streams (2560 and 1024 wide) are never added -- only
   the K/V meet.
5. ``o_proj`` -> residual -> ``post_attention_layernorm`` -> SwiGLU -> residual.

The 50 action tokens are **not** ``input_ids``: no embedding lookup, no tokenizer budget, and the
backbone never sees them. They exist only in the expert's residual stream, created by
``action_in_proj``. Position information comes entirely from in-layer mRoPE.

One prompt form, both regimes: the input step builds the prefix in inference form even at training
(no assistant turn holding actions, no ``labels``), so ``visible`` is just the attention mask and the
suffix mRoPE positions continue from it. ``_prefix_geometry`` refuses a batch carrying ``labels``
rather than guessing which correction to apply.

Flow matching itself: ``x_t = t*eps + (1-t)*a``, target ``u_t = eps - a``,
``t ~ 0.001 + 0.999*Beta(1.5, 1)``, Euler integration from t=1 to t=0.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import Qwen3VLForConditionalGeneration
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    apply_rotary_pos_emb as qwen_apply_rotary_pos_emb,
)

from lerobot.utils.constants import ACTION

from .backbone import GriffinAlphaBackbonePolicy
from .configuration_griffin_alpha import BACKBONE_HEAD_DIM, BACKBONE_NUM_KV_HEADS, GriffinAlphaConfig

# Additive-mask "disallowed" value (openpi / pi0.5 convention).
MASK_NEG = -2.3819763e38
# A runtime key in transformers' attention registry, registered by capture_backbone_kv. Never
# serialized, so unlike the type string it is safe to rename.
_CAPTURE_IMPL = "griffin_alpha_kv_capture"


# --------------------------------------------------------------------------------------
# Flow-matching helpers -- the same schedule as lerobot's pi0.5 so results stay comparable.
# --------------------------------------------------------------------------------------
def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape (batch_size,).")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:
    # Beta sampling goes through _sample_dirichlet, unimplemented on MPS -- sample on CPU.
    dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta))
    return dist.sample((bsize,)).to(device)


def _init_modulation_zero(dense: nn.Linear, width: int) -> None:
    """Zero the projection and set the gate bias to 1, so modulation starts as the identity.

    scale=0, shift=0, gate=1 means the network at init is exactly the un-modulated one -- the same
    reason pi0.5 zero-initialises its adaptive-RMSNorm dense.
    """
    nn.init.zeros_(dense.weight)
    with torch.no_grad():
        dense.bias.zero_()
        dense.bias[2 * width :].fill_(1.0)


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def next_visible_position(position_ids: Tensor, visible: Tensor) -> Tensor:
    """Where the action tokens start: one past the last **visible** prefix position.

    ``position_ids`` is Qwen's (3, B, P) mRoPE grid; ``visible`` is (B, P). Derived from the visible
    mask rather than raw sequence length so left-padding never shifts the suffix.
    """
    bsize = visible.shape[0]
    masked = position_ids.masked_fill(~visible[None], -1)
    return masked.reshape(3, bsize, -1).amax(dim=(0, 2)) + 1


def expert_attention_mask(visible: Tensor, chunk: int, dtype: torch.dtype) -> Tensor:
    """Additive (B, 1, chunk, P + chunk): visible prefix + the full action block.

    The action tokens attend to each other bidirectionally and to the visible prefix only.
    """
    bsize, prefix_len = visible.shape
    prefix_allow = visible[:, None, None, :].expand(bsize, 1, chunk, prefix_len)
    suffix_allow = torch.ones(bsize, 1, chunk, chunk, dtype=torch.bool, device=visible.device)
    allow = torch.cat([prefix_allow, suffix_allow], dim=-1)
    return torch.zeros(allow.shape, dtype=dtype, device=visible.device).masked_fill(~allow, MASK_NEG)


@contextlib.contextmanager
def capture_backbone_kv(text_config, store: dict):
    """Route the backbone's text attention through a recorder that stashes each layer's (K, V).

    The captured tensors are the attention *inputs*, so they are post-RoPE / post-QK-norm /
    post-KV-share regardless of kernel. There is no detach switch: frozen, the captured tensors carry
    no grad anyway; unfrozen, the expert's gradient is meant to reach the backbone through them.

    The recorder delegates to whatever kernel was already configured rather than forcing eager, so
    what is captured is independent of the kernel and memory/throughput stay those of the configured
    attention (forcing eager would materialize (q_heads, T, T) per layer for nothing).
    """
    saved = text_config._attn_implementation

    def _recorder(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        store[module.layer_idx] = (key, value)
        real = ALL_ATTENTION_FUNCTIONS.get_interface(saved, modeling_qwen3_vl.eager_attention_forward)
        # Put the REAL implementation string back on the config for the duration of the delegate:
        # transformers' FA2 path re-reads ``module.config._attn_implementation`` at call time and tries
        # to import a flash kernel BY THAT NAME. Restored to the capture string in ``finally`` because
        # the swap has to survive for the NEXT layer's dispatch, which is looked up per layer.
        text_config._attn_implementation = saved
        try:
            return real(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        finally:
            text_config._attn_implementation = _CAPTURE_IMPL

    ALL_ATTENTION_FUNCTIONS.register(_CAPTURE_IMPL, _recorder)
    # Register a mask function too, aliased to the kernel we delegate to: ``create_causal_mask``
    # dispatches on the SAME implementation string, and an unknown key makes it return None -- which
    # would silently drop the attention mask and let a left-padded batch attend to its padding.
    mask_fn = ALL_MASK_ATTENTION_FUNCTIONS.get(saved)
    if mask_fn is not None:
        ALL_MASK_ATTENTION_FUNCTIONS.register(_CAPTURE_IMPL, mask_fn)
    text_config._attn_implementation = _CAPTURE_IMPL
    try:
        yield
    finally:
        # Restore only the config. The registrations are namespaced and inert unless
        # ``_attn_implementation`` points at them, and re-registering is idempotent.
        text_config._attn_implementation = saved


# --------------------------------------------------------------------------------------
# The expert
# --------------------------------------------------------------------------------------
def _fp32_linear(module: nn.Linear, x: Tensor) -> Tensor:
    """``module`` evaluated in fp32, whatever dtype its parameters ended up in.

    The precision-critical I/O modules (action in/out projections, timestep projection) carry their
    precision in the OP rather than in the parameter dtype, exactly as ``backbone``'s vision
    patch-embed wrapper does: a launcher may recast the whole module to bf16 after construction, and
    this keeps the velocity the loss differences produced in fp32 regardless. Autograd differentiates
    through the casts, so grads reach the parameter normally. ~0.1% of the expert's FLOPs.
    """
    return F.linear(x.float(), module.weight.float(), None if module.bias is None else module.bias.float())


class FlowMatchingExpert(nn.Module):
    """Narrow Qwen3-VL decoder stack that attends per layer over cat[backbone K/V, own K/V].

    ``head_dim`` and ``num_key_value_heads`` are inherited, not chosen: the concat requires
    element-wise agreement on the last dim and the same kv-head count. Setting ``head_dim``
    explicitly is load-bearing -- Qwen3VLTextAttention would otherwise fall back to
    ``hidden_size // num_attention_heads`` (= 64 at width 1024 with 16 heads).
    """

    def __init__(self, config: GriffinAlphaConfig, text_config) -> None:
        super().__init__()
        self.config = config
        width = config.expert_width
        self.depth = config.expert_depth or text_config.num_hidden_layers

        expert_cfg = type(text_config).from_dict(text_config.to_dict())
        expert_cfg.hidden_size = width
        expert_cfg.intermediate_size = config.expert_mlp_dim
        expert_cfg.num_attention_heads = config.expert_num_attention_heads
        expert_cfg.num_key_value_heads = BACKBONE_NUM_KV_HEADS
        expert_cfg.head_dim = BACKBONE_HEAD_DIM
        expert_cfg.num_hidden_layers = self.depth
        expert_cfg._attn_implementation = "eager"
        self.expert_config = expert_cfg

        self.layers = nn.ModuleList([Qwen3VLTextDecoderLayer(expert_cfg, i) for i in range(self.depth)])
        self.norm = Qwen3VLTextRMSNorm(width, eps=expert_cfg.rms_norm_eps)

        action_dim = config.max_action_dim
        self.action_in_proj = nn.Linear(action_dim, width)
        self.action_out_proj = nn.Linear(width, action_dim)
        # Zero-init the velocity head: the expert starts as a zero-velocity predictor, which is what
        # makes it safe to bolt onto pretrained weights.
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

        # pi0.5's adaptive RMSNorm is the ONLY time pathway: a shared MLP projects the timestep, then
        # an independent Linear(w, 3w) at each site produces (scale, shift, gate).
        self.num_modulation_sites = 2 * self.depth + 1  # two norms per layer, plus the final norm
        self.time_mlp_in = nn.Linear(width, width)
        self.site_modulation = nn.ModuleList([nn.Linear(width, 3 * width) for _ in range(self.num_modulation_sites)])
        for dense in self.site_modulation:
            _init_modulation_zero(dense, width)

        self.gradient_checkpointing = False

    def layer_dtype(self) -> torch.dtype:
        return next(self.layers.parameters()).dtype

    def embed_suffix(self, x_t: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (initial hidden state, per-layer modulation conditioning)."""
        width = self.action_in_proj.out_features
        cfg = self.config
        time_emb = create_sinusoidal_pos_embedding(timestep, width, cfg.min_period, cfg.max_period, device=x_t.device)
        # Outside any autocast: these two are the entry half of the precision-critical I/O pair (the
        # exit half is action_out_proj). A bf16 time projection would quantise a conditioning signal
        # whose useful range starts at t=0.001.
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            hidden = _fp32_linear(self.action_in_proj, x_t)
            cond = F.silu(_fp32_linear(self.time_mlp_in, time_emb))
        return hidden, cond

    def _modulate(self, x: Tensor, cond: Tensor, site: int) -> tuple[Tensor, Tensor]:
        dense = self.site_modulation[site]
        # ``cond`` leaves the fp32 timestep projection while the site denses run in the compute dtype,
        # so the one dtype handoff between the two sides happens here.
        scale, shift, gate = dense(cond.to(dense.weight.dtype)).chunk(3, dim=-1)
        x = x * (1 + scale[:, None, :].to(x.dtype)) + shift[:, None, :].to(x.dtype)
        return x, gate[:, None, :].to(x.dtype)

    def _layer_forward(self, index, hidden, prefix_kv, cos, sin, add_mask, cond):
        layer = self.layers[index]
        attn = layer.self_attn

        residual = hidden
        normed = layer.input_layernorm(hidden)
        normed, gate = self._modulate(normed, cond, 2 * index)
        bsize, q_len, _ = normed.shape
        heads = (bsize, q_len, -1, attn.head_dim)
        query = attn.q_norm(attn.q_proj(normed).view(heads)).transpose(1, 2)
        key = attn.k_norm(attn.k_proj(normed).view(heads)).transpose(1, 2)
        value = attn.v_proj(normed).view(heads).transpose(1, 2)
        query, key = qwen_apply_rotary_pos_emb(query, key, cos, sin)

        prefix_key, prefix_value = prefix_kv
        key = torch.cat([prefix_key.to(key.dtype), key], dim=2)
        value = torch.cat([prefix_value.to(value.dtype), value], dim=2)
        attn_out, _ = modeling_qwen3_vl.eager_attention_forward(
            attn, query, key, value, add_mask, dropout=0.0, scaling=attn.scaling
        )
        attn_out = attn.o_proj(attn_out.reshape(bsize, q_len, -1))
        hidden = residual + gate * attn_out

        residual = hidden
        mlp_in = layer.post_attention_layernorm(hidden)
        mlp_in, gate = self._modulate(mlp_in, cond, 2 * index + 1)
        mlp_out = layer.mlp(mlp_in)
        return residual + gate * mlp_out

    def forward(self, x_t, timestep, prefix_kv, cos, sin, add_mask):
        """x_t: (B, chunk, action_dim) -> v_t (B, chunk, action_dim), fp32."""
        hidden, cond = self.embed_suffix(x_t, timestep)
        hidden = hidden.to(self.layer_dtype())
        for index in range(self.depth):
            if self.gradient_checkpointing and self.training:
                hidden = torch.utils.checkpoint.checkpoint(
                    self._layer_forward,
                    index, hidden, prefix_kv[index], cos, sin, add_mask, cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden = self._layer_forward(index, hidden, prefix_kv[index], cos, sin, add_mask, cond)
        hidden = self.norm(hidden)
        hidden, _ = self._modulate(hidden, cond, 2 * self.depth)
        # The exit half of the fp32 I/O pair: the velocity the flow loss differences against ``u_t``
        # is produced here, so this projection stays out of autocast whichever precision mode is in
        # force, and the matmul is promoted rather than the tensor merely relabelled.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return _fp32_linear(self.action_out_proj, hidden)


# --------------------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------------------
class GriffinAlphaPolicy(GriffinAlphaBackbonePolicy):
    """Griffin Alpha-S with the flow-matching action expert (the default head)."""

    config_class = GriffinAlphaConfig
    name = "griffin_alpha"

    config: GriffinAlphaConfig

    def __init__(
        self,
        config: GriffinAlphaConfig,
        qwen3vl_model: Qwen3VLForConditionalGeneration | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(config, qwen3vl_model, *args, **kwargs)

        text_config = self.model.config.text_config
        if text_config.head_dim != BACKBONE_HEAD_DIM or text_config.num_key_value_heads != BACKBONE_NUM_KV_HEADS:
            raise ValueError(
                f"This head inherits head_dim/num_key_value_heads from the backbone so the expert's "
                f"K/V concatenate with it, but the backbone reports head_dim="
                f"{text_config.head_dim}, num_key_value_heads={text_config.num_key_value_heads} "
                f"(expected {BACKBONE_HEAD_DIM}/{BACKBONE_NUM_KV_HEADS})."
            )

        self.expert = FlowMatchingExpert(config, text_config)
        # Cast the whole expert to the backbone's compute dtype so the per-layer K/V concat stays
        # cheap, THEN pull the precision-critical projections back to fp32. Cast-then-exempt rather
        # than cast-the-parts: the adaptive-RMSNorm site denses are a quarter of the head and belong
        # in the compute dtype; only the timestep projection and the action I/O (~1.1M params) are
        # precision-critical.
        model_dtype = next(self.model.parameters()).dtype
        self._expert_autocast_dtype: torch.dtype | None = None
        if model_dtype in (torch.bfloat16, torch.float16):
            if self.config.expert_param_dtype == "backbone":
                self.expert.to(model_dtype)
                for module in (self.expert.action_in_proj, self.expert.action_out_proj, self.expert.time_mlp_in):
                    module.to(torch.float32)
            else:
                # "float32": parameters stay fp32 (so a plain AdamW keeps fp32 moments) and the
                # matmuls are cast at run time by the autocast in _expert_step.
                self._expert_autocast_dtype = model_dtype
        self.expert.gradient_checkpointing = bool(config.gradient_checkpointing)

        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad_(False)

    # -- optimizer ------------------------------------------------------------------------------
    def get_optim_params(self):
        """Two param groups: the pretrained backbone at ``optimizer_lr``, the expert at ``expert_optimizer_lr``."""
        expert_params = [p for p in self.expert.parameters() if p.requires_grad]
        expert_ids = {id(p) for p in expert_params}
        backbone_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in expert_ids]
        groups = [
            {"params": backbone_params, "lr": self.config.optimizer_lr},
            {"params": expert_params, "lr": self.config.expert_optimizer_lr},
        ]
        return [group for group in groups if group["params"]]

    # -- geometry -------------------------------------------------------------------------------
    def _prefix_geometry(self, model_inputs: dict[str, Tensor]):
        """Returns (mRoPE position ids, visible prefix mask, next text position per sample).

        ``visible`` is just the attention mask, because the input step emits ONE prompt form for both
        regimes (inference form, no assistant turn holding actions). Train and inference therefore see
        byte-identical prefixes and the position offset needs no correction.
        """
        attention_mask = model_inputs["attention_mask"]
        mm_token_type_ids = model_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            raise KeyError(
                "batch is missing 'mm_token_type_ids', which Qwen3-VL's get_rope_index indexes per "
                "sample to place image tokens on the mRoPE grid. The Qwen processor returns it "
                "alongside input_ids; check it survived _filter_model_inputs (MODEL_INPUT_KEYS)."
            )
        position_ids, _ = self.model.model.get_rope_index(
            model_inputs["input_ids"],
            mm_token_type_ids,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )  # (3, B, P)

        if model_inputs.get("labels") is not None:
            # Masking the turn off would shift every action-token position; not masking it would leak
            # the answer into the regression target. Neither fallback is safe enough to pick quietly.
            raise ValueError(
                "the batch carries `labels`, which means the preprocessor is emitting a FAST assistant "
                "turn. The flow head needs one prompt form for both regimes -- build the pipelines with "
                "make_griffin_alpha_pre_post_processors."
            )
        visible = attention_mask.bool()
        return position_ids, visible, next_visible_position(position_ids, visible)

    def _suffix_rope(self, next_pos: Tensor, chunk: int):
        bsize = next_pos.shape[0]
        offsets = torch.arange(chunk, device=next_pos.device)
        suffix = (next_pos[:, None] + offsets[None, :])[None].expand(3, bsize, chunk)
        dummy = torch.zeros(1, 1, 1, device=next_pos.device, dtype=self.expert.layer_dtype())
        return self.model.model.language_model.rotary_emb(dummy, suffix)

    _expert_mask = staticmethod(expert_attention_mask)

    def _run_prefix(self, model_inputs: dict[str, Tensor], *, with_grad: bool):
        """One backbone forward; returns (outputs, per-layer K/V store, visible, next_pos)."""
        store: dict[int, tuple[Tensor, Tensor]] = {}
        text_config = self.model.config.text_config
        context = contextlib.nullcontext() if with_grad else torch.no_grad()
        # ``labels`` is filtered out defensively (passing it would run a cross-entropy nobody reads).
        # ``logits_to_keep=1`` cuts the lm_head projection from the whole sequence to one position: at
        # batch 32, S=500 and a 154k vocabulary the full projection is ~5 GB of bf16 logits per step.
        forward_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
        # The geometry runs BEFORE the forward and its position ids go INTO it, so the expert's
        # positions and the backbone's are one tensor rather than two independently derived copies.
        # Passing (3, B, P) reproduces Qwen3VLModel.forward's default path exactly.
        position_ids, visible, next_pos = self._prefix_geometry(model_inputs)
        with context, capture_backbone_kv(text_config, store):
            outputs = self.model(**forward_inputs, position_ids=position_ids, logits_to_keep=1)
        if len(store) != self.expert.depth:
            raise RuntimeError(
                f"captured {len(store)} backbone layers but the expert has {self.expert.depth}; "
                "per-layer lockstep requires expert_depth == backbone depth."
            )
        return outputs, store, visible, next_pos

    def _suffix_conditioning(self, visible: Tensor, next_pos: Tensor, chunk: int):
        """(cos, sin, additive mask): a pure function of the prefix and the chunk length, built once per chunk."""
        cos, sin = self._suffix_rope(next_pos, chunk)
        return cos, sin, self._expert_mask(visible, chunk, self.expert.layer_dtype())

    def _expert_autocast(self):
        """bf16 matmuls over fp32 expert weights (``expert_param_dtype="float32"``), scoped to the EXPERT.

        Scoped deliberately: an autocast spanning the backbone would re-cast the inputs of the vision
        patch-embed fp32 wrapper back to bf16 (conv3d is on autocast's cast list). A no-op under
        ``expert_param_dtype="backbone"`` and on CPU.
        """
        dtype = self._expert_autocast_dtype
        device = next(self.expert.parameters()).device
        if dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=dtype)

    def _expert_step(self, x_t, timestep, store, visible, next_pos):
        with self._expert_autocast():
            cos, sin, add_mask = self._suffix_conditioning(visible, next_pos, x_t.shape[1])
            return self.expert(x_t, timestep, store, cos, sin, add_mask)

    # -- masks ----------------------------------------------------------------------------------
    def _dim_mask(self, batch: dict, bsize: int, action_dim: int, device) -> Tensor:
        """(B, 1, action_dim): padded action dims must contribute exactly zero to the loss."""
        mask = torch.ones(bsize, 1, action_dim, dtype=torch.bool, device=device)
        dims = batch.get("n_action_dims")
        if dims is None:
            return mask
        if isinstance(dims, Tensor):
            dims = dims.tolist()
        if not isinstance(dims, (list, tuple)):
            dims = [dims] * bsize
        for i, real in enumerate(dims):
            if real is not None and int(real) < action_dim:
                mask[i, :, int(real) :] = False
        return mask

    @staticmethod
    def _step_mask(batch: dict, bsize: int, chunk: int, device) -> Tensor:
        """(B, chunk, 1): drop chunk steps that are dataset padding past the episode end.

        lerobot marks those steps with ``action_is_pad``; when the batch carries no such mask every
        step is kept.
        """
        is_pad = batch.get(f"{ACTION}_is_pad")
        if not isinstance(is_pad, Tensor):
            return torch.ones(bsize, chunk, 1, dtype=torch.bool, device=device)
        if is_pad.shape[-1] < chunk:
            raise ValueError(
                f"{ACTION}_is_pad covers {is_pad.shape[-1]} steps but the chunk is {chunk}. A pad mask "
                "from before a resample cannot be sliced to the resampled length; resample it with the "
                "action, or drop it in the dataset transform."
            )
        return (~is_pad.bool()).to(device)[:, :chunk, None]

    # -- training -------------------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        model_inputs = self._filter_model_inputs(batch)
        _, store, visible, next_pos = self._run_prefix(model_inputs, with_grad=not cfg.freeze_backbone)

        if ACTION not in batch:
            raise KeyError(
                "batch is missing 'action'; the flow head needs the normalized action chunk. Build the "
                "pipelines with make_griffin_alpha_pre_post_processors, whose input step emits it "
                "alongside the VLM inputs."
            )

        action_dim = cfg.max_action_dim
        raw_dim = batch[ACTION].shape[-1]
        if raw_dim > action_dim:
            raise ValueError(
                f"the action chunk is {raw_dim}-D but the expert's head is {action_dim}-D "
                f"(action_dim_override={cfg.action_dim_override}). The head width is baked into the "
                "weights; a wider embodiment needs a base rebuilt with a wider head."
            )
        actions = pad_vector(batch[ACTION].to(torch.float32), action_dim)
        bsize, chunk = actions.shape[0], actions.shape[1]
        device = actions.device

        noise = torch.randn(actions.shape, dtype=torch.float32, device=device)
        time = sample_beta(cfg.time_sampling_beta_alpha, cfg.time_sampling_beta_beta, bsize, device)
        time = (time * cfg.time_sampling_scale + cfg.time_sampling_offset).to(torch.float32)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        velocity = self._expert_step(x_t, time, store, visible, next_pos)

        keep = self._step_mask(batch, bsize, chunk, device) & self._dim_mask(batch, bsize, action_dim, device)
        keep = keep.expand_as(velocity)
        flow_loss = (((velocity - u_t) ** 2) * keep).sum() / keep.sum().clamp_min(1)

        loss = flow_loss
        return loss, {"flow_loss": flow_loss.item(), "loss": loss.item()}

    # -- inference ------------------------------------------------------------------------------
    def _integrate(self, store, visible, next_pos, noise, num_steps):
        """Euler-integrate the flow from t=1 (noise) to t=0 (actions)."""
        # cos/sin and the additive mask depend only on the prefix and the chunk length, never on t, so
        # they are built once for the whole trajectory rather than num_steps times.
        cos, sin, add_mask = self._suffix_conditioning(visible, next_pos, noise.shape[1])
        x_t = noise
        dt = -1.0 / num_steps
        for step in range(num_steps):
            time = torch.full((x_t.shape[0],), 1.0 + step * dt, dtype=torch.float32, device=x_t.device)
            velocity = self.expert(x_t, time, store, cos, sin, add_mask)
            x_t = x_t + dt * velocity
        return x_t

    def _finalize(self, x_t: Tensor, batch: dict) -> Tensor:
        real_dim = self.config.output_features[ACTION].shape[-1]
        from_output_features = True
        dims = batch.get("n_action_dims")
        if isinstance(dims, Tensor):
            dims = dims.tolist()
        if isinstance(dims, (list, tuple)) and dims and dims[0] is not None:
            # One width for the whole batch: a mixed-embodiment batch is refused rather than silently
            # truncated to row 0's width.
            distinct = {int(d) for d in dims if d is not None}
            if len(distinct) > 1:
                raise ValueError(
                    f"n_action_dims has mixed widths {sorted(distinct)} in one batch; the returned chunk "
                    "is a single tensor and cannot carry per-row widths. Split the batch by embodiment."
                )
            real_dim = distinct.pop()
            from_output_features = False

        if from_output_features and self.config.se3_segment_start_idxs:
            # With SE(3) segments the model faces a WIDER vector than output_features' raw width
            # (xyz + rot6d per matrix), and it is the model-facing one the postprocessor consumes.
            # Truncating to the raw width here would corrupt every rotation silently, so refuse.
            raise NotImplementedError(
                f"se3_segment_start_idxs={self.config.se3_segment_start_idxs} means the model-facing "
                f"action width differs from output_features' raw {real_dim}, and this path has no "
                "n_action_dims to read the model-facing width from. The SE(3) recipe is not supported "
                "with the flow head at inference."
            )
        return x_t[:, :, :real_dim]

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, num_steps: int | None = None, **kwargs
    ) -> Tensor:
        """(B, chunk, real_action_dim).

        ``noise`` lets a caller pass its own (seeded) starting noise of shape
        ``(B, horizon, max_action_dim)``; ``num_steps`` overrides ``config.num_inference_steps``.
        """
        self.eval()
        model_inputs = self._filter_model_inputs(batch)
        model_inputs.pop("labels", None)
        _, store, visible, next_pos = self._run_prefix(model_inputs, with_grad=False)

        chunk = self.config.horizon
        action_dim = self.config.max_action_dim
        bsize = visible.shape[0]
        if noise is None:
            noise = torch.randn(bsize, chunk, action_dim, dtype=torch.float32, device=visible.device)
        else:
            noise = noise.to(device=visible.device, dtype=torch.float32)
            if noise.shape[0] != bsize:
                raise ValueError(f"noise has {noise.shape[0]} rows but the batch has {bsize}.")

        x_t = self._integrate(store, visible, next_pos, noise, num_steps or self.config.num_inference_steps)
        return self._resample_chunk(self._finalize(x_t, batch))
