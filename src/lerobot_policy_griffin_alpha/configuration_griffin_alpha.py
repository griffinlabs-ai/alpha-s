"""Configuration for ``griffin_alpha``: Griffin Alpha-S with the flow-matching action expert.

This is the default head. The FAST-token variant is ``griffin_alpha_fast``
(``configuration_griffin_alpha_fast.py``); both share every backbone/prompt/action-space field
through ``GriffinAlphaBackboneConfig``.

Naming is load-bearing -- lerobot resolves the whole triple from this class
(``lerobot/policies/factory.py``): ``GriffinAlphaConfig`` -> policy class ``GriffinAlphaPolicy`` in
``modeling_griffin_alpha`` (this module's name with ``configuration_`` -> ``modeling_``), and the
registered type string ``griffin_alpha`` -> factory ``make_griffin_alpha_pre_post_processors`` in
``processor_griffin_alpha``. Renaming any one of the three without the others breaks
``make_policy`` at runtime, not import time.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION

from .backbone import MAX_ACTION_DIM, GriffinAlphaBackboneConfig

# Inherited from Qwen3-VL-4B and NOT free parameters. The expert's own K/V must concatenate with
# the backbone's (B, 8, S, 128) along the sequence axis, which pins head_dim and the kv-head count;
# rope_theta / mrope_section are then inherited wholesale so the expert's queries live in the same
# rotational frame as the backbone's keys. Captured K/V are post-RoPE, so there is no "project them
# into a different head geometry" escape hatch either.
BACKBONE_HEAD_DIM = 128
BACKBONE_NUM_KV_HEADS = 8

# The expert's default I/O width: the canonical 32-wide action vector the bases were pre-trained
# with. Every embodiment's actions occupy the leading slots, the rest is zero padding, and the real
# per-sample width travels as ``n_action_dims`` -- which is exactly what the flow loss masks on. So a
# head of this width can take any embodiment, and one narrowed to a single embodiment's dim cannot.
CANONICAL_ACTION_DIMS = MAX_ACTION_DIM

# Where the expert's PARAMETERS live. Strings, not torch.dtypes, because this is serialized into
# every checkpoint's config.json.
EXPERT_PARAM_DTYPES = frozenset({"float32", "backbone"})


@PreTrainedConfig.register_subclass("griffin_alpha")
@dataclass
class GriffinAlphaConfig(GriffinAlphaBackboneConfig):
    """Config for the flow-matching expert over the Qwen3-VL-4B backbone.

    Inherits every backbone field (image-token band, proprio prompt, freeze knobs, action-space
    transform, optimizer/scheduler) so the prompt and the action space are identical to the FAST
    variant's; only the head differs.
    """

    # ---- Mixture-of-transformers action expert ------------------------------------------------
    # Width 1024 is what every pi generation uses (pi0, pi0.5, pi0.6), independent of backbone width;
    # the invariants across those three are width 1024, mlp/width = 4, depth = backbone depth, and
    # head_dim / kv heads inherited from the backbone. Per layer at these dims: attn 6.29M (q 1024x2048,
    # k/v 1024x1024, o 2048x1024) + mlp 12.58M (3 x 1024 x 4096, SwiGLU) = 18.87M; x36 = 679.5M. With
    # the adaptive-RMSNorm time conditioning (230.9M) and the I/O projections the head totals 910.5M.
    expert_width: int = 1024
    expert_mlp_dim: int = 4096
    # None binds expert depth to the backbone's at build time (36 for Qwen3-VL-4B). A different depth
    # would need a layer-stride mapping and stops being per-layer lockstep.
    expert_depth: int | None = None
    # Must be a multiple of BACKBONE_NUM_KV_HEADS (GQA requires num_heads % num_kv_heads == 0). 16 q
    # heads over the 8 inherited kv heads is a grouping of 2, so the concatenated K/V get repeated
    # inside attention.
    expert_num_attention_heads: int = 16

    # ---- Time conditioning -------------------------------------------------------------------
    # pi0.5's form: a separate MLP projects the flow time and adaptive RMSNorm injects it at EVERY
    # normalization site of the expert (two per layer plus the final norm). It is the only time
    # pathway; there is no input concatenation.

    # ---- Flow matching (rectified flow; constants shared with pi0 / pi0.5) ---------------------
    # ``x_t = t*eps + (1-t)*a``, target ``u_t = eps - a``, ``t ~ offset + scale * Beta(alpha, beta)``.
    # ``num_inference_steps`` is an INFERENCE-TIME knob, not a training commitment: the objective is
    # trained over the whole t interval, so one checkpoint integrates at any step count. On LIBERO one
    # Euler step matched ten; on a bimanual robot ten was clearly better. Measure before lowering it.
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # ---- Backbone handling --------------------------------------------------------------------
    # False (the default): joint training of backbone and expert -- the expert's gradient reaches the
    # backbone through the captured K/V, and the two get separate learning rates (``optimizer_lr`` for
    # the backbone, ``expert_optimizer_lr`` for the expert; see ``get_optim_params``). This is how the
    # released LIBERO checkpoint was fine-tuned.
    #
    # True: the backbone is a frozen feature extractor (``requires_grad=False`` on every backbone
    # parameter, prefix forward under ``no_grad``) and the expert is the only trainable module. This is
    # how the released base's expert was pre-trained, and the cheaper option when data is scarce or
    # memory is tight.
    freeze_backbone: bool = False

    # Put the ground-truth subtask (when the dataset provides one) into the prefix the expert
    # conditions on, by ending the prompt with a truncated assistant turn:
    #
    #     with a subtask:  <|im_start|>assistant\nsubtask: <text>\naction:_
    #     without one:     <|im_start|>assistant\naction:_
    #
    # Both are strict token prefixes of the pre-training sequence, so the expert takes over exactly
    # where pre-training's first action token sat. What it commits to: the subtask does not exist at
    # inference, so an inference batch always takes the no-subtask branch -- a form the model trained
    # on, but a minority one. The input step warns about this once per process. The released
    # checkpoints were trained with this on.
    condition_on_subtask: bool = True

    # ---- Head width -----------------------------------------------------------------------------
    # An int PINS the head width; ``None`` derives it from the ACTION output feature. Pinning to the
    # canonical 32 is the default because lerobot's ``make_policy`` overwrites ``output_features`` from
    # the dataset's raw action shape at load time, so a derived head would be re-sized by whichever
    # dataset a run happens to load. The loss masks the unused slots per sample via ``n_action_dims``.
    action_dim_override: int | None = CANONICAL_ACTION_DIMS

    # The expert is trained from scratch while the backbone is pretrained, so they get separate
    # learning rates (two param groups in ``get_optim_params``).
    expert_optimizer_lr: float = 5e-5

    # ---- Precision ------------------------------------------------------------------------------
    # "backbone": cast the whole expert to the backbone's compute dtype (bf16 on the released
    # checkpoints) except the action I/O projections and the timestep projection, which stay fp32
    # because the velocity the loss differences is produced there. Matches the released checkpoints.
    # "float32": keep every expert parameter in fp32 (so a plain AdamW keeps fp32 moments) and run its
    # matmuls under an autocast scoped to the expert. Costs memory; consider it for long from-scratch
    # expert training without a launcher that keeps fp32 master weights.
    expert_param_dtype: str = "backbone"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.expert_depth is not None and self.expert_depth <= 0:
            raise ValueError(f"expert_depth must be positive or None, got {self.expert_depth}")
        if self.expert_num_attention_heads % BACKBONE_NUM_KV_HEADS != 0:
            raise ValueError(
                f"expert_num_attention_heads ({self.expert_num_attention_heads}) must be a "
                f"multiple of the backbone's {BACKBONE_NUM_KV_HEADS} kv heads: the expert's K/V "
                "are concatenated with the backbone's, so GQA grouping must divide evenly."
            )
        if self.expert_width % 2 != 0:
            raise ValueError(
                f"expert_width must be even ({self.expert_width}); the sinusoidal time embedding "
                "is built at this width and needs an even dimension."
            )
        if self.expert_param_dtype not in EXPERT_PARAM_DTYPES:
            raise ValueError(
                f"expert_param_dtype must be one of {sorted(EXPERT_PARAM_DTYPES)}, got "
                f"{self.expert_param_dtype!r}. It is a string rather than a torch.dtype because it "
                "is serialized into every checkpoint's config.json."
            )

    @property
    def max_action_dim(self) -> int:
        """Expert head width: pinned by action_dim_override, else the ACTION output-feature shape."""
        if self.action_dim_override is not None:
            return self.action_dim_override
        return self.output_features[ACTION].shape[-1]
