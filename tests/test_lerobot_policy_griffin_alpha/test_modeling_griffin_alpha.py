"""Unit tests for the flow-matching expert and its two silent-failure contracts.

These run on CPU with a tiny expert and stubbed backbone K/V — no checkpoint, no HF download. What
they cover is exactly the logic that is new here; the backbone itself is upstream code.

Deliberately NOT covered yet (needs a real checkpoint, so it belongs in a GPU smoke run):
the registration triple resolving through ``make_policy``, and that flow gradients reach backbone
parameters iff the backbone is unfrozen.
"""

import dataclasses

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from lerobot_policy_griffin_alpha import GriffinAlphaConfig, FlowMatchingExpert
from lerobot_policy_griffin_alpha.configuration_griffin_alpha import (
    BACKBONE_HEAD_DIM,
    BACKBONE_NUM_KV_HEADS,
    CANONICAL_ACTION_DIMS,
)
from lerobot_policy_griffin_alpha.modeling_griffin_alpha import (
    MASK_NEG,
    expert_attention_mask,
    next_visible_position,
)

CHUNK = 4
ACTION_DIM = 8
PREFIX = 6
BATCH = 2


def tiny_text_config() -> Qwen3VLTextConfig:
    """A 2-layer stand-in for Qwen3-VL-4B's text stack, keeping the inherited head geometry."""
    return Qwen3VLTextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=BACKBONE_NUM_KV_HEADS,
        num_key_value_heads=BACKBONE_NUM_KV_HEADS,
        head_dim=BACKBONE_HEAD_DIM,
        vocab_size=128,
    )


def flow_config(**overrides) -> GriffinAlphaConfig:
    kwargs = dict(
        horizon=CHUNK,
        n_action_steps=CHUNK,
        expert_width=64,
        expert_mlp_dim=128,
        expert_depth=2,
        expert_num_attention_heads=BACKBONE_NUM_KV_HEADS,
        action_dim_override=ACTION_DIM,
    )
    kwargs.update(overrides)
    return GriffinAlphaConfig(**kwargs)


def build(config) -> tuple[FlowMatchingExpert, tuple[torch.Tensor, torch.Tensor]]:
    text_config = tiny_text_config()
    expert = FlowMatchingExpert(config, text_config)
    rotary = Qwen3VLTextRotaryEmbedding(expert.expert_config)
    positions = torch.arange(CHUNK)[None, :].expand(3, BATCH, CHUNK)
    cos, sin = rotary(torch.zeros(1, 1, 1), positions)
    return expert, (cos, sin)


def stub_prefix_kv(depth: int) -> dict:
    torch.manual_seed(0)
    return {
        i: (
            torch.randn(BATCH, BACKBONE_NUM_KV_HEADS, PREFIX, BACKBONE_HEAD_DIM),
            torch.randn(BATCH, BACKBONE_NUM_KV_HEADS, PREFIX, BACKBONE_HEAD_DIM),
        )
        for i in range(depth)
    }


def run_expert(config, x_t=None, time=None):
    expert, (cos, sin) = build(config)
    visible = torch.ones(BATCH, PREFIX, dtype=torch.bool)
    mask = expert_attention_mask(visible, CHUNK, torch.float32)
    if x_t is None:
        x_t = torch.randn(BATCH, CHUNK, ACTION_DIM)
    if time is None:
        time = torch.full((BATCH,), 0.7)
    return expert, expert(x_t, time, stub_prefix_kv(expert.depth), cos, sin, mask)


# ---------------------------------------------------------------- shapes and the K/V concat
def test_expert_attends_over_prefix_plus_suffix():
    """The concat is the only place the two stacks meet; a wrong kv-head count fails here."""
    config = flow_config()
    expert, (cos, sin) = build(config)
    attn = expert.layers[0].self_attn
    assert attn.head_dim == BACKBONE_HEAD_DIM
    assert attn.k_proj.out_features == BACKBONE_NUM_KV_HEADS * BACKBONE_HEAD_DIM
    # residual stream is narrower than the attention interior — intended, see the dimension audit
    assert attn.q_proj.in_features == config.expert_width
    assert attn.q_proj.out_features == config.expert_num_attention_heads * BACKBONE_HEAD_DIM


# ---------------------------------------------------------------- initialization
def test_velocity_is_exactly_zero_at_init():
    """Zero-init action_out_proj: the expert starts as a zero-velocity predictor, which is what
    makes it safe to attach to pretrained weights."""
    _, velocity = run_expert(flow_config())
    assert torch.count_nonzero(velocity) == 0


def test_time_reaches_the_loss():
    """The invariant that survives zero-init: the time pathway must receive gradient.

    π0.5's adaRMS dense is zero-initialised (scale=0, shift=0, gate=1), so with the input concat off
    the velocity is genuinely **t-independent at step 0** — same structure as DiT's adaLN-Zero. That
    is benign only because the modulation still gets a gradient and learns to use t, which is what
    this asserts. Guards against the time signal being silently disconnected.
    """
    config = flow_config()
    expert, (cos, sin) = build(config)
    torch.nn.init.normal_(expert.action_out_proj.weight, std=0.02)
    mask = expert_attention_mask(torch.ones(BATCH, PREFIX, dtype=torch.bool), CHUNK, torch.float32)

    velocity = expert(torch.randn(BATCH, CHUNK, ACTION_DIM), torch.full((BATCH,), 0.7),
                      stub_prefix_kv(expert.depth), cos, sin, mask)
    velocity.pow(2).mean().backward()

    # The zero-init chain means the pathway wakes up in two stages, so assert the right stage.
    # Stage 1: the per-site denses get gradient (dL/dW = dL/dmodulation . cond^T, and cond != 0).
    site_grads = sum(
        dense.weight.grad.abs().sum() for dense in expert.site_modulation
        if dense.weight.grad is not None
    )
    assert site_grads > 0, "the per-layer modulation receives no gradient"
    # Stage 2 is NOT yet reachable: with W = 0, dL/dcond = W^T . dL/dmodulation = 0, so the shared
    # time MLP upstream is still frozen. It starts learning once the site weights move off zero.
    assert expert.time_mlp_in.weight.grad.abs().sum() == 0

    expert.zero_grad()
    for dense in expert.site_modulation:
        torch.nn.init.normal_(dense.weight, std=0.02)
    velocity = expert(torch.randn(BATCH, CHUNK, ACTION_DIM), torch.full((BATCH,), 0.7),
                      stub_prefix_kv(expert.depth), cos, sin, mask)
    velocity.pow(2).mean().backward()
    assert expert.time_mlp_in.weight.grad.abs().sum() > 0, "the time MLP never connects"


# ---------------------------------------------------------------- contract 1: the mask
def test_masked_prefix_tokens_are_unreachable():
    visible = torch.tensor([[True, True, False, False, True, True]] * BATCH)
    mask = expert_attention_mask(visible, CHUNK, torch.float32)
    assert mask.shape == (BATCH, 1, CHUNK, PREFIX + CHUNK)
    assert (mask[:, :, :, 2] == MASK_NEG).all()
    assert (mask[:, :, :, 3] == MASK_NEG).all()
    assert (mask[:, :, :, [0, 1, 4, 5]] == 0).all()
    # the action block is fully bidirectional, never causal
    assert (mask[:, :, :, PREFIX:] == 0).all()


def test_assistant_turn_is_invisible_to_the_expert():
    """The FAST answer must not leak into the regression target. ``labels != -100`` marks the
    assistant turn, so the visible mask is attention_mask & (labels == -100)."""
    attention_mask = torch.ones(1, PREFIX, dtype=torch.bool)
    labels = torch.full((1, PREFIX), -100)
    labels[0, -2:] = 42  # the assistant turn carrying the FAST tokens
    visible = attention_mask & (labels == -100)
    mask = expert_attention_mask(visible, CHUNK, torch.float32)
    assert (mask[0, 0, :, PREFIX - 2 : PREFIX] == MASK_NEG).all()
    assert (mask[0, 0, :, : PREFIX - 2] == 0).all()


# ---------------------------------------------------------------- contract 2: positions
def test_action_positions_come_from_the_visible_prefix():
    """Train and inference must agree. The training sequence is longer by the assistant turn, so a
    length-derived offset would disagree; a visible-derived one does not."""
    infer_positions = torch.arange(4)[None, None, :].expand(3, 1, 4)
    infer_visible = torch.ones(1, 4, dtype=torch.bool)

    train_positions = torch.arange(7)[None, None, :].expand(3, 1, 7)
    train_visible = torch.ones(1, 7, dtype=torch.bool)
    train_visible[0, 4:] = False  # the assistant turn

    assert next_visible_position(infer_positions, infer_visible).item() == 4
    assert next_visible_position(train_positions, train_visible).item() == 4
    # the naive version disagrees, which is the bug being guarded against
    assert train_positions.max().item() + 1 == 7


# ---------------------------------------------------------------- config guards
def test_kv_head_divisibility_is_enforced():
    with pytest.raises(ValueError, match="multiple of"):
        flow_config(expert_num_attention_heads=BACKBONE_NUM_KV_HEADS - 1)


def test_action_dim_override_pins_the_head_width():
    """Without the override lerobot's make_policy would size the head from the dataset action
    shape, which breaks cross-embodiment loading and the SE(3) recipe outright."""
    assert flow_config(action_dim_override=ACTION_DIM).max_action_dim == ACTION_DIM
    # None is the opt-in "derive from the dataset" path, so it must actually follow
    # output_features rather than coincide with the canonical default.
    derived = flow_config(
        action_dim_override=None,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM + 3,))},
    )
    assert derived.max_action_dim == ACTION_DIM + 3
    assert ACTION_DIM + 3 != CANONICAL_ACTION_DIMS, "the derived width must differ from the default"


def test_the_config_default_head_width_is_canonical():
    """The canonical width is a property of the TYPE, not of ``convert.py``'s argparse.

    It defaulted to None here while only the converter's ``--action_dim`` supplied 32, so a config
    built by hand, by a test, or by any other caller silently fell back to whatever action width the
    loaded dataset reported — an embodiment-shaped head presenting as a canonical one.
    """
    field = next(
        f for f in dataclasses.fields(GriffinAlphaConfig) if f.name == "action_dim_override"
    )
    assert field.default == CANONICAL_ACTION_DIMS
    assert GriffinAlphaConfig().max_action_dim == CANONICAL_ACTION_DIMS


# ---------------------------------------------------------------- the naming contract
class TestRegistrationContract:
    """lerobot resolves policy + processors from the config class by string-replacing the module
    name, and fails at *runtime* rather than import if any of the three names drift. Mirrors
    ``test_configuration_griffin_alpha_qwen3_vl.py::TestRegistrationContract``. This is also what
    proves a standalone folder is allowed: nothing here depends on living inside the plugin.
    """

    TYPE = "griffin_alpha"

    def test_type_string_is_registered(self):
        from lerobot.configs.policies import PreTrainedConfig

        assert self.TYPE in PreTrainedConfig.get_known_choices()
        assert PreTrainedConfig.get_choice_class(self.TYPE) is GriffinAlphaConfig

    def test_type_is_distinct_from_the_fast_only_policy(self):
        """The type string is baked into every saved checkpoint; sharing one would make existing
        FAST-only checkpoints deserialize into this policy."""
        from lerobot_policy_griffin_alpha.configuration_griffin_alpha_fast import (
            GriffinAlphaFASTConfig,
        )

        assert GriffinAlphaFASTConfig.get_choice_name(GriffinAlphaFASTConfig) != self.TYPE

    def test_policy_class_resolves(self):
        from lerobot.policies.factory import get_policy_class

        from lerobot_policy_griffin_alpha.modeling_griffin_alpha import (
            GriffinAlphaPolicy,
        )

        assert get_policy_class(self.TYPE) is GriffinAlphaPolicy

    def test_processor_factory_resolves(self):
        import importlib

        from lerobot_policy_griffin_alpha.processor_griffin_alpha import (
            make_griffin_alpha_pre_post_processors,
        )

        module_path = GriffinAlphaConfig.__module__.replace("configuration_", "processor_")
        factory = getattr(importlib.import_module(module_path), f"make_{self.TYPE}_pre_post_processors")
        assert factory is make_griffin_alpha_pre_post_processors

    def test_processor_step_registry_name_is_new(self):
        """Registry names are serialized into saved processor JSON — never rename one in place."""
        from lerobot.processor import ProcessorStepRegistry

        from lerobot_policy_griffin_alpha.processor_griffin_alpha import (
            GriffinAlphaInputProcessorStep,
        )

        registered = ProcessorStepRegistry.get("griffinlabs/griffin_alpha_input")
        assert registered is GriffinAlphaInputProcessorStep
        assert (
            ProcessorStepRegistry.get("griffinlabs/griffin_alpha_fast_input")
            is not GriffinAlphaInputProcessorStep
        )


# ---------------------------------------------------------------- GQA grouping and real dims
def test_grouped_query_attention_over_concatenated_kv():
    """16 q heads over the 8 inherited kv heads is a grouping of 2, so ``repeat_kv`` fires inside
    attention — a different path from the 1:1 case the other tests use. The concatenated K/V carry
    8 heads (pre-repeat), which is what the kernel expects."""
    config = flow_config(expert_width=256, expert_mlp_dim=512, expert_num_attention_heads=16)
    expert, (cos, sin) = build(config)
    attn = expert.layers[0].self_attn
    assert attn.num_key_value_groups == 2
    assert attn.q_proj.out_features == 16 * BACKBONE_HEAD_DIM
    assert attn.k_proj.out_features == BACKBONE_NUM_KV_HEADS * BACKBONE_HEAD_DIM

    mask = expert_attention_mask(torch.ones(BATCH, PREFIX, dtype=torch.bool), CHUNK, torch.float32)
    velocity = expert(torch.randn(BATCH, CHUNK, ACTION_DIM), torch.full((BATCH,), 0.5),
                      stub_prefix_kv(expert.depth), cos, sin, mask)
    assert velocity.shape == (BATCH, CHUNK, ACTION_DIM)


SHIPPED = dict(
    horizon=50, n_action_steps=50, expert_width=1024, expert_mlp_dim=4096, expert_depth=36,
    expert_num_attention_heads=16, action_dim_override=32,
)

