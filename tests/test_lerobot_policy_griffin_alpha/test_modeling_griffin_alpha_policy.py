"""End-to-end tests on a *real* but tiny Qwen3-VL backbone — no checkpoint, no download, CPU.

This is the layer the pure-expert tests cannot reach. It exercises the machinery that only exists
once a genuine Qwen3-VL is in the loop: the K/V capture (an attention-implementation swap that
either fires on every layer or silently does not), ``get_rope_index`` and the visible-prefix offset,
the additive mask against a real prefix length, and above all **whether the flow gradient actually
stops at the backbone**.

Note on the fixture's pathology, which is load-bearing rather than incidental: a 2-layer randomly
initialised Qwen3-VL with random pixels produces **non-finite logits**, so its cross-entropy is nan
while its per-layer K/V stay finite. That is exactly the shape of the real failure an over-budget
sample causes (FAST answer right-truncated away -> all labels -100 -> nan CE), so the toy doubles as
the regression fixture for the non-finite-CE guard.
"""

import pytest
import torch

from transformers.models.qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from lerobot_policy_griffin_alpha import GriffinAlphaConfig, GriffinAlphaPolicy
from lerobot_policy_griffin_alpha.modeling_griffin_alpha import capture_backbone_kv

CHUNK = 4
ACTION_DIM = 7
IMAGE_TOKEN = 5
SEQ = 8


def _tiny_config(**overrides) -> GriffinAlphaConfig:
    """Split out of tiny_policy so the precision tests can build a policy WITHOUT the trailing
    ``.to(torch.float32)`` — which would erase the very dtypes they are asserting on."""
    text = Qwen3VLTextConfig(
        hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=8, num_key_value_heads=8, head_dim=128, vocab_size=200,
    )
    vision = Qwen3VLVisionConfig(
        hidden_size=64, intermediate_size=128, depth=2, num_heads=2,
        patch_size=16, spatial_merge_size=2, out_hidden_size=256, deepstack_visual_indexes=[0],
    )
    qwen = Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict())
    qwen.image_token_id = IMAGE_TOKEN  # the real id (151655) is outside the tiny vocab

    kwargs = dict(
        qwen3vl_config=qwen, horizon=CHUNK, n_action_steps=CHUNK,
        expert_width=64, expert_mlp_dim=128, expert_depth=2, expert_num_attention_heads=8,
        action_dim_override=ACTION_DIM,
        attn_implementation="sdpa", gradient_checkpointing=False, device="cpu",
    )
    kwargs.update(overrides)
    return GriffinAlphaConfig(**kwargs)


def tiny_policy(**overrides) -> GriffinAlphaPolicy:
    return GriffinAlphaPolicy(_tiny_config(**overrides)).to(torch.float32)


def bf16_backbone_policy(**overrides) -> GriffinAlphaPolicy:
    """A policy whose BACKBONE is bf16, which is the only configuration where the precision branch
    does anything — and deliberately not cast to fp32 afterwards."""
    config = _tiny_config(**overrides)
    # The production route: the parent casts the backbone to whatever qwen3vl_config.dtype records,
    # because the plain constructor (unlike HF from_pretrained) does not build under it.
    config.qwen3vl_config.dtype = "bfloat16"
    return GriffinAlphaPolicy(config)


def tiny_batch(legacy_assistant_turn: bool = False) -> dict:
    """One image (grid 1x2x2 -> 4 patches -> 1 merged token), inference-form prompt.

    No ``labels``: the flow processor emits one prompt for both regimes
    (``emit_assistant_turn=False``), so a training batch looks exactly like an inference one plus the
    action chunk. ``legacy_assistant_turn=True`` reproduces the old FAST-turn batch, which the policy
    now has to refuse.

    ``mm_token_type_ids`` is not optional: Qwen3-VL raises if image_grid_thw is passed without it.
    """
    ids = torch.tensor([[10, IMAGE_TOKEN, 11, 12, 13, 14, 15, 16]])
    batch = {
        "input_ids": ids,
        "attention_mask": torch.ones(1, SEQ, dtype=torch.long),
        "pixel_values": torch.randn(4, 3 * 2 * 16 * 16),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.int32),
        "action": torch.randn(1, CHUNK, ACTION_DIM),
    }
    if legacy_assistant_turn:
        labels = torch.full((1, SEQ), -100)
        labels[0, -3:] = ids[0, -3:]
        batch["labels"] = labels
    return batch


# ------------------------------------------------------------------ the freeze contract
@pytest.mark.parametrize("freeze,expect_backbone_grads", [(True, False), (False, True)])
def test_flow_gradient_reaches_the_backbone_only_when_unfrozen(freeze, expect_backbone_grads):
    """Both directions, because both failures are silent.

    A `requires_grad_(False)` loop that missed a submodule, or a hardcoded `no_grad` on the prefix
    forward that ignores the flag, each produce a run that is quietly not the configuration you
    asked for. The second one actually happened on the cross-attention branch: `no_grad` was
    unconditional, so freeze_backbone=False set requires_grad everywhere and then blocked
    everything, and this test is what caught it.
    """
    torch.manual_seed(0)
    policy = tiny_policy(freeze_backbone=freeze)
    # action_out_proj is zero-initialised, and a zero output projection blocks the flow gradient
    # from reaching ANYTHING upstream of it — including the backbone on the unfrozen arm. Benign in
    # training (it self-resolves after one optimizer step) but it makes a gradient check at step 0
    # measure nothing, so perturb the head into a non-degenerate state first.
    torch.nn.init.normal_(policy.expert.action_out_proj.weight, std=0.02)
    policy.zero_grad()
    loss, _ = policy.forward(tiny_batch())
    loss.backward()

    backbone_hits = [
        name for name, param in policy.model.named_parameters()
        if param.grad is not None and param.grad.abs().sum() > 0
    ]
    expert_hits = [
        name for name, param in policy.expert.named_parameters()
        if param.grad is not None and param.grad.abs().sum() > 0
    ]
    assert bool(backbone_hits) is expect_backbone_grads, backbone_hits[:5]
    assert expert_hits, "the expert must always receive the flow gradient"


def test_freezing_reports_only_the_expert_as_trainable():
    """What lerobot-train prints as num_learnable_params. Verified against a real launch: a tiny
    converted base reported 184K trainable of 21M total, two steps, checkpoint written."""
    policy = tiny_policy(freeze_backbone=True)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in policy.expert.parameters())


def test_only_one_loss_is_reported():
    """The FAST cross-entropy is gone with the backbone frozen — it existed to give a *trainable*
    backbone an on-manifold gradient, and lm_head is weight-tied to embed_tokens, so nothing it
    could train is left unfrozen. A stray second term here means it came back by accident.

    Doubles as the whole-path smoke test: ``_run_prefix`` raises when the capture count does not
    equal the expert depth, so merely REACHING a finite loss here proves the attention-implementation
    swap fired on every layer.
    """
    torch.manual_seed(0)
    policy = tiny_policy()
    loss, loss_dict = policy.forward(tiny_batch())
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.tensor(loss_dict["flow_loss"]))
    assert set(loss_dict) == {"loss", "flow_loss"}


# ------------------------------------------------------------------ capture transparency
@pytest.mark.parametrize("impl", ["sdpa", "eager"])
@pytest.mark.parametrize("padded", [False, True])
def test_the_kv_capture_is_transparent_to_the_backbone(impl, padded):
    """Swapping the attention implementation must not change what the backbone computes.

    It did. ``create_causal_mask`` dispatches on the SAME implementation string as the attention
    interface, and the recorder's key is not in ``ALL_MASK_ATTENTION_FUNCTIONS`` (only sdpa, eager,
    flash_* and flex are), so an unknown key made it return None and the attention mask was silently
    dropped. Causality survived — sdpa and eager infer it when the mask is None — but *padding does
    not*, so a batch with unequal prompt lengths attended to its pad positions. Measured 4.3 max-abs
    drift in the hidden states on a left-padded batch before the mask function was aliased.

    Every real batch has unequal prompt lengths, so this ran on every step of a real run.
    """
    torch.manual_seed(0)
    policy = tiny_policy(attn_implementation=impl)
    batch = tiny_batch()
    inputs = {k: v for k, v in batch.items() if k != "action"}
    if padded:
        inputs["attention_mask"] = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.long)

    with torch.no_grad():
        plain = policy.model(**inputs, output_hidden_states=True).hidden_states
    store: dict = {}
    with torch.no_grad(), capture_backbone_kv(policy.model.config.text_config, store):
        swapped = policy.model(**inputs, output_hidden_states=True).hidden_states

    for layer, (a, b) in enumerate(zip(plain, swapped)):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=f"layer {layer} drifted under capture")


def test_the_captured_kv_is_causal():
    """A later token must not change an earlier position's captured K/V.

    This used to be "the assistant turn cannot reach the prefix" — the property that made masking
    the expert off a FAST answer meaningful. The turn is gone (the processor emits one prompt for
    both regimes), but the property is still the reason the expert can read prefix K/V at all: if
    the capture were non-causal, every prefix key would carry information from positions the expert
    is not supposed to see, and no attention mask could undo it.

    Hold the images fixed — ``tiny_batch()`` regenerates random pixel_values on every call, and
    comparing two batches that differ in their images too reads as a leak of ~4.4 when there is none.
    """
    torch.manual_seed(0)
    policy = tiny_policy()
    batch = tiny_batch()
    inputs = {k: v for k, v in batch.items() if k != "action"}

    def captured(ids):
        store: dict = {}
        with torch.no_grad(), capture_backbone_kv(policy.model.config.text_config, store):
            policy.model(**{**inputs, "input_ids": ids})
        return store

    answer_changed = inputs["input_ids"].clone()
    answer_changed[0, -3:] = torch.tensor([99, 98, 97])
    before, after = captured(inputs["input_ids"]), captured(answer_changed)
    prompt = slice(None, SEQ - 3)
    for layer in before:
        for which, (a, b) in enumerate(zip(before[layer], after[layer])):
            torch.testing.assert_close(
                a[:, :, prompt], b[:, :, prompt], rtol=0, atol=0,
                msg=f"layer {layer} {'K' if which == 0 else 'V'} at prompt positions saw the answer",
            )


def test_the_prefix_forward_skips_the_lm_head_and_its_loss():
    """The FAST head is not trained here, so computing its logits over a 154k vocabulary and then a
    cross-entropy is pure cost: 4.93 GB of bf16 logits at batch 32 / S=500, plus the fp32 upcast
    cross_entropy does internally."""
    torch.manual_seed(0)
    policy = tiny_policy()
    batch = tiny_batch()
    outputs, _, visible, _ = policy._run_prefix(policy._filter_model_inputs(batch), with_grad=False)
    assert outputs.logits.shape[1] == 1, "lm_head was projected over the whole sequence"
    assert getattr(outputs, "loss", None) is None, "cross_entropy ran for a discarded loss"
    # The whole prefix is visible now: there is no assistant turn to exclude.
    assert bool(visible.all()), "something is masking prefix positions off the expert again"


def test_a_legacy_assistant_turn_batch_is_refused():
    """`labels` in the batch means the preprocessor predates emit_assistant_turn, so its prompt has
    a FAST answer that inference will not have. Both fallbacks are unsafe — masking the turn off
    reintroduces the 3-token mRoPE offset (the header is a *loss* mask boundary, not a context one),
    not masking it leaks the answer into the regression target — so the policy refuses."""
    torch.manual_seed(0)
    policy = tiny_policy()
    with pytest.raises(ValueError, match="FAST assistant turn"):
        policy.forward(tiny_batch(legacy_assistant_turn=True))


def test_a_mixed_width_batch_is_refused_rather_than_truncated():
    """`_finalize` returns one tensor, so it cannot carry per-row action widths. Applying row 0's
    width to the batch silently returns wrong dims for the others; the training loss does mask per
    sample (_dim_mask) but a returned chunk cannot."""
    torch.manual_seed(0)
    policy = tiny_policy()
    batch = tiny_batch()
    batch["n_action_dims"] = [ACTION_DIM, ACTION_DIM - 2]
    with pytest.raises(ValueError, match="mixed widths"):
        policy._finalize(torch.zeros(2, CHUNK, ACTION_DIM), batch)


# ------------------------------------------------------- the geometry feeds the forward, once
def test_the_backbone_is_given_the_same_positions_the_expert_uses():
    """`_prefix_geometry` runs before the forward and its position ids go into it.

    Two things this pins. First that it is transparent: Qwen3VLModel hands `get_rope_index`'s own
    (3, B, P) straight down, so passing ours must not change the captured K/V — only a FOUR-row
    tensor takes a different branch there. Second that the expert's mRoPE offset and the backbone's
    positions are now one tensor rather than two independent derivations of it.
    """
    torch.manual_seed(0)
    policy = tiny_policy()
    batch = tiny_batch()
    model_inputs = policy._filter_model_inputs(batch)

    ours, _, _ = policy._prefix_geometry(model_inputs)
    _, captured, _, _ = policy._run_prefix(model_inputs, with_grad=False)

    # what the backbone computes for itself, with no position_ids handed in
    forward_inputs = {k: v for k, v in model_inputs.items() if k != "labels"}
    theirs = policy.model.model.compute_3d_position_ids(
        input_ids=forward_inputs["input_ids"],
        inputs_embeds=None,
        image_grid_thw=forward_inputs["image_grid_thw"],
        attention_mask=forward_inputs["attention_mask"],
        mm_token_type_ids=forward_inputs["mm_token_type_ids"],
    )
    torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
    assert len(captured) == policy.expert.depth


# ------------------------------------------------------------------------ width guards
# ------------------------------------------------------------------------ precision
def test_float32_mode_keeps_master_weights_and_adam_in_fp32():
    """expert_param_dtype="float32" is for plain torch.optim launchers: AdamW allocates its moments with
    zeros_like(p), so bf16 parameters would mean bf16 moments and no fp32 master copy. At this init
    scale a large share of weights would then have a bf16 ULP larger than half an Adam step, i.e. the
    updates round to zero. The parameters stay fp32 and only the matmuls autocast to bf16."""
    policy = bf16_backbone_policy(expert_param_dtype="float32")
    assert next(policy.model.parameters()).dtype == torch.bfloat16, "fixture did not cast the backbone"
    assert {p.dtype for p in policy.expert.parameters()} == {torch.float32}
    assert policy._expert_autocast_dtype == torch.bfloat16, "the matmuls must still run in bf16"

    optimizer = torch.optim.AdamW(policy.get_optim_params())
    for param in policy.expert.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()
    moments = {optimizer.state[p]["exp_avg"].dtype for p in policy.expert.parameters()}
    assert moments == {torch.float32}


def test_the_backbone_mode_casts_and_keeps_the_io_in_fp32():
    """The default "backbone" mode casts the expert to the backbone's dtype (for launchers that keep
    their own fp32 master weights). Only the action I/O and the timestep projection stay fp32, and
    there is nothing to autocast."""
    policy = bf16_backbone_policy()          # no override: "backbone" is the default
    assert policy.config.expert_param_dtype == "backbone"
    assert policy.expert.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert policy.expert.action_in_proj.weight.dtype == torch.float32
    assert policy.expert.action_out_proj.weight.dtype == torch.float32
    assert policy.expert.time_mlp_in.weight.dtype == torch.float32
    assert policy._expert_autocast_dtype is None, "nothing to autocast once the params are bf16"


def test_an_fp32_expert_runs_against_a_bf16_prefix():
    """The seam mixed precision introduces: the captured K/V are the backbone's (bf16) and the
    expert's own are fp32 here, so _layer_forward's `prefix_key.to(key.dtype)` is the only thing
    keeping the concat legal."""
    torch.manual_seed(0)
    policy = bf16_backbone_policy(expert_param_dtype="float32")
    loss, loss_dict = policy.forward(tiny_batch())
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss), loss_dict


def test_a_pre_resample_pad_mask_is_refused_rather_than_sliced():
    """A dataset whose chunk is resampled (e.g. 25 raw steps -> 50) may carry a pad mask from BEFORE the
    resample. Slicing it silently would make `keep.expand_as(velocity)` die with a bare "expanded size
    must match" naming neither the source nor the resample, so `_step_mask` refuses it instead."""
    torch.manual_seed(0)
    policy = tiny_policy()
    with pytest.raises(ValueError, match="covers 2 steps but the chunk is 4"):
        policy._step_mask({"action_is_pad": torch.zeros(1, CHUNK // 2, dtype=torch.bool)},
                          1, CHUNK, torch.device("cpu"))


def test_a_missing_pad_mask_keeps_every_step():
    """A batch without `action_is_pad` (not every dataset supplies one) must keep every step, i.e. the
    mask is all-ones rather than absent."""
    policy = tiny_policy()
    mask = policy._step_mask({}, 2, CHUNK, torch.device("cpu"))
    assert mask.shape == (2, CHUNK, 1)
    assert bool(mask.all())


# ------------------------------------------------------------------------ inverse resample
def test_inference_applies_the_inverse_resample():
    """The input pipeline resamples the target chunk to `horizon`; the parent's FAST
    predict_action_chunk resamples back and the flow override did not, so a base with
    resample_action_chunk_size set returned `horizon` steps to an environment expecting the
    dataset's rate."""
    torch.manual_seed(0)
    policy = tiny_policy(resample_action_chunk_size=CHUNK * 2)
    assert policy._action_resampler is not None
    chunk = policy.predict_action_chunk(tiny_batch())
    assert chunk.shape == (1, CHUNK * 2, ACTION_DIM)


def test_caller_supplied_noise_makes_inference_deterministic():
    """`predict_action_chunk(noise=...)` integrates from the given noise, so the same seeded noise
    yields the same chunk twice while a different draw does not."""
    torch.manual_seed(0)
    policy = tiny_policy()
    gen = torch.Generator().manual_seed(1234)
    noise = torch.randn(1, CHUNK, ACTION_DIM, generator=gen)
    first = policy.predict_action_chunk(tiny_batch(), noise=noise)
    second = policy.predict_action_chunk(tiny_batch(), noise=noise.clone())
    assert first.shape == (1, CHUNK, ACTION_DIM)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    other = policy.predict_action_chunk(tiny_batch(), noise=torch.randn(1, CHUNK, ACTION_DIM, generator=gen))
    assert not torch.equal(first, other)


def test_noise_with_the_wrong_batch_size_is_refused():
    policy = tiny_policy()
    with pytest.raises(ValueError, match="rows"):
        policy.predict_action_chunk(tiny_batch(), noise=torch.zeros(2, CHUNK, ACTION_DIM))
