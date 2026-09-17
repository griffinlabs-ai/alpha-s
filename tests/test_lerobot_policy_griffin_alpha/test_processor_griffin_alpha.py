"""The processor path on the real Qwen3-VL tokenizer/image processor — CPU only, no model.

Session-scoped so the processor is built once (mirroring
``tests/test_lerobot_policy_griffin_alpha/conftest.py``). Everything here is free to run: the only
external asset is the Qwen3-VL *processor* files, not weights.

What this covers that the expert-level tests cannot: that the flow head's one processor change (the
normalized action chunk re-attached to the batch) actually survives the real pipeline, and the three
upstream traps that produce wrong numbers rather than errors — right-truncation dropping the FAST
answer, double image rescaling, and action-token ids drifting off the model's base vocab.
"""

import numpy as np
import pytest
import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_griffin_alpha import (
    GriffinAlphaConfig,
    GriffinAlphaInputProcessorStep,
    make_griffin_alpha_pre_post_processors,
)

HORIZON = 8
RAW_ACTION_DIM = 7   # LIBERO: 7-D OSC_POSE deltas
STATE_DIM = 8        # eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2)
IMAGE_KEY = "observation.images.image"


def flow_config(**overrides) -> GriffinAlphaConfig:
    kwargs = dict(
        horizon=HORIZON, n_action_steps=HORIZON, action_dim_override=RAW_ACTION_DIM, device="cpu",
        image_keys=(IMAGE_KEY,),
        # LIBERO's actions are already per-step OSC deltas, so no chunk-relative transform.
        relative_action_mask=None, se3_segment_start_idxs=[],
        # required: the plugin refuses to guess it (no silent default)
        arm_control_mode="eef_pose",
        apply_image_augmentation=False,
    )
    kwargs.update(overrides)
    return GriffinAlphaConfig(**kwargs)


def stats(q01_action: float = -1.0, q99_action: float = 1.0) -> dict:
    return {
        OBS_STATE: {"q01": np.full(STATE_DIM, -1.0), "q99": np.full(STATE_DIM, 1.0)},
        ACTION: {"q01": np.full(RAW_ACTION_DIM, q01_action),
                 "q99": np.full(RAW_ACTION_DIM, q99_action)},
    }


def sample(action_value: float = 0.0, height: int = 128, width: int = 128) -> dict:
    """A dataset-style batch dict — the pipeline's default converter is ``batch_to_transition``."""
    return {
        IMAGE_KEY: torch.rand(3, height, width),   # float32 in [0, 1], as lerobot decodes
        OBS_STATE: torch.zeros(STATE_DIM),
        ACTION: torch.full((HORIZON, RAW_ACTION_DIM), action_value),
        "task": "pick up the black bowl",
        f"{ACTION}_is_pad": torch.zeros(HORIZON, dtype=torch.bool),
    }


@pytest.fixture(scope="session")
def pipeline():
    return make_griffin_alpha_pre_post_processors(flow_config(), dataset_stats=stats())


@pytest.fixture(scope="session")
def flow_step(pipeline):
    """isinstance, not a substring of the class name.

    This used to be `next(step for step in pre.steps if "Flow" in type(step).__name__)`, which broke
    silently-but-loudly when the class was renamed Flow -> FM: the generator went empty and every
    test using this fixture errored with a bare StopIteration at setup, naming nothing. A substring
    match on a class name is a rename waiting to happen, so the class is imported and matched by
    identity instead.
    """
    pre, _ = pipeline
    return next(
        step for step in pre.steps
        if isinstance(step, GriffinAlphaInputProcessorStep)
    )


# ---------------------------------------------------------------- pipeline structure
def test_pipeline_structure(pipeline):
    """lerobot's relative-action pair is always present (enabled follows the config); the SE(3)
    steps are absent unless configured; the input step sits right before the device step."""
    pre, post = pipeline
    names = [type(step).__name__ for step in pre.steps]
    assert names == [
        "RenameObservationsProcessorStep",
        "GriffinAlphaAddBatchDimensionProcessorStep",
        "RelativeActionsProcessorStep",
        "NormalizerProcessorStep",
        "GriffinAlphaInputProcessorStep",
        "DeviceProcessorStep",
    ], names
    assert [type(step).__name__ for step in post.steps] == [
        "UnnormalizerProcessorStep",
        "AbsoluteActionsProcessorStep",
        "DeviceProcessorStep",
    ]
    assert not any("SE3" in name for name in names)


def test_delta_controlled_target_disables_relative_actions():
    """LIBERO's actions are already per-step OSC deltas: a checkpoint for it carries
    use_relative_actions=False, and the built-in pair is then present but inert."""
    pre, post = make_griffin_alpha_pre_post_processors(
        flow_config(use_relative_actions=False), dataset_stats=stats()
    )
    relative = next(s for s in pre.steps if type(s).__name__ == "RelativeActionsProcessorStep")
    absolute = next(s for s in post.steps if type(s).__name__ == "AbsoluteActionsProcessorStep")
    assert relative.enabled is False and absolute.enabled is False
    batch = pre(sample(action_value=0.5))
    torch.testing.assert_close(batch[ACTION], torch.full((1, HORIZON, RAW_ACTION_DIM), 0.5))


# ---------------------------------------------------------------- the flow head's one addition
def test_action_chunk_is_normalized_not_raw():
    """The emitted tensor must be post-normalizer — the flow target lives in the transformed space.
    With q01=0/q99=2 the map is x -> x - 1, so a raw 1.0 must arrive as 0.0."""
    pre, _ = make_griffin_alpha_pre_post_processors(
        flow_config(), dataset_stats=stats(q01_action=0.0, q99_action=2.0)
    )
    batch = pre(sample(action_value=1.0))
    torch.testing.assert_close(batch[ACTION], torch.zeros(1, HORIZON, RAW_ACTION_DIM))


def test_action_chunk_is_padded_to_the_head_width():
    """The head is pinned at action_dim_override; the loss masks back down per sample."""
    pre, _ = make_griffin_alpha_pre_post_processors(
        flow_config(action_dim_override=32), dataset_stats=stats()
    )
    batch = pre(sample(action_value=0.5))
    assert batch[ACTION].shape == (1, HORIZON, 32)
    assert torch.count_nonzero(batch[ACTION][..., RAW_ACTION_DIM:]) == 0


def test_inference_sample_carries_no_action(pipeline):
    """No action -> no assistant turn, so the prefix is clean and the K/V are cacheable."""
    pre, _ = pipeline
    observation = sample()
    observation.pop(ACTION)
    observation.pop(f"{ACTION}_is_pad")
    batch = pre(observation)
    assert ACTION not in batch
    assert "labels" not in batch


# ---------------------------------------------------------------- the three silent traps
def test_images_are_not_rescaled_twice(flow_step):
    """lerobot decodes to float32 [0, 1]; an image processor that also divides by 255 collapses
    pixel_values. The Qwen paths hardcode do_rescale=False."""
    assert flow_step._vla_processor.image_processor.do_rescale is False


def test_action_tokens_start_at_the_models_base_vocab(flow_step):
    """If these drift, the FAST branch trains on untrained embedding rows — no crash, no warning."""
    tokenizer = flow_step._vla_processor.tokenizer
    base = flow_step.tokenizer_vocab_pad_to
    assert tokenizer.convert_tokens_to_ids("<robot_action_0>") == base
    assert tokenizer.convert_tokens_to_ids("<robot_action_2047>") == base + 2047
    assert len(tokenizer) == base + 2048 + 256   # + action vocab + proprio vocab


# ---------------------------------------------------------------- the visual-token band
@pytest.mark.parametrize("height,width,expected", [(128, 128, 64), (120, 160, 63), (144, 256, 66)])
def test_image_token_count_follows_aspect_ratio_not_resolution(pipeline, height, width, expected):
    """The band is steered by two pixel bounds, so the count depends on aspect ratio alone:
    1:1 -> 64, 4:3 -> 63, 16:9 -> 66. Collapsing the bounds to one value breaks this."""
    pre, _ = pipeline
    batch = pre(sample(height=height, width=width))
    image_token_id = 151655   # Qwen3-VL's image placeholder
    assert int((batch["input_ids"][0] == image_token_id).sum()) == expected


def test_the_pad_width_is_reported_so_the_loss_can_mask_it():
    """Padding to the head width is only safe if the loss is told the real width.

    ``_dim_mask`` masks on ``n_action_dims``, and the base step forwards it from ``info`` — which a
    dataset pipeline may or may not populate. lerobot-train hands every step an empty info, so
    without this the key was ``None`` and the loss treated the
    zero pad as signal: 7 of 32 dims real means the real ones carry 7/32 of the gradient, and each
    padded dim (a=0, so x_t = t*eps and u_t = eps) asks the expert for a 1/t gain with t down to
    0.001. Both silent; the existing tiny-backbone tests set the key by hand and so could not see it.
    """
    pre, _ = make_griffin_alpha_pre_post_processors(
        flow_config(action_dim_override=32), dataset_stats=stats()
    )
    batch = pre(sample())
    assert batch["n_action_dims"] == [RAW_ACTION_DIM]


def test_a_dataset_supplied_width_is_not_overwritten(flow_step):
    """A dataset may supply per-row widths (one batch can mix embodiments). The processor's own value
    is per-batch and cannot express that, so it must never clobber info's."""
    from lerobot.processor.pipeline import TransitionKey

    action = torch.zeros(2, HORIZON, RAW_ACTION_DIM)
    transition = {
        TransitionKey.ACTION: action,
        TransitionKey.OBSERVATION: {
            IMAGE_KEY: torch.rand(2, 3, 64, 64),
            OBS_STATE: torch.zeros(2, STATE_DIM),
        },
        TransitionKey.INFO: {"n_action_dims": [RAW_ACTION_DIM, RAW_ACTION_DIM - 3]},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["a", "b"]},
    }
    result = flow_step(transition)
    assert result[TransitionKey.COMPLEMENTARY_DATA]["n_action_dims"] == [
        RAW_ACTION_DIM,
        RAW_ACTION_DIM - 3,
    ]
