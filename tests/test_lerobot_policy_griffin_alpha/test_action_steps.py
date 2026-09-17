"""Tests for the backbone-agnostic action-space steps in ``action_steps``."""

import pytest
import torch
from lerobot.processor import ProcessorStepRegistry, TransitionKey
from lerobot.utils.constants import OBS_STATE

from lerobot_policy_griffin_alpha.action_steps import (
    AbsoluteActionWithSE3ProcessorStep,
    GriffinAlphaAddBatchDimensionProcessorStep,
    RelativeActionWithSE3ProcessorStep,
    ResampleActionProcessorStep,
    SE3MatrixToXYZRot6DProcessorStep,
    XYZRot6DToSE3MatrixProcessorStep,
    reconnect_se3_steps,
)
from lerobot_policy_griffin_alpha.configuration_griffin_alpha import GriffinAlphaConfig

from .helpers import make_sample_env_transition


def _random_rotation_matrix() -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _make_se3_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    matrix = torch.eye(4, dtype=rotation.dtype)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix
class TestGriffinAlphaAddBatchDimensionProcessorStep:
    def test_unbatched_adds_batch_dimension(self, griffin_alpha_config: GriffinAlphaConfig):
        step = GriffinAlphaAddBatchDimensionProcessorStep()
        transition = make_sample_env_transition(griffin_alpha_config, batched=False)
        result = step(transition)

        # check state
        assert result[TransitionKey.OBSERVATION][OBS_STATE].ndim == 2
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 7)
        # check images
        for image_key in griffin_alpha_config.image_keys:
            assert isinstance(result[TransitionKey.OBSERVATION][image_key], list)
            assert len(result[TransitionKey.OBSERVATION][image_key]) == 1
            assert result[TransitionKey.OBSERVATION][image_key][0].shape == (3, 64, 64)
        # check info
        assert result[TransitionKey.INFO]["arm_control_mode"] == ["joint"]
        assert result[TransitionKey.INFO]["embodiment_prompt"] == ["test_robot"]
        assert result[TransitionKey.INFO]["n_action_dims"] == [7]
        # check complementary data
        assert result[TransitionKey.COMPLEMENTARY_DATA]["task"] == ["pick up the cup"]
        assert result[TransitionKey.COMPLEMENTARY_DATA]["subtask"] == [""]
        # check action
        assert result[TransitionKey.ACTION] is None


    def test_already_batched_unchanged(self, griffin_alpha_config: GriffinAlphaConfig, sample_env_transition):
        step = GriffinAlphaAddBatchDimensionProcessorStep()
        result = step(sample_env_transition)
        assert result[TransitionKey.OBSERVATION][OBS_STATE].shape == sample_env_transition[TransitionKey.OBSERVATION][OBS_STATE].shape

    def test_serialization_methods(self):
        step = GriffinAlphaAddBatchDimensionProcessorStep()
        assert step.get_config() == {}
        assert step.state_dict() == {}
        step.load_state_dict({})


class TestRelativeActionWithSE3ProcessorStep:
    def test_real_only_subtracts_state(self):
        step = RelativeActionWithSE3ProcessorStep(
            mask=[True, True, True],
            se3_segment_start_idxs=None,
            state_key="observation.state",
        )
        action = torch.tensor([[[1.0, 2.0, 3.0]]])
        state = torch.tensor([[1.0, 3.0, 1.0]])
        transition = {
            "action": action,
            "observation": {"observation.state": state},
        }

        result = step(transition)
        expected = action - state.unsqueeze(-2)
        assert torch.equal(result["action"], expected)

    def test_action_none_returns_identity(self):
        step = RelativeActionWithSE3ProcessorStep(
            mask=[True, True, True],
            state_key="observation.state",
        )
        transition = {
            "action": None,
            "observation": {"observation.state": torch.tensor([[1.0, 2.0, 3.0]])},
        }

        result = step(transition)
        assert result is transition

    def test_get_config_round_trip_via_registry(self):
        step = RelativeActionWithSE3ProcessorStep(
            mask=[True, False, True],
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )

        cfg = step.get_config()
        assert isinstance(cfg["mask"], list)
        assert isinstance(cfg["se3_segment_start_idxs"], list)
        restored = ProcessorStepRegistry.get("griffinlabs/relative_action_with_se3_processor")(**cfg)

        assert restored.mask == step.mask
        assert torch.equal(restored._mask, torch.tensor(step.mask, dtype=torch.bool))
        assert restored.se3_segment_start_idxs == step.se3_segment_start_idxs
        assert restored.state_key == step.state_key


class TestSE3MatrixToXYZRot6DProcessorStep:
    def test_converts_action_and_state_segments(self):
        step = SE3MatrixToXYZRot6DProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        se3_flat = torch.eye(4).reshape(1, 1, 16)
        state = torch.eye(4).reshape(1, 16)
        transition = {
            "action": se3_flat.repeat(1, 2, 1),
            "observation": {"observation.state": state},
        }

        result = step(transition)
        assert result["action"].shape[-1] == 9
        assert result["observation"]["observation.state"].shape[-1] == 9

    def test_action_none_still_converts_state(self):
        step = SE3MatrixToXYZRot6DProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        transition = {
            "action": None,
            "observation": {"observation.state": torch.eye(4).reshape(1, 16)},
        }

        result = step(transition)
        assert result["action"] is None
        assert result["observation"]["observation.state"].shape[-1] == 9

    def test_get_config_round_trip_via_registry(self):
        step = SE3MatrixToXYZRot6DProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        cfg = step.get_config()
        assert isinstance(cfg["se3_segment_start_idxs"], list)

        restored = ProcessorStepRegistry.get("griffinlabs/se3_mat_to_xyz_rot6d_processor")(**cfg)
        assert restored.se3_segment_start_idxs == step.se3_segment_start_idxs
        assert restored.state_key == step.state_key


class TestXYZRot6DToSE3MatrixProcessorStep:
    def test_inverts_se3_to_xyz_rot6d(self):
        forward = SE3MatrixToXYZRot6DProcessorStep(se3_segment_start_idxs=[0], state_key="observation.state")
        inverse = XYZRot6DToSE3MatrixProcessorStep(se3_segment_start_idxs=[0], state_key="observation.state")

        identity = torch.eye(4)
        random_rotation = _random_rotation_matrix()
        random_translation = torch.randn(3)
        random_se3 = _make_se3_matrix(random_rotation, random_translation)

        action = torch.stack([identity.reshape(16), random_se3.reshape(16)], dim=0).unsqueeze(0)
        state = identity.reshape(1, 16)
        transition = {"action": action, "observation": {"observation.state": state}}

        reduced = forward(transition)
        reconstructed = inverse({"action": reduced["action"]})

        assert torch.allclose(reconstructed["action"], action, atol=1e-5)

    def test_get_config_round_trip_via_registry(self):
        step = XYZRot6DToSE3MatrixProcessorStep(
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        cfg = step.get_config()
        restored = ProcessorStepRegistry.get("griffinlabs/xyz_rot6d_to_se3_mat_processor")(**cfg)

        assert restored.se3_segment_start_idxs == step.se3_segment_start_idxs
        assert restored.state_key == step.state_key


class TestAbsoluteActionWithSE3ProcessorStep:
    def test_inverts_relative_action_real_only(self):
        state = torch.randn(1, 5)
        absolute_action = torch.randn(1, 3, 5)
        relative_step = RelativeActionWithSE3ProcessorStep(
            mask=[True] * 5,
            se3_segment_start_idxs=None,
            state_key="observation.state",
        )
        relative_step({"action": None, "observation": {"observation.state": state}})
        relative_action = relative_step(
            {"action": absolute_action, "observation": {"observation.state": state}}
        )["action"]
        absolute_step = AbsoluteActionWithSE3ProcessorStep(
            mask=[True] * 5,
            se3_segment_start_idxs=None,
            state_key="observation.state",
            relative_step=relative_step,
        )

        reconstructed = absolute_step(
            {"action": relative_action, "observation": {"observation.state": state}}
        )["action"]
        assert torch.allclose(reconstructed, absolute_action)

    def test_inverts_relative_action_with_se3_segment(self):
        state_se3 = _make_se3_matrix(_random_rotation_matrix(), torch.randn(3))
        action_se3_a = _make_se3_matrix(_random_rotation_matrix(), torch.randn(3))
        action_se3_b = _make_se3_matrix(_random_rotation_matrix(), torch.randn(3))
        state = state_se3.reshape(1, 16)
        absolute_action = torch.stack([action_se3_a.reshape(16), action_se3_b.reshape(16)], dim=0).unsqueeze(0)
        relative_step = RelativeActionWithSE3ProcessorStep(
            mask=[True] * 16,
            se3_segment_start_idxs=[0],
            state_key="observation.state",
        )
        relative_step({"action": None, "observation": {"observation.state": state}})
        relative_action = relative_step(
            {"action": absolute_action, "observation": {"observation.state": state}}
        )["action"]
        absolute_step = AbsoluteActionWithSE3ProcessorStep(
            mask=[True] * 16,
            se3_segment_start_idxs=[0],
            state_key="observation.state",
            relative_step=relative_step,
        )

        reconstructed = absolute_step(
            {"action": relative_action, "observation": {"observation.state": state}}
        )["action"]
        assert torch.allclose(reconstructed, absolute_action, atol=1e-5)

    def test_get_config_round_trip_via_registry(self):
        step = AbsoluteActionWithSE3ProcessorStep(
            mask=[True, False, True],
            se3_segment_start_idxs=[0],
            state_key="observation.state",
            relative_step=None,
        )
        cfg = step.get_config()
        restored = ProcessorStepRegistry.get("griffinlabs/absolute_action_with_se3_processor")(**cfg)

        assert restored.mask == step.mask
        assert restored.se3_segment_start_idxs == step.se3_segment_start_idxs
        assert restored.state_key == step.state_key


class TestResampleActionProcessorStep:
    def test_divisible_chunk_subsamples(self):
        step = ResampleActionProcessorStep(target_chunk_size=2)
        action = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        transition = {"action": action, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        expected = action[:, 1::2, :]
        assert result["action"].shape == (1, 2, 3)
        assert torch.equal(result["action"], expected)

    def test_non_divisible_chunk_interpolates(self):
        step = ResampleActionProcessorStep(target_chunk_size=3)
        action = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        transition = {"action": action, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        assert result["action"].shape == (1, 3, 3)

    def test_equal_chunk_size_returns_unchanged(self):
        step = ResampleActionProcessorStep(target_chunk_size=4)
        action = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
        transition = {"action": action, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        assert result is transition

    def test_action_none_returns_unchanged(self):
        step = ResampleActionProcessorStep(target_chunk_size=2)
        transition = {"action": None, "observation": {"observation.state": torch.zeros(1, 3)}}

        result = step(transition)
        assert result is transition

    def test_get_config_round_trip_via_registry(self):
        step = ResampleActionProcessorStep(target_chunk_size=2, state_key="observation.state")
        cfg = step.get_config()
        restored = ProcessorStepRegistry.get("griffinlabs/resample_action_processor")(**cfg)

        assert restored.target_chunk_size == step.target_chunk_size
        assert restored.state_key == step.state_key




class TestReconnectSE3Steps:
    def test_reconnects_unpaired_absolute_step(self):
        relative = RelativeActionWithSE3ProcessorStep(mask=[True, False], se3_segment_start_idxs=None)
        absolute = AbsoluteActionWithSE3ProcessorStep(mask=[True, False], se3_segment_start_idxs=None)

        class _Pipe:
            def __init__(self, steps):
                self.steps = steps

        assert absolute.relative_step is None
        reconnect_se3_steps(_Pipe([relative]), _Pipe([absolute]))
        assert absolute.relative_step is relative

    def test_noop_without_se3_relative_step(self):
        absolute = AbsoluteActionWithSE3ProcessorStep(mask=[True], se3_segment_start_idxs=None)

        class _Pipe:
            def __init__(self, steps):
                self.steps = steps

        reconnect_se3_steps(_Pipe([]), _Pipe([absolute]))
        assert absolute.relative_step is None
