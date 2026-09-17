"""Backbone-agnostic action-space processor steps shared by both Griffin Alpha-S policy types.

Registry names (``griffinlabs/...``) are serialized into every saved processor JSON, so they are part
of the checkpoint format: never rename one in place.

Two families live here:

- ``GriffinAlphaAddBatchDimensionProcessorStep`` -- the only step every checkpoint uses.
- The SE(3) recipe (``RelativeActionWithSE3ProcessorStep`` / ``AbsoluteActionWithSE3ProcessorStep``,
  ``SE3MatrixToXYZRot6DProcessorStep`` / ``XYZRot6DToSE3MatrixProcessorStep``) plus
  ``ResampleActionProcessorStep``. These are opt-in through ``se3_segment_start_idxs`` /
  ``resample_action_chunk_size`` on the policy config. The default relative-action path uses
  lerobot's own ``RelativeActionsProcessorStep`` / ``AbsoluteActionsProcessorStep`` instead (see
  ``pipeline_common``), because lerobot re-pairs those two after deserialization and not ours --
  a harness that loads an SE(3) checkpoint must call :func:`reconnect_se3_steps` itself.
"""

from collections.abc import Set
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.interpolate import CubicSpline

import lerobot.processor
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


@dataclass
@ProcessorStepRegistry.register("griffinlabs/griffin_alpha_add_batch_dimension")
class GriffinAlphaAddBatchDimensionProcessorStep(ProcessorStep):
    """Add a batch dimension to unbatched transitions, keyed on observation.state."""

    @staticmethod
    def _add_batch_dim_to_dict(d: dict) -> dict:
        batched = d.copy()
        for key, value in d.items():
            if isinstance(value, torch.Tensor) and not key.startswith(f"{OBS_IMAGES}."):
                batched[key] = value.unsqueeze(0)
            else:
                batched[key] = [value]
        return batched

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition[TransitionKey.OBSERVATION]

        if observation[OBS_STATE].ndim == 2:
            return transition.copy()

        new_transition = transition.copy()

        new_transition[TransitionKey.OBSERVATION] = self._add_batch_dim_to_dict(observation)

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            new_transition[TransitionKey.ACTION] = action.unsqueeze(0)

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is not None:
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = \
                self._add_batch_dim_to_dict(complementary_data)

        new_transition[TransitionKey.INFO] = \
            self._add_batch_dim_to_dict(transition[TransitionKey.INFO])

        return new_transition

    def transform_features(self, features):
        return features


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/resample_action_processor")
class ResampleActionProcessorStep(lerobot.processor.ProcessorStep):
    """
    Resample the action tensor from any chunk size to a target chunk size,
    by cubic spline interpolation.

    Args:
     - target_chunk_size: The target chunk size to resample the action tensor to.
     - state_key: Key for the state tensor that corresponds to the action tensor.
       If None, the initial state is assumed to be zero.
       Leave as None if the action tensor is in delta space.
    """

    target_chunk_size: int
    state_key: str | None = None

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            return transition

        if self.state_key is not None:
            initial_state = transition["observation"][self.state_key].unsqueeze(-2)
        else:
            initial_state = torch.zeros_like(action[..., :1, :])
        orig_chunk_size = action.shape[-2]

        if orig_chunk_size == self.target_chunk_size:
            return transition

        new_transition = transition.copy()

        if orig_chunk_size % self.target_chunk_size == 0:
            # If the original chunk size is a multiple of the target chunk size,
            # we can simply take every n-th action.
            step_size = orig_chunk_size // self.target_chunk_size
            new_transition["action"] = action[..., step_size - 1 :: step_size, :]
            return new_transition

        new_transition["action"] = self._interpolate(action, initial_state, orig_chunk_size)
        return new_transition

    def _interpolate(self, action, initial_state, orig_chunk_size):
        trajectory = torch.cat([initial_state, action], dim=-2)
        old_times = np.linspace(0, 1, orig_chunk_size + 1)
        new_times = np.linspace(1 / self.target_chunk_size, 1, self.target_chunk_size)

        traj_np = trajectory.cpu().numpy()
        cs = CubicSpline(old_times, traj_np, axis=-2)
        resampled = cs(new_times)

        return torch.from_numpy(resampled).to(dtype=action.dtype, device=action.device)

    def transform_features(self, features):
        original_shape = features["action"]["action"].shape
        features["action"]["action"] = PolicyFeature(
            FeatureType.ACTION,
            (self.target_chunk_size, original_shape[-1]),
        )
        return features

    def get_config(self):
        return {
            "target_chunk_size": self.target_chunk_size,
            "state_key": self.state_key,
        }


@dataclass
class EmbeddedSE3Segmenter:
    se3_segment_start_idxs: Set[int] | list[int] | None = None

    def __post_init__(self):
        split_points = []
        for start_idx in sorted(self.se3_segment_start_idxs or set()):
            split_points.append((start_idx, "se3_matrix"))
            split_points.append((start_idx + 16, "real"))
        if len(split_points) == 0 or split_points[0][0] != 0:
            split_points.insert(0, (None, "real"))
        self.slices = [
            (
                slice(
                    split_points[i][0],
                    split_points[i + 1][0] if i + 1 < len(split_points) else None,
                ),
                split_points[i][1],
            )
            for i in range(len(split_points))
        ]


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/relative_action_with_se3_processor")
class RelativeActionWithSE3ProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert action tensor from absolute space to relative space.
    Expects an action shape of [..., chunk_size, degrees_of_freedom].
    Expects transition to have a state tensor in the same vector space as the action tensor.

    Args:
     - mask: Mask of which action tensor dimensions to convert to relative space.
       `True` dimensions output relative space, `False` dimensions keep absolute space.
     - se3_segment_start_idxs: Indices of the action tensor dimensions that correspond to SE(3) matrices.
       If None, all action tensor dimensions are assumed to be real.
     - state_key: Key for the state tensor that corresponds to the action tensor.
    """

    mask: list[bool]
    se3_segment_start_idxs: list[int] | None = None
    state_key: str = OBS_STATE
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._mask = torch.tensor(self.mask, dtype=torch.bool)
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            self._last_state = transition["observation"].get(self.state_key)
            return transition

        new_transition = transition.copy()

        state = transition["observation"][self.state_key].unsqueeze(-2)
        leading_dims = action.shape[:-1]

        assert self._mask.shape[-1] == action.shape[-1]

        segments = []

        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "real":
                segment = action[..., segment_slice] - state[..., segment_slice]
            elif operand_type == "se3_matrix":
                action_se3 = action[..., segment_slice].view(*leading_dims, 4, 4)
                state_se3 = state[..., segment_slice].view(*leading_dims[:-1], 1, 4, 4)
                segment = state_se3.inverse().matmul(action_se3)
                segment = segment.view(*leading_dims, 16)
            segments.append(segment)

        deltas = torch.cat(segments, dim=-1)
        mask = self._mask.to(action.device)
        new_transition["action"] = torch.where(mask.expand(action.shape), deltas, action)
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {
            "mask": self.mask,
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/se3_mat_to_xyz_rot6d_processor")
class SE3MatrixToXYZRot6DProcessorStep(lerobot.processor.ProcessorStep):
    """
    Convert SE(3) matrices in the action tensor to XYZ and rot6d.
    """

    se3_segment_start_idxs: list[int]
    state_key: str = OBS_STATE

    PER_SE3_MATRIX_DIM_REDUCTION = 7    # 16 dimensions -> 9 dimensions

    def __post_init__(self):
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    @staticmethod
    def convert(action: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [action[..., :3, 3], action[..., :3, :2].flatten(start_dim=-2)],
            dim=-1,
        )

    def __call__(self, transition):
        new_transition = transition.copy()
        action = transition.get("action")
        state = transition["observation"][self.state_key]

        action_segments = []
        state_segments = []

        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "real":
                if action is not None:
                    action_segments.append(action[..., segment_slice])
                state_segments.append(state[..., segment_slice])
            elif operand_type == "se3_matrix":
                if action is not None:
                    action_se3 = action[..., segment_slice].view(*action.shape[:-1], 4, 4)
                    action_segments.append(self.convert(action_se3))
                state_se3 = state[..., segment_slice].view(*state.shape[:-1], 4, 4)
                state_segments.append(self.convert(state_se3))

        if action is not None:
            new_transition["action"] = torch.cat(action_segments, dim=-1)
        new_transition["observation"] = transition["observation"].copy()
        new_transition["observation"][self.state_key] = torch.cat(state_segments, dim=-1)
        return new_transition

    def transform_features(self, features):
        old_shape = features[PipelineFeatureType.ACTION]["action"].shape
        num_se3_segments = sum(1 for _, t in self._segmenter.slices if t == "se3_matrix")
        features[PipelineFeatureType.ACTION]["action"] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        old_shape = features[PipelineFeatureType.OBSERVATION][self.state_key].shape
        features[PipelineFeatureType.OBSERVATION][self.state_key] = PolicyFeature(
            FeatureType.STATE,
            old_shape[:-1] + (old_shape[-1] - num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        return features

    def get_config(self):
        return {
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/xyz_rot6d_to_se3_mat_processor")
class XYZRot6DToSE3MatrixProcessorStep(lerobot.processor.ProcessorStep):
    se3_segment_start_idxs: list[int]
    state_key: str = OBS_STATE

    PER_SE3_MATRIX_DIM_REDUCTION = 7    # 16 dimensions -> 9 dimensions

    def __post_init__(self):
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)
        self._num_se3_segments = 0
        self._reduced_slices: list[tuple[slice, str]] = []
        reduced_start = 0
        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "se3_matrix":
                self._num_se3_segments += 1
                reduced_stop = reduced_start + 9
                self._reduced_slices.append((slice(reduced_start, reduced_stop), operand_type))
                reduced_start = reduced_stop
                continue

            if segment_slice.stop is None:
                self._reduced_slices.append((slice(reduced_start, None), operand_type))
                continue

            segment_start = segment_slice.start or 0
            width = segment_slice.stop - segment_start
            reduced_stop = reduced_start + width
            self._reduced_slices.append((slice(reduced_start, reduced_stop), operand_type))
            reduced_start = reduced_stop

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            return transition

        new_transition = transition.copy()
        segments = []
        for segment_slice, operand_type in self._reduced_slices:
            segment = action[..., segment_slice]
            if operand_type == "real":
                segments.append(segment)
                continue

            xyz = segment[..., :3]
            rot6d = segment[..., 3:9]
            rot_cols = rot6d.view(*segment.shape[:-1], 3, 2)
            a1 = rot_cols[..., :, 0]
            a2 = rot_cols[..., :, 1]
            b1 = torch.nn.functional.normalize(a1, dim=-1)
            b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
            b3 = torch.cross(b1, b2, dim=-1)
            rotation = torch.stack((b1, b2, b3), dim=-1)

            se3 = torch.zeros(*segment.shape[:-1], 4, 4, dtype=segment.dtype, device=segment.device)
            se3[..., :3, :3] = rotation
            se3[..., :3, 3] = xyz
            se3[..., 3, 3] = 1
            segments.append(se3.flatten(start_dim=-2))

        new_transition["action"] = torch.cat(segments, dim=-1)
        return new_transition

    def transform_features(self, features):
        old_shape = features[PipelineFeatureType.ACTION]["action"].shape
        features[PipelineFeatureType.ACTION]["action"] = PolicyFeature(
            FeatureType.ACTION,
            old_shape[:-1] + (old_shape[-1] + self._num_se3_segments * self.PER_SE3_MATRIX_DIM_REDUCTION,),
        )
        return features

    def get_config(self):
        return {
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


@dataclass
@lerobot.processor.ProcessorStepRegistry.register("griffinlabs/absolute_action_with_se3_processor")
class AbsoluteActionWithSE3ProcessorStep(lerobot.processor.ProcessorStep):
    mask: list[bool]
    se3_segment_start_idxs: list[int] | None = None
    state_key: str = OBS_STATE
    relative_step: "RelativeActionWithSE3ProcessorStep | None" = field(default=None, repr=False)

    def __post_init__(self):
        self._mask = torch.tensor(self.mask, dtype=torch.bool)
        self._segmenter = EmbeddedSE3Segmenter(self.se3_segment_start_idxs)

    def __call__(self, transition):
        action = transition.get("action")
        if action is None:
            return transition
        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionWithSE3ProcessorStep requires a paired RelativeActionWithSE3ProcessorStep."
            )

        state = self.relative_step._last_state
        if state is None:
            raise RuntimeError(
                "No cached state found in paired RelativeActionWithSE3ProcessorStep. "
                "Run the preprocessor before this postprocessor."
            )

        state = state.to(device=action.device, dtype=action.dtype)
        state = state.unsqueeze(-2)
        leading_dims = action.shape[:-1]

        assert self._mask.shape[-1] == action.shape[-1]

        segments = []
        for segment_slice, operand_type in self._segmenter.slices:
            if operand_type == "real":
                segment = action[..., segment_slice] + state[..., segment_slice]
            elif operand_type == "se3_matrix":
                action_se3 = action[..., segment_slice].view(*leading_dims, 4, 4)
                state_se3 = state[..., segment_slice].view(*leading_dims[:-1], 1, 4, 4)
                segment = state_se3.matmul(action_se3).view(*leading_dims, 16)
            segments.append(segment)

        new_transition = transition.copy()
        absolutes = torch.cat(segments, dim=-1)
        mask = self._mask.to(action.device)
        new_transition["action"] = torch.where(mask.expand(action.shape), absolutes, action)
        return new_transition

    def transform_features(self, features):
        return features

    def get_config(self):
        return {
            "mask": self.mask,
            "se3_segment_start_idxs": self.se3_segment_start_idxs,
            "state_key": self.state_key,
        }


def reconnect_se3_steps(preprocessor, postprocessor) -> None:
    """Re-pair ``AbsoluteActionWithSE3ProcessorStep.relative_step`` after deserialization.

    The pairing is a live object reference and is not serialized. lerobot performs this wiring
    for its own ``RelativeActionsProcessorStep`` / ``AbsoluteActionsProcessorStep`` inside
    ``make_pre_post_processors`` but knows nothing about these classes, so a harness that loads an
    SE(3) checkpoint through ``make_pre_post_processors(pretrained_path=...)`` must call this once
    on the two pipelines it gets back. A no-op when the preprocessor has no SE(3) relative step.
    """
    relative_step = next(
        (s for s in getattr(preprocessor, "steps", []) if isinstance(s, RelativeActionWithSE3ProcessorStep)),
        None,
    )
    if relative_step is None:
        return
    for step in getattr(postprocessor, "steps", []):
        if isinstance(step, AbsoluteActionWithSE3ProcessorStep) and step.relative_step is None:
            step.relative_step = relative_step
