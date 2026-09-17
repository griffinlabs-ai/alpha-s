"""The pre/post-processor pipeline shared by both heads.

Step order (``[x]`` = present only when the config asks for it)::

    pre : Rename({}) -> AddBatchDim
          -> RelativeActionsProcessorStep(enabled=use_relative_actions)      # lerobot's, ALWAYS present
             [se3] RelativeActionWithSE3ProcessorStep(relative_action_mask)
          -> [se3] SE3MatrixToXYZRot6D -> [resample] ResampleActionProcessorStep(horizon)
          -> Normalizer(observation.state; actions too when stats carry them)  # ALWAYS present
          -> <input step: flow | FAST> -> Device(config.device)
    post: Unnormalizer(output_features)                                         # ALWAYS present
          -> [se3] XYZRot6DToSE3Matrix
          -> AbsoluteActionsProcessorStep(enabled=use_relative_actions)         # lerobot's, ALWAYS present
             [se3] AbsoluteActionWithSE3ProcessorStep
          -> Device("cpu")

lerobot's relative/absolute pair and the (un)normalizers are always present, even when disabled or
built without stats, for three reasons: ``lerobot-train --policy.path=...`` overrides
``normalizer_processor`` / ``unnormalizer_processor`` / ``device_processor`` /
``rename_observations_processor`` (and, with ``use_relative_actions``, the relative pair) BY NAME
and errors if a named step is missing; a present-but-disabled pair lets a user flip
``--policy.use_relative_actions`` on a published checkpoint; and lerobot re-pairs its own
``AbsoluteActionsProcessorStep.relative_step`` after deserialization with no plugin code.

The SE(3) pair is ours and lerobot does not re-pair it: call ``action_steps.reconnect_se3_steps``
after loading an SE(3) checkpoint through ``make_pre_post_processors(pretrained_path=...)``.
"""

from __future__ import annotations

import logging

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .action_steps import (
    AbsoluteActionWithSE3ProcessorStep,
    GriffinAlphaAddBatchDimensionProcessorStep,
    RelativeActionWithSE3ProcessorStep,
    ResampleActionProcessorStep,
    SE3MatrixToXYZRot6DProcessorStep,
    XYZRot6DToSE3MatrixProcessorStep,
)
from .backbone import GriffinAlphaBackboneConfig

logger = logging.getLogger(__name__)


def build_pre_post_processors(
    config: GriffinAlphaBackboneConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    input_step: ProcessorStep,
) -> tuple[
    PolicyProcessorPipeline[EnvTransition, EnvTransition],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Assemble the two pipelines around a head-specific ``input_step``. See the module docstring."""
    if config.use_relative_actions and config.action_feature_names is None and config.input_features:
        logger.warning(
            "use_relative_actions=True but action_feature_names is None: lerobot's relative-action "
            "step will make EVERY action dimension relative, grippers included. Give the dataset's "
            "`action` feature `names` (lerobot fills action_feature_names from them), or set "
            "use_relative_actions=False."
        )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        GriffinAlphaAddBatchDimensionProcessorStep(),
    ]

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=list(config.relative_exclude_joints),
        action_names=config.action_feature_names,
    )
    input_steps.append(relative_step)

    se3_relative_step = None
    if config.se3_segment_start_idxs and config.relative_action_mask is not None:
        se3_relative_step = RelativeActionWithSE3ProcessorStep(
            mask=config.relative_action_mask,
            se3_segment_start_idxs=config.se3_segment_start_idxs,
            state_key=config.state_key,
        )
        input_steps.append(se3_relative_step)
    if config.se3_segment_start_idxs:
        input_steps.append(
            SE3MatrixToXYZRot6DProcessorStep(
                se3_segment_start_idxs=config.se3_segment_start_idxs, state_key=config.state_key
            )
        )
    if config.resample_action_chunk_size is not None:
        input_steps.append(ResampleActionProcessorStep(target_chunk_size=config.horizon))

    # ``features`` lists only observation.state, but NormalizerProcessorStep normalizes every key its
    # stats carry -- so the action chunk is normalized too whenever ``dataset_stats`` has "action".
    # With stats=None (a base checkpoint before any dataset is attached) the step is inert until
    # lerobot-train overrides its stats from the dataset.
    norm_map: dict[str, NormalizationMode] = config.normalization_mapping
    state_feature = (config.input_features or {}).get(OBS_STATE)
    if state_feature is not None:
        state_shape = tuple(state_feature.shape)
    elif dataset_stats is not None and OBS_STATE in dataset_stats:
        # lerobot stats dicts also carry scalars (e.g. "count"); take the first array-valued entry.
        arrays = [v for v in dataset_stats[OBS_STATE].values() if hasattr(v, "shape") and len(v.shape) > 0]
        state_shape = tuple(arrays[0].shape) if arrays else (0,)
    else:
        state_shape = (0,)
    normalizer_features = {OBS_STATE: PolicyFeature(FeatureType.STATE, state_shape)}
    input_steps.append(
        NormalizerProcessorStep(
            features=normalizer_features,
            norm_map=norm_map,
            stats=dataset_stats,
            normalize_observation_keys={OBS_STATE},
        )
    )

    input_steps.extend([input_step, DeviceProcessorStep(device=config.device)])

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(features=config.output_features, norm_map=norm_map, stats=dataset_stats),
    ]
    if config.se3_segment_start_idxs:
        output_steps.append(
            XYZRot6DToSE3MatrixProcessorStep(
                se3_segment_start_idxs=config.se3_segment_start_idxs, state_key=config.state_key
            )
        )
    output_steps.append(AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step))
    if se3_relative_step is not None:
        output_steps.append(
            AbsoluteActionWithSE3ProcessorStep(
                mask=config.relative_action_mask,
                se3_segment_start_idxs=config.se3_segment_start_idxs,
                state_key=config.state_key,
                relative_step=se3_relative_step,
            )
        )
    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[EnvTransition, EnvTransition](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_output=lambda tr: tr[TransitionKey.COMPLEMENTARY_DATA],
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
