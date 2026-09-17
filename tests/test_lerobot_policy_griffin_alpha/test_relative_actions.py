"""The default (lerobot built-in) relative-action path and its interplay with the SE(3) path."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_griffin_alpha.action_steps import (
    AbsoluteActionWithSE3ProcessorStep,
    RelativeActionWithSE3ProcessorStep,
    reconnect_se3_steps,
)
from lerobot_policy_griffin_alpha.configuration_griffin_alpha_fast import GriffinAlphaFASTConfig
from lerobot_policy_griffin_alpha.processor_griffin_alpha_fast import (
    GriffinAlphaFASTInputProcessorStep,
    make_griffin_alpha_fast_pre_post_processors,
)

HORIZON = 4
ACTION_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"]


def _stats(dim: int = 7) -> dict:
    ones = torch.ones(dim)
    return {
        OBS_STATE: {"q01": -ones, "q99": ones},
        ACTION: {"q01": -ones, "q99": ones},
    }


def _config(**overrides) -> GriffinAlphaFASTConfig:
    kwargs = dict(
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        image_keys=("observation.images.head",),
        action_feature_names=ACTION_NAMES,
        arm_control_mode="joint",
        horizon=HORIZON,
        n_action_steps=HORIZON,
        device="cpu",
        apply_image_augmentation=False,
    )
    kwargs.update(overrides)
    return GriffinAlphaFASTConfig(**kwargs)


def _build(config, fast_action_tokenizer, qwen3_vl_vlm_processor, **kwargs):
    with patch(
        "lerobot_policy_griffin_alpha.processor_griffin_alpha_fast.AutoProcessor.from_pretrained",
        return_value=fast_action_tokenizer,
    ):
        with patch.object(GriffinAlphaFASTInputProcessorStep, "_make_vla_processor", return_value=qwen3_vl_vlm_processor):
            return make_griffin_alpha_fast_pre_post_processors(config, **kwargs)


def _sample(state: torch.Tensor, action: torch.Tensor | None) -> dict:
    batch = {
        "observation.images.head": torch.rand(3, 64, 64),
        OBS_STATE: state,
        "task": "wave",
    }
    if action is not None:
        batch[ACTION] = action
    return batch


class TestBuiltinRelativePath:
    def test_default_is_relative_with_gripper_absolute(self, fast_action_tokenizer, qwen3_vl_vlm_processor):
        config = _config()
        assert config.use_relative_actions is True
        pre, post = _build(config, fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats())
        relative = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
        assert relative._build_mask(7) == [True] * 6 + [False]

    def test_preprocessor_subtracts_state_on_arm_dims_only(self, fast_action_tokenizer, qwen3_vl_vlm_processor):
        pre, _ = _build(_config(), fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats())
        # Stats of +-1 make the quantile normalizer the identity on [-1, 1], so the FAST tokens are
        # computed from the relative chunk; we check the relative step's own output instead.
        relative = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
        state = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]])
        action = torch.full((1, HORIZON, 7), 0.5)
        from lerobot.processor import TransitionKey, create_transition

        out = relative(create_transition(observation={OBS_STATE: state}, action=action))
        got = out[TransitionKey.ACTION]
        torch.testing.assert_close(got[..., :6], action[..., :6] - state[:, None, :6])
        torch.testing.assert_close(got[..., 6], action[..., 6])  # gripper untouched

    def test_postprocessor_restores_absolute_actions(self, fast_action_tokenizer, qwen3_vl_vlm_processor):
        pre, post = _build(_config(), fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats())
        state = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
        pre(_sample(state, None))  # caches the reference state
        relative_chunk = torch.zeros(1, HORIZON, 7)
        absolute = post(relative_chunk)
        torch.testing.assert_close(absolute[0, :, :6], state[None, :6].expand(HORIZON, 6))
        torch.testing.assert_close(absolute[0, :, 6], torch.zeros(HORIZON))

    def test_disabled_path_is_inert(self, fast_action_tokenizer, qwen3_vl_vlm_processor):
        pre, post = _build(
            _config(use_relative_actions=False), fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats()
        )
        state = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
        pre(_sample(state, None))
        chunk = torch.full((1, HORIZON, 7), 0.25)
        torch.testing.assert_close(post(chunk), chunk)

    def test_save_and_reload_repairs_the_pair(self, fast_action_tokenizer, qwen3_vl_vlm_processor, tmp_path):
        """lerobot re-pairs its own relative/absolute steps after deserialization, with no plugin code."""
        config = _config()
        pre, post = _build(config, fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats())
        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha_fast.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(GriffinAlphaFASTInputProcessorStep, "_make_vla_processor", return_value=qwen3_vl_vlm_processor):
                pre2, post2 = make_pre_post_processors(config, pretrained_path=str(tmp_path))
        relative2 = next(s for s in pre2.steps if isinstance(s, RelativeActionsProcessorStep))
        absolute2 = next(s for s in post2.steps if isinstance(s, AbsoluteActionsProcessorStep))
        assert absolute2.relative_step is relative2
        assert relative2.enabled and relative2.action_names == ACTION_NAMES

        state = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9])
        pre2(_sample(state, None))
        torch.testing.assert_close(post2(torch.zeros(1, HORIZON, 7))[0, :, :6], state[None, :6].expand(HORIZON, 6))

    def test_missing_action_names_warns(self, fast_action_tokenizer, qwen3_vl_vlm_processor, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="lerobot_policy_griffin_alpha.pipeline_common"):
            _build(_config(action_feature_names=None), fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats())
        assert "grippers included" in caplog.text


class TestSE3Path:
    def test_se3_pair_is_wired_and_reconnected_after_reload(self, fast_action_tokenizer, qwen3_vl_vlm_processor, tmp_path):
        config = _config(
            use_relative_actions=False,
            se3_segment_start_idxs=[0],
            relative_action_mask=[True] * 16 + [False],
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(17,))},
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(17,)),
                "observation.images.head": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            },
        )
        pre, post = _build(config, fast_action_tokenizer, qwen3_vl_vlm_processor, dataset_stats=_stats(10))
        relative = next(s for s in pre.steps if isinstance(s, RelativeActionWithSE3ProcessorStep))
        absolute = next(s for s in post.steps if isinstance(s, AbsoluteActionWithSE3ProcessorStep))
        assert absolute.relative_step is relative

        pre.save_pretrained(tmp_path)
        post.save_pretrained(tmp_path)
        with patch(
            "lerobot_policy_griffin_alpha.processor_griffin_alpha_fast.AutoProcessor.from_pretrained",
            return_value=fast_action_tokenizer,
        ):
            with patch.object(GriffinAlphaFASTInputProcessorStep, "_make_vla_processor", return_value=qwen3_vl_vlm_processor):
                pre2, post2 = make_pre_post_processors(config, pretrained_path=str(tmp_path))
        absolute2 = next(s for s in post2.steps if isinstance(s, AbsoluteActionWithSE3ProcessorStep))
        assert absolute2.relative_step is None, "lerobot does not know our SE(3) pair"
        reconnect_se3_steps(pre2, post2)
        assert absolute2.relative_step is next(s for s in pre2.steps if isinstance(s, RelativeActionWithSE3ProcessorStep))


class TestConfigValidation:
    def test_mask_without_se3_is_rejected(self):
        with pytest.raises(ValueError, match="SE\\(3\\) path"):
            _config(relative_action_mask=[True] * 7)

    def test_se3_with_builtin_relative_is_rejected(self):
        with pytest.raises(ValueError, match="use_relative_actions=False"):
            _config(se3_segment_start_idxs=[0])

    def test_empty_se3_list_means_none(self):
        assert _config(se3_segment_start_idxs=[]).se3_segment_start_idxs is None
