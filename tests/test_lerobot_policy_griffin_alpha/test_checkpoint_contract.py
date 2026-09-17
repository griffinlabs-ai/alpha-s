"""The saved-checkpoint contract: what a published ``config.json`` / processor JSON must satisfy.

draccus is strict about unknown ``config.json`` keys, and lerobot rebuilds processors from their
saved ``registry_name`` + ``config`` -- so a factory-built pipeline must round-trip through
save/load unchanged, for both types.
"""

import json
from unittest.mock import patch

import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

from lerobot_policy_griffin_alpha import (
    GriffinAlphaConfig,
    GriffinAlphaFASTConfig,
    GriffinAlphaFASTInputProcessorStep,
    GriffinAlphaInputProcessorStep,
    make_griffin_alpha_fast_pre_post_processors,
    make_griffin_alpha_pre_post_processors,
)

FEATURES = dict(
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    },
    arm_control_mode="eef_pose",
    embodiment_prompt="LIBERO simulated Franka Emika Panda, 1 gripper",
    use_relative_actions=False,
    n_action_steps=10,
    device="cpu",
)


def _stats():
    return {
        OBS_STATE: {"q01": -torch.ones(8), "q99": torch.ones(8)},
        ACTION: {"q01": -torch.ones(7), "q99": torch.ones(7)},
    }


def _step_signature(pipeline):
    out = []
    for step in pipeline.steps:
        cfg = step.get_config() if hasattr(step, "get_config") else {}
        cfg = {k: v for k, v in cfg.items() if k not in ("stats", "features")}
        out.append((type(step)._registry_name if hasattr(type(step), "_registry_name") else type(step).__name__, cfg))
    return out


@pytest.mark.parametrize(
    "config_cls, factory, step_cls",
    [
        (GriffinAlphaConfig, make_griffin_alpha_pre_post_processors, GriffinAlphaInputProcessorStep),
        (GriffinAlphaFASTConfig, make_griffin_alpha_fast_pre_post_processors, GriffinAlphaFASTInputProcessorStep),
    ],
    ids=["griffin_alpha", "griffin_alpha_fast"],
)
def test_config_and_processors_round_trip(config_cls, factory, step_cls, fast_action_tokenizer, qwen3_vl_vlm_processor, tmp_path):
    config = config_cls(**FEATURES)

    # config.json: strict-decodable back into the same type with the same fields.
    config._save_pretrained(tmp_path)
    saved = json.load(open(tmp_path / "config.json"))
    assert saved["type"] == config_cls.get_choice_name(config_cls)
    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert type(restored) is config_cls
    assert restored.use_relative_actions is False
    assert restored.image_keys == ()
    assert restored.resolved_image_keys == ("observation.images.image", "observation.images.image2")

    # processors: same registry names and same step configs after save/load.
    with patch(
        "lerobot_policy_griffin_alpha.processor_griffin_alpha_fast.AutoProcessor.from_pretrained",
        return_value=fast_action_tokenizer,
    ):
        with patch.object(step_cls, "_make_vla_processor", return_value=qwen3_vl_vlm_processor):
            pre, post = factory(config, dataset_stats=_stats())
            pre.save_pretrained(tmp_path)
            post.save_pretrained(tmp_path)
            pre2, post2 = make_pre_post_processors(config, pretrained_path=str(tmp_path))

    assert _step_signature(pre2) == _step_signature(pre)
    assert _step_signature(post2) == _step_signature(post)
    names = [n for n, _ in _step_signature(pre)]
    assert names == [
        "rename_observations_processor",
        "griffinlabs/griffin_alpha_add_batch_dimension",
        "relative_actions_processor",
        "normalizer_processor",
        step_cls._registry_name,
        "device_processor",
    ]
    assert [n for n, _ in _step_signature(post)] == [
        "unnormalizer_processor",
        "absolute_actions_processor",
        "device_processor",
    ]
    # The input step records the cameras it was built with, in order.
    input_cfg = json.load(open(tmp_path / "policy_preprocessor.json"))["steps"][4]
    assert input_cfg["registry_name"] == step_cls._registry_name
    assert input_cfg["config"]["image_keys"] == ["observation.images.image", "observation.images.image2"]


def test_unknown_config_key_is_rejected(tmp_path):
    """The contract every published checkpoint depends on: a stray key fails loudly at load time."""
    config = GriffinAlphaConfig(**FEATURES)
    config._save_pretrained(tmp_path)
    data = json.load(open(tmp_path / "config.json"))
    data["some_field_from_a_newer_plugin"] = 1
    json.dump(data, open(tmp_path / "config.json", "w"))
    with pytest.raises(Exception, match="some_field_from_a_newer_plugin"):
        PreTrainedConfig.from_pretrained(tmp_path)
