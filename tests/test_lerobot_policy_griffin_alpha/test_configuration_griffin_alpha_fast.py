"""Config tests for the Qwen3-VL backbone — mirrors test_configuration_griffin_alpha.py.

Also pins the lerobot plugin-resolution contract, which is derived from *names* at runtime and so
cannot be caught by an import-time error.
"""

import dataclasses
import importlib
import json

import draccus
import pytest
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.factory import get_policy_class
from lerobot.utils.constants import ACTION
from transformers.models.qwen3_vl import Qwen3VLConfig

from lerobot_policy_griffin_alpha import (
    GriffinAlphaConfig,
    GriffinAlphaPolicy,
    GriffinAlphaFASTConfig,
    GriffinAlphaFASTPolicy,
    make_griffin_alpha_fast_pre_post_processors,
)
from lerobot_policy_griffin_alpha.backbone import DEFAULT_ACTION_TOKEN_MIN, QWEN_PATCH_MERGE


def test_package_exports():
    assert GriffinAlphaFASTConfig is not None
    assert GriffinAlphaFASTPolicy is not None
    assert make_griffin_alpha_fast_pre_post_processors is not None


class TestRegistrationContract:
    """lerobot derives the policy class, module paths, and factory name from strings at runtime."""

    def test_both_backbones_registered_under_distinct_types(self):
        choices = PreTrainedConfig.get_known_choices()
        assert "griffin_alpha" in choices
        assert "griffin_alpha_fast" in choices
        assert (
            PreTrainedConfig.get_choice_class("griffin_alpha_fast") is GriffinAlphaFASTConfig
        )
        # The two type strings must resolve to their own config classes: a checkpoint's config.json
        # selects the head by this string.
        assert PreTrainedConfig.get_choice_class("griffin_alpha") is GriffinAlphaConfig

    def test_policy_class_resolves_from_type_string(self):
        assert get_policy_class("griffin_alpha_fast") is GriffinAlphaFASTPolicy
        assert get_policy_class("griffin_alpha") is GriffinAlphaPolicy

    def test_config_class_name_matches_policy_class_name(self):
        """lerobot builds "<Name>Policy" from "<Name>Config" — renaming one alone breaks resolution."""
        expected = GriffinAlphaFASTConfig.__name__.removesuffix("Config") + "Policy"
        assert expected == GriffinAlphaFASTPolicy.__name__

    def test_processor_factory_name_and_module_resolve(self):
        """factory._make_processors_from_policy_config derives both from the type string/module."""
        config_module = GriffinAlphaFASTConfig.__module__
        module = importlib.import_module(config_module.replace("configuration_", "processor_"))
        fn_name = "make_griffin_alpha_fast_pre_post_processors"
        assert getattr(module, fn_name) is make_griffin_alpha_fast_pre_post_processors

    def test_modeling_module_resolves_from_config_module(self):
        module = importlib.import_module(
            GriffinAlphaFASTConfig.__module__.replace("configuration_", "modeling_")
        )
        assert getattr(module, "GriffinAlphaFASTPolicy") is GriffinAlphaFASTPolicy

    def test_policy_name_matches_registered_type(self):
        assert GriffinAlphaFASTPolicy.name == "griffin_alpha_fast"


class TestConfigDefaults:
    def test_action_token_min_is_qwen3_vl_4b_base_vocab(self):
        # The Qwen3-VL-4B embedding table has 151936 rows; the action tokens must start there.
        assert DEFAULT_ACTION_TOKEN_MIN == 151936
        config = GriffinAlphaFASTConfig()
        assert config.action_token_min == 151936
        assert config.action_token_max == 151936 + config.action_vocab_size - 1

    def test_max_tokens_per_image_default_is_the_validated_value(self):
        # 72 keeps 4:3 / 16:9 frames at 63/66 tokens AND admits 2:1's 6x12=72 grid, which a 70 cap
        # floored to 5x11=55. 64 was an earlier untested default and must not creep back in.
        assert GriffinAlphaFASTConfig().max_tokens_per_image == 72

    def test_image_pixel_bounds_derivation(self):
        config = GriffinAlphaFASTConfig(max_tokens_per_image=72, min_tokens_per_image=60)
        assert QWEN_PATCH_MERGE == 32
        assert config.image_pixel_budget == 72 * 32**2 == 73728
        assert config.image_pixel_floor == 60 * 32**2 == 61440
        assert config.image_processor_size == {"shortest_edge": 61440, "longest_edge": 73728}

    def test_min_tokens_default_is_below_max(self):
        config = GriffinAlphaFASTConfig()
        assert config.min_tokens_per_image == 60
        assert config.min_tokens_per_image < config.max_tokens_per_image

    def test_min_equal_to_max_is_rejected(self):
        """Collapsing the band sends undersized frames through smart_resize's ceil branch, which
        overshoots max_tokens_per_image (a 64x64 frame became 81 tokens against a nominal 70)."""
        with pytest.raises(ValueError, match="min_tokens_per_image"):
            GriffinAlphaFASTConfig(min_tokens_per_image=72, max_tokens_per_image=72)
        with pytest.raises(ValueError, match="min_tokens_per_image"):
            GriffinAlphaFASTConfig(min_tokens_per_image=80, max_tokens_per_image=72)

    def test_default_image_keys_are_derived_from_visual_input_features(self):
        """Empty means "every VISUAL input feature, in order" -- no camera names baked into the type."""
        assert GriffinAlphaFASTConfig().image_keys == ()
        assert GriffinAlphaFASTConfig().resolved_image_keys == ()
        config = GriffinAlphaFASTConfig(
            input_features={
                "observation.images.cam_a": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
                "observation.images.cam_b": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            }
        )
        assert config.resolved_image_keys == ("observation.images.cam_a", "observation.images.cam_b")
        explicit = GriffinAlphaFASTConfig(image_keys=("observation.images.cam_b",), input_features=config.input_features)
        assert explicit.resolved_image_keys == ("observation.images.cam_b",)

    def test_no_other_backbone_knobs(self):
        """Qwen3-VL has no audio tower and no untied lm_head, so these must not be offered."""
        field_names = {f.name for f in dataclasses.fields(GriffinAlphaFASTConfig)}
        assert "freeze_audio_tower" not in field_names
        assert "train_lm_head_only" not in field_names


def test_n_action_steps_cannot_exceed_horizon():
    with pytest.raises(ValueError, match="n_action_steps"):
        GriffinAlphaFASTConfig(
            horizon=40,
            n_action_steps=60,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        )


def test_validate_features_adds_missing_action(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    griffin_alpha_fast_config.output_features.pop(ACTION, None)
    griffin_alpha_fast_config.validate_features()
    assert ACTION in griffin_alpha_fast_config.output_features


def test_get_optimizer_preset(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    preset = griffin_alpha_fast_config.get_optimizer_preset()
    assert isinstance(preset, AdamWConfig)
    assert preset.lr == griffin_alpha_fast_config.optimizer_lr
    assert preset.betas == griffin_alpha_fast_config.optimizer_betas
    assert preset.eps == griffin_alpha_fast_config.optimizer_eps
    assert preset.weight_decay == griffin_alpha_fast_config.optimizer_weight_decay
    assert preset.grad_clip_norm == griffin_alpha_fast_config.optimizer_grad_clip_norm


def test_get_scheduler_preset(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    preset = griffin_alpha_fast_config.get_scheduler_preset()
    assert isinstance(preset, CosineDecayWithWarmupSchedulerConfig)
    assert preset.peak_lr == griffin_alpha_fast_config.optimizer_lr
    assert preset.num_warmup_steps == griffin_alpha_fast_config.scheduler_warmup_steps
    assert preset.num_decay_steps == griffin_alpha_fast_config.scheduler_decay_steps
    assert preset.decay_lr == griffin_alpha_fast_config.scheduler_decay_lr


def test_action_delta_indices(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    assert griffin_alpha_fast_config.action_delta_indices == list(
        range(griffin_alpha_fast_config.horizon)
    )


def test_reward_delta_indices_is_none(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    assert griffin_alpha_fast_config.reward_delta_indices is None


def test_observation_delta_indices_is_none(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    assert griffin_alpha_fast_config.observation_delta_indices is None


def test_max_action_dim(griffin_alpha_fast_config: GriffinAlphaFASTConfig):
    assert (
        griffin_alpha_fast_config.max_action_dim
        == griffin_alpha_fast_config.output_features[ACTION].shape[-1]
    )


def test_qwen3vl_config_encode_decode_hook():
    cfg = Qwen3VLConfig()
    encoded = draccus.encode(cfg)
    assert isinstance(encoded, dict)
    assert encoded == cfg.to_dict()

    decoded = draccus.decode(Qwen3VLConfig, encoded)
    assert isinstance(decoded, Qwen3VLConfig)
    assert decoded.to_dict() == cfg.to_dict()


def test_qwen3vl_config_draccus_round_trip(
    griffin_alpha_fast_config: GriffinAlphaFASTConfig, tmp_path
):
    config_file = tmp_path / "config.json"
    # Encode against the base class, as lerobot's ``PreTrainedConfig._save_pretrained`` does, so
    # draccus includes the choice ``type`` key that ``from_pretrained`` resolves the subclass from.
    with open(config_file, "w") as f:
        json.dump(draccus.encode(griffin_alpha_fast_config, PreTrainedConfig), f, indent=2)

    with draccus.config_type("json"):
        restored = draccus.parse(PreTrainedConfig, str(config_file), args=[])

    assert isinstance(restored, GriffinAlphaFASTConfig)
    assert isinstance(restored.qwen3vl_config, Qwen3VLConfig)
    assert restored.qwen3vl_config.to_dict() == griffin_alpha_fast_config.qwen3vl_config.to_dict()


def test_config_save_load_round_trip(
    griffin_alpha_fast_config: GriffinAlphaFASTConfig, tmp_path
):
    griffin_alpha_fast_config._save_pretrained(tmp_path)
    restored = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(restored, GriffinAlphaFASTConfig)
    assert isinstance(restored.qwen3vl_config, Qwen3VLConfig)
    # transformers' PretrainedConfig.__eq__ compares __dict__, which
    # carries volatile bookkeeping (transformers_version, model_type) that differs between an
    # in-memory config and a reloaded one. to_dict() is the lossless comparison.
    assert restored.qwen3vl_config.to_dict() == griffin_alpha_fast_config.qwen3vl_config.to_dict()
    for f in dataclasses.fields(GriffinAlphaFASTConfig):
        if f.name == "qwen3vl_config":
            continue
        assert getattr(restored, f.name) == getattr(griffin_alpha_fast_config, f.name), f.name


def test_saved_config_carries_the_qwen_type_string(
    griffin_alpha_fast_config: GriffinAlphaFASTConfig, tmp_path
):
    """A saved Qwen checkpoint must be self-identifying, so serve/eval dispatch picks it up."""
    import json

    griffin_alpha_fast_config._save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["type"] == "griffin_alpha_fast"
