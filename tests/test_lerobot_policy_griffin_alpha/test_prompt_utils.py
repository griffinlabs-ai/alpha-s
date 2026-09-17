"""Tests for the prompt helpers in ``prompt_utils``."""

import pytest
import torch

from lerobot_policy_griffin_alpha.prompt_utils import (
    GriffinAlphaImageTransform,
    index_optional_list,
    make_proprio_state_tokens,
    map_fast_token_to_vlm_action,
    map_normalized_state_to_vlm_proprio,
)


class TestHelperFunctions:
    def testindex_optional_list_none(self):
        assert index_optional_list(None, 0) is None

    def testindex_optional_list_indexing(self):
        assert index_optional_list(["a", "b"], 1) == "b"

    def test_map_fast_token_to_vlm_action(self):
        assert map_fast_token_to_vlm_action(["0", "1"]) == "<robot_action_0><robot_action_1>"

    def test_make_proprio_state_tokens(self):
        tokens = make_proprio_state_tokens(3)
        assert tokens == ["<proprio_state_0>", "<proprio_state_1>", "<proprio_state_2>"]

    def test_map_normalized_state_to_vlm_proprio_clamps_and_buckets(self):
        state = torch.tensor([-2.0, 0.0, 2.0])
        result = map_normalized_state_to_vlm_proprio(state, vocab_size=4)
        assert result.startswith("<proprio_state_")
        assert result.endswith(">")

    def test_map_normalized_state_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="proprio_vocab_size must be positive"):
            map_normalized_state_to_vlm_proprio(torch.tensor([0.0]), vocab_size=0)


class TestGriffinAlphaImageTransform:
    def test_get_random_crop_transform_size(self):
        crop = GriffinAlphaImageTransform.get_random_crop_transform((224, 224), 0.9)
        linear_scale = 0.9**0.5
        expected_h = round(224 * linear_scale)
        expected_w = round(224 * linear_scale)
        assert crop.size == (expected_h, expected_w)

    def test_call_preserves_channels(self):
        transform = GriffinAlphaImageTransform()
        image = torch.rand(3, 224, 224)
        output = transform(image)
        assert output.shape[0] == 3
        assert output.shape[1] <= 224
        assert output.shape[2] <= 224

    def test_get_center_crop_transform_size(self):
        crop = GriffinAlphaImageTransform.get_center_crop_transform((224, 224), 0.9)
        linear_scale = 0.9**0.5
        expected_h = round(224 * linear_scale)
        expected_w = round(224 * linear_scale)
        assert crop.size == (expected_h, expected_w)

    def test_center_crop_reduces_size_and_preserves_channels(self):
        transform = GriffinAlphaImageTransform()
        image = torch.rand(3, 224, 224)
        output = transform.center_crop(image)
        assert output.shape[0] == 3
        assert output.shape[1] < 224
        assert output.shape[2] < 224

