"""Processor tests for the Qwen3-VL backbone.

The action-space steps (relative/absolute, SE(3), resample, batch-dim) wired by
``make_griffin_alpha_fast_pre_post_processors`` are covered by ``test_action_steps.py``. What is
covered here is the prompt surface — vocab alignment, the dynamic-resolution pixel budget,
``do_rescale``, ChatML label masking, truncation exposure — plus the factory's own wiring.
"""

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
import logging

from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DataProcessorPipeline,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
)
from lerobot.processor import TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from lerobot_policy_griffin_alpha.backbone import QWEN_PATCH_MERGE
from lerobot_policy_griffin_alpha.configuration_griffin_alpha_fast import GriffinAlphaFASTConfig
from lerobot_policy_griffin_alpha.action_steps import (
    AbsoluteActionWithSE3ProcessorStep,
    GriffinAlphaAddBatchDimensionProcessorStep,
    RelativeActionWithSE3ProcessorStep,
    SE3MatrixToXYZRot6DProcessorStep,
    XYZRot6DToSE3MatrixProcessorStep,
)
from lerobot_policy_griffin_alpha.processor_griffin_alpha_fast import (
    GriffinAlphaFASTInputProcessorStep,
    make_griffin_alpha_fast_pre_post_processors,
)

_MODULE = "lerobot_policy_griffin_alpha.processor_griffin_alpha_fast"
_AUTOPROCESSOR = f"{_MODULE}.AutoProcessor.from_pretrained"


@pytest.fixture
def qwen3_vlm_step(
    griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
):
    with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
        with patch.object(
            GriffinAlphaFASTInputProcessorStep,
            "_make_vla_processor",
            return_value=qwen3_vl_vlm_processor,
        ):
            return GriffinAlphaFASTInputProcessorStep.from_config(
                griffin_alpha_fast_processor_config
            )


@pytest.fixture
def qwen3_vlm_step_mocks(fast_action_tokenizer, qwen3_vl_vlm_processor):
    return patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer), patch.object(
        GriffinAlphaFASTInputProcessorStep,
        "_make_vla_processor",
        return_value=qwen3_vl_vlm_processor,
    )


class TestVocabAlignment:
    """The highest-risk invariant in the port: action tokens must land on real embedding rows.

    Qwen's tokenizer assigns ids only up to 151668 while the model's table has 151936 rows. Without
    the placeholder padding, <robot_action_0> would sit at 151669 — on top of 267 untrained rows —
    and resize_token_embeddings would add 267 too few rows, with no crash and no warning.
    """

    def test_action_tokens_start_at_model_base_vocab(self, qwen3_vlm_step):
        tokenizer = qwen3_vlm_step._vla_processor.tokenizer
        assert tokenizer.convert_tokens_to_ids("<robot_action_0>") == 151936
        assert qwen3_vlm_step.tokenizer_vocab_pad_to == 151936

    def test_action_and_proprio_ranges_are_contiguous(self, qwen3_vlm_step):
        tokenizer = qwen3_vlm_step._vla_processor.tokenizer
        action_min = tokenizer.convert_tokens_to_ids("<robot_action_0>")
        action_max = tokenizer.convert_tokens_to_ids("<robot_action_2047>")
        proprio_min = tokenizer.convert_tokens_to_ids("<proprio_state_0>")
        proprio_max = tokenizer.convert_tokens_to_ids("<proprio_state_255>")
        assert action_max == action_min + 2047
        assert proprio_min == action_max + 1
        assert proprio_max == action_max + 256

    def test_pad_target_comes_from_config_action_token_min(
        self, griffin_alpha_fast_processor_config, qwen3_vlm_step
    ):
        """One source of truth — no second hardcoded 151936 to drift out of sync."""
        assert (
            qwen3_vlm_step.tokenizer_vocab_pad_to
            == griffin_alpha_fast_processor_config.action_token_min
        )

    def test_misalignment_is_rejected(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        """A pad target that disagrees with the tokenizer layout must fail loudly, not silently."""
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch.object(
                GriffinAlphaFASTInputProcessorStep,
                "_make_vla_processor",
                return_value=qwen3_vl_vlm_processor,
            ):
                with pytest.raises(ValueError, match="do not start at the model base vocab"):
                    GriffinAlphaFASTInputProcessorStep(tokenizer_vocab_pad_to=151000)

    def test_real_make_vla_processor_produces_aligned_tokenizer(self):
        """Exercises the genuine _make_vla_processor (no patching) — the padding really happens."""
        step = GriffinAlphaFASTInputProcessorStep()
        tokenizer = step._vla_processor.tokenizer
        assert tokenizer.convert_tokens_to_ids("<robot_action_0>") == 151936
        assert len(tokenizer) == 151936 + 2048 + 256


class TestImagePipeline:
    def test_pixel_budget_gives_the_documented_token_counts(self, qwen3_vlm_step):
        """480x640 (4:3) and 480x848 (~16:9) frames yield 63/66/66 merged tokens.

        Raising the cap from 70 to 72 (for 2:1) must not disturb them. If a transformers bump changes
        smart-resize, this test is the tripwire.
        """
        frames = [torch.rand(3, 480, 640), torch.rand(3, 480, 848), torch.rand(3, 480, 848)]
        grid = qwen3_vlm_step._vla_processor.image_processor(frames, return_tensors="pt")[
            "image_grid_thw"
        ].tolist()
        merged = [(t * h * w) // (QWEN_PATCH_MERGE // 16) ** 2 for t, h, w in grid]
        assert merged == [63, 66, 66]
        assert sum(merged) == 195
        # Real camera frames are all above the budget, so it acts as a cap for them.
        assert all(m <= qwen3_vlm_step.max_tokens_per_image for m in merged)

    @staticmethod
    def _merged_tokens(step, height: int, width: int) -> int:
        grid = step._vla_processor.image_processor(
            [torch.rand(3, height, width)], return_tensors="pt"
        )["image_grid_thw"].tolist()[0]
        return (grid[1] * grid[2]) // 4

    @pytest.mark.parametrize(
        ("ratio", "height", "width", "expected"),
        [
            ("1:1", 480, 480, 64),
            ("5:4", 480, 600, 63),
            ("4:3", 480, 640, 63),
            ("3:2", 480, 720, 60),
            ("16:10", 480, 768, 60),
            ("16:9", 1080, 1920, 66),
            ("16:9", 480, 854, 66),
            # 2:1's only aspect-preserving grids are 5x11=55 and 6x12=72, so it needs the cap at 72.
            ("2:1", 480, 960, 72),
            ("21:9", 480, 1120, 60),
            ("3:4", 480, 360, 63),
            ("9:16", 480, 270, 66),
        ],
    )
    def test_token_count_per_aspect_ratio(self, qwen3_vlm_step, ratio, height, width, expected):
        """The contract: 60-72 merged tokens, set by aspect ratio — 4:3 -> 63, 1:1 -> 64, 16:9 -> 66."""
        assert self._merged_tokens(qwen3_vlm_step, height, width) == expected

    @pytest.mark.parametrize("height", [240, 480, 720, 1080, 1440])
    def test_token_count_is_resolution_independent(self, qwen3_vlm_step, height):
        """A 4:3 frame yields 63 tokens at any capture resolution — only the ratio matters."""
        assert self._merged_tokens(qwen3_vlm_step, height, round(height * 4 / 3)) == 63

    def test_every_ratio_stays_in_band(self, qwen3_vlm_step):
        """No aspect ratio falls outside 60-72 — including the awkward 2:1 and portrait cases."""
        shapes = [(480, 480), (480, 600), (480, 640), (480, 720), (480, 768), (480, 854),
                  (480, 960), (480, 1120), (480, 360), (480, 270), (480, 600), (1080, 1920)]
        for height, width in shapes:
            merged = self._merged_tokens(qwen3_vlm_step, height, width)
            assert 60 <= merged <= qwen3_vlm_step.max_tokens_per_image, f"{height}x{width} -> {merged}"

    def test_undersized_frames_do_not_overshoot_the_cap(self, qwen3_vlm_step):
        """Regression: with min_pixels == max_pixels these took smart_resize's ceil branch and a
        64x64 frame came out at 81 tokens, above the nominal 70. The lower bound must stay strictly
        below the upper one."""
        for height, width in [(64, 64), (32, 32), (224, 224), (64, 48)]:
            merged = self._merged_tokens(qwen3_vlm_step, height, width)
            assert merged <= qwen3_vlm_step.max_tokens_per_image, f"{height}x{width} -> {merged}"
            assert merged >= 60, f"{height}x{width} -> {merged}"

    def test_min_must_stay_below_max(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with pytest.raises(ValueError, match="min_tokens_per_image"):
                GriffinAlphaFASTInputProcessorStep(
                    min_tokens_per_image=72, max_tokens_per_image=72
                )

    def test_pixel_bounds_come_from_the_token_band(self, qwen3_vlm_step):
        assert qwen3_vlm_step._vla_processor.image_processor.size == {
            "shortest_edge": qwen3_vlm_step.min_tokens_per_image * QWEN_PATCH_MERGE**2,
            "longest_edge": qwen3_vlm_step.max_tokens_per_image * QWEN_PATCH_MERGE**2,
        }

    def test_do_rescale_disabled_so_lerobot_floats_survive(self, qwen3_vlm_step):
        """lerobot hands over float32 [0,1]; a second 1/255 rescale would saturate every pixel."""
        image_processor = qwen3_vlm_step._vla_processor.image_processor
        assert image_processor.do_rescale is False

        pixel_values = image_processor(
            [torch.rand(3, 224, 224, generator=torch.Generator().manual_seed(0))],
            return_tensors="pt",
        )["pixel_values"].float()
        # A healthy spread is O(1); double-rescaling collapses it to <0.02.
        assert (pixel_values.max() - pixel_values.min()).item() > 1.0

    def test_augmentation_only_when_labels_and_grad(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor,
        qwen3_sample_training_transition, qwen3_sample_env_transition,
    ):
        # The shared helper builds integer image tensors; ColorJitter needs the production dtype
        # (lerobot hands over float32 in [0, 1]), so substitute float frames here.
        for transition in (qwen3_sample_training_transition, qwen3_sample_env_transition):
            observation = transition[TransitionKey.OBSERVATION]
            for key in ("observation.images.head",):
                observation[key] = [torch.rand(3, 64, 64)]

        config = replace(griffin_alpha_fast_processor_config, apply_image_augmentation=True)
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch.object(
                GriffinAlphaFASTInputProcessorStep,
                "_make_vla_processor",
                return_value=qwen3_vl_vlm_processor,
            ):
                step = GriffinAlphaFASTInputProcessorStep.from_config(config)

        spy = MagicMock(wraps=step._image_transform)
        step._image_transform = spy

        # Labels + grad enabled -> augment.
        with torch.enable_grad():
            step(qwen3_sample_training_transition)
        assert spy.call_count > 0

        # No labels -> never augment, even with grad enabled.
        spy.reset_mock()
        with torch.enable_grad():
            step(qwen3_sample_env_transition)
        assert spy.call_count == 0

        # Labels but grad disabled (eval) -> never augment.
        spy.reset_mock()
        with torch.no_grad():
            step(qwen3_sample_training_transition)
        assert spy.call_count == 0


class TestPromptAndLabels:
    def test_unlabeled_path_output_keys(self, qwen3_vlm_step, qwen3_sample_env_transition):
        batch_input = qwen3_vlm_step(qwen3_sample_env_transition)[TransitionKey.COMPLEMENTARY_DATA]
        assert "input_ids" in batch_input
        assert "attention_mask" in batch_input
        assert "image_grid_thw" in batch_input
        assert "pixel_values" in batch_input
        assert "labels" not in batch_input

    def test_labeled_path_masks_everything_before_assistant_turn(
        self, qwen3_vlm_step, qwen3_sample_training_transition
    ):
        batch_input = qwen3_vlm_step(qwen3_sample_training_transition)[
            TransitionKey.COMPLEMENTARY_DATA
        ]
        assert "labels" in batch_input
        labels = batch_input["labels"]
        input_ids = batch_input["input_ids"]
        turn_start_id = qwen3_vlm_step._turn_start_token_id
        assert turn_start_id == 151644  # <|im_start|>

        turn_indices = (input_ids[0] == turn_start_id).nonzero(as_tuple=False)
        assert turn_indices.numel() >= 2, "expected a user turn and an assistant turn"
        last_turn = turn_indices[-1].item()
        # Everything before the assistant turn is masked out of the loss...
        assert (labels[0, :last_turn] == -100).all()
        # ...and something after it is not (the action tokens are the training signal).
        assert (labels[0, last_turn:] != -100).any()

    def test_action_tokens_are_supervised(self, qwen3_vlm_step, qwen3_sample_training_transition):
        batch_input = qwen3_vlm_step(qwen3_sample_training_transition)[
            TransitionKey.COMPLEMENTARY_DATA
        ]
        labels = batch_input["labels"][0]
        supervised = labels[labels != -100]
        action_min = qwen3_vlm_step.tokenizer_vocab_pad_to
        action_max = action_min + qwen3_vlm_step.action_vocab_size - 1
        assert ((supervised >= action_min) & (supervised <= action_max)).any(), (
            "no FAST action token survived into the labels"
        )

    def test_proprio_tokens_in_prompt(self, qwen3_vlm_step, qwen3_sample_env_transition):
        batch_input = qwen3_vlm_step(qwen3_sample_env_transition)[TransitionKey.COMPLEMENTARY_DATA]
        decoded = qwen3_vlm_step._vla_processor.tokenizer.decode(batch_input["input_ids"][0])
        assert "<proprio_state_" in decoded

    def test_action_tokens_in_prompt(self, qwen3_vlm_step, qwen3_sample_training_transition):
        batch_input = qwen3_vlm_step(qwen3_sample_training_transition)[
            TransitionKey.COMPLEMENTARY_DATA
        ]
        decoded = qwen3_vlm_step._vla_processor.tokenizer.decode(batch_input["input_ids"][0])
        assert "<robot_action_" in decoded

    def test_conditioning_fields_in_prompt(self, qwen3_vlm_step, qwen3_sample_env_transition):
        batch_input = qwen3_vlm_step(qwen3_sample_env_transition)[TransitionKey.COMPLEMENTARY_DATA]
        decoded = qwen3_vlm_step._vla_processor.tokenizer.decode(batch_input["input_ids"][0])
        assert "embodiment: test_robot" in decoded
        assert "arm control mode: joint" in decoded
        assert "pick up the cup" in decoded

    def test_include_proprio_false_omits_proprio_tokens(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor,
        qwen3_sample_env_transition,
    ):
        config = replace(griffin_alpha_fast_processor_config, include_proprio=False)
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch.object(
                GriffinAlphaFASTInputProcessorStep,
                "_make_vla_processor",
                return_value=qwen3_vl_vlm_processor,
            ):
                step = GriffinAlphaFASTInputProcessorStep.from_config(config)

        batch_input = step(qwen3_sample_env_transition)[TransitionKey.COMPLEMENTARY_DATA]
        decoded = step._vla_processor.tokenizer.decode(batch_input["input_ids"][0])
        assert "<proprio_state_" not in decoded

    def test_missing_arm_control_mode_raises(self, qwen3_vlm_step, qwen3_sample_env_transition):
        qwen3_sample_env_transition[TransitionKey.INFO]["arm_control_mode"] = [None]
        with pytest.raises(ValueError, match="arm_control_mode must be provided"):
            qwen3_vlm_step(qwen3_sample_env_transition)

    def test_forwards_truncation_to_max_sequence_length(
        self, qwen3_vlm_step, qwen3_sample_env_transition
    ):
        spy = MagicMock(wraps=qwen3_vlm_step._vla_processor)
        qwen3_vlm_step._vla_processor = spy
        qwen3_vlm_step(qwen3_sample_env_transition)
        _, kwargs = spy.call_args
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == qwen3_vlm_step.max_sequence_length

    def _step_with_budget(
        self, base_config, fast_action_tokenizer, qwen3_vl_vlm_processor, max_sequence_length
    ):
        config = replace(base_config, max_sequence_length=max_sequence_length)
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch.object(
                GriffinAlphaFASTInputProcessorStep,
                "_make_vla_processor",
                return_value=qwen3_vl_vlm_processor,
            ):
                return GriffinAlphaFASTInputProcessorStep.from_config(config)

    def test_tail_truncation_cuts_the_action_chunk_and_warns(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor,
        qwen3_sample_training_transition, caplog,
    ):
        """The dangerous case: images survive, so nothing errors, but the FAST labels are clipped.

        This is why ``num_truncated_samples`` exists — an over-budget sample would otherwise train
        on a partial action chunk completely silently.
        """
        roomy = self._step_with_budget(
            griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor, 4096
        )
        import lerobot_policy_griffin_alpha.processor_griffin_alpha_fast as fast_module

        fast_module._WARNED_TRUNCATION = False
        with caplog.at_level(logging.WARNING, logger=fast_module.__name__):
            full = roomy(qwen3_sample_training_transition)[TransitionKey.COMPLEMENTARY_DATA]
        full_length = full["input_ids"].shape[1]
        assert "lost part of their FAST action tokens" not in caplog.text, "a roomy budget must not warn"

        # Trim a handful of tokens off the end: the image block and both turn starts survive, but
        # the assistant turn's closing <|im_end|> (and some action tokens) do not.
        tight = self._step_with_budget(
            griffin_alpha_fast_processor_config,
            fast_action_tokenizer,
            qwen3_vl_vlm_processor,
            full_length - 5,
        )
        with caplog.at_level(logging.WARNING, logger=fast_module.__name__):
            batch_input = tight(qwen3_sample_training_transition)[TransitionKey.COMPLEMENTARY_DATA]

        assert batch_input["input_ids"].shape[1] == full_length - 5
        input_ids = batch_input["input_ids"][0]
        last_turn = (input_ids == tight._turn_start_token_id).nonzero(as_tuple=False)[-1].item()
        # No closing <|im_end|> after the assistant turn start => the chunk was cut short.
        assert not (input_ids[last_turn:] == tight._turn_end_token_id).any()
        assert "1 sample(s) in this batch lost part of their FAST action tokens" in caplog.text

    def test_truncation_severe_enough_to_drop_image_tokens_raises(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor,
        qwen3_sample_training_transition,
    ):
        """Pins upstream behaviour: transformers rejects truncation that eats image placeholders.

        Good news — that failure mode is loud. Only tail truncation is silent (test above).
        """
        step = self._step_with_budget(
            griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor, 8
        )
        with pytest.raises(ValueError, match="Mismatch in `image` token count"):
            step(qwen3_sample_training_transition)


class TestSerialization:
    def test_round_trip_from_get_config(
        self, qwen3_vlm_step, qwen3_vlm_step_mocks, qwen3_sample_env_transition
    ):
        auto_patch, vla_patch = qwen3_vlm_step_mocks
        with auto_patch, vla_patch:
            restored = GriffinAlphaFASTInputProcessorStep(**qwen3_vlm_step.get_config())
            result = restored(qwen3_sample_env_transition)
        assert "input_ids" in result[TransitionKey.COMPLEMENTARY_DATA]
        assert qwen3_vlm_step.get_config() == restored.get_config()
        assert qwen3_vlm_step == restored

    def test_get_config_includes_qwen_specific_fields(self, qwen3_vlm_step):
        config = qwen3_vlm_step.get_config()
        assert config["tokenizer_vocab_pad_to"] == 151936
        assert config["max_tokens_per_image"] == 72
        assert config["base_vlm_processor_name"] == "Qwen/Qwen3-VL-4B-Instruct"

    def test_registered_under_its_own_registry_string(self):
        """Distinct from the flow step's string so saved pipelines never cross-resolve."""
        assert (
            ProcessorStepRegistry.get("griffinlabs/griffin_alpha_fast_input")
            is GriffinAlphaFASTInputProcessorStep
        )

    def test_serialization_methods(self, qwen3_vlm_step):
        assert qwen3_vlm_step.state_dict() == {}
        qwen3_vlm_step.load_state_dict({})


class TestMakePrePostProcessors:
    def _build(self, config, fast_action_tokenizer, qwen3_vl_vlm_processor, **kwargs):
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch.object(
                GriffinAlphaFASTInputProcessorStep,
                "_make_vla_processor",
                return_value=qwen3_vl_vlm_processor,
            ):
                return make_griffin_alpha_fast_pre_post_processors(config, **kwargs)

    def test_returns_two_named_pipelines(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        pre, post = self._build(
            griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
        )
        assert isinstance(pre, DataProcessorPipeline)
        assert isinstance(post, DataProcessorPipeline)
        assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME
        assert post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
        assert isinstance(pre.steps[1], GriffinAlphaAddBatchDimensionProcessorStep)
        assert isinstance(pre.steps[-2], GriffinAlphaFASTInputProcessorStep)

    def test_se3_relative_action_mask_wires_paired_steps(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        config = replace(
            griffin_alpha_fast_processor_config,
            use_relative_actions=False, se3_segment_start_idxs=[0], relative_action_mask=[True] * 17,
        )
        pre, post = self._build(config, fast_action_tokenizer, qwen3_vl_vlm_processor)

        relative = next(s for s in pre.steps if isinstance(s, RelativeActionWithSE3ProcessorStep))
        absolute = next(s for s in post.steps if isinstance(s, AbsoluteActionWithSE3ProcessorStep))
        # Same object, so the postprocessor reads the state the preprocessor cached.
        assert absolute.relative_step is relative
        # lerobot's built-in pair is still present, disabled.
        builtin = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
        assert builtin.enabled is False

    def test_builtin_relative_pair_is_present_and_enabled_by_default(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        pre, post = self._build(griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor)
        relative = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
        absolute = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))
        assert relative.enabled is True and absolute.enabled is True
        assert relative.exclude_joints == ["gripper"]
        assert absolute.relative_step is relative

    def test_builtin_relative_pair_can_be_disabled(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        config = replace(griffin_alpha_fast_processor_config, use_relative_actions=False)
        pre, post = self._build(config, fast_action_tokenizer, qwen3_vl_vlm_processor)
        relative = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
        absolute = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))
        assert relative.enabled is False and absolute.enabled is False

    def test_relative_action_mask_without_se3_is_rejected(self, griffin_alpha_fast_processor_config):
        with pytest.raises(ValueError, match="SE\\(3\\) path"):
            replace(griffin_alpha_fast_processor_config, relative_action_mask=[True] * 7)

    def test_se3_with_builtin_relative_is_rejected(self, griffin_alpha_fast_processor_config):
        with pytest.raises(ValueError, match="use_relative_actions=False"):
            replace(griffin_alpha_fast_processor_config, se3_segment_start_idxs=[0])

    def test_se3_is_supported(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        """The SE(3) path is available on the FAST head too."""
        config = replace(
            griffin_alpha_fast_processor_config,
            output_features=griffin_alpha_fast_processor_config.output_features,
            use_relative_actions=False,
            se3_segment_start_idxs=[0],
        )
        pre, post = self._build(config, fast_action_tokenizer, qwen3_vl_vlm_processor)
        assert any(isinstance(s, SE3MatrixToXYZRot6DProcessorStep) for s in pre.steps)
        assert any(isinstance(s, XYZRot6DToSE3MatrixProcessorStep) for s in post.steps)

    def test_se3_plus_resample_is_rejected(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        # Rejected by the config itself, before any pipeline is built.
        with pytest.raises(ValueError, match="Resampling and SE\\(3\\)"):
            replace(
                griffin_alpha_fast_processor_config,
                use_relative_actions=False,
                se3_segment_start_idxs=[0],
                resample_action_chunk_size=8,
            )

    def test_no_relative_mask_means_no_action_space_steps(
        self, griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
    ):
        pre, post = self._build(
            griffin_alpha_fast_processor_config, fast_action_tokenizer, qwen3_vl_vlm_processor
        )
        assert not any(isinstance(s, RelativeActionWithSE3ProcessorStep) for s in pre.steps)
        assert not any(isinstance(s, AbsoluteActionWithSE3ProcessorStep) for s in post.steps)
