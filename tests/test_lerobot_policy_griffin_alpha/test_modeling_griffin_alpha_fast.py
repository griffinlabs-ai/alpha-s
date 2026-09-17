"""Modeling tests for the FAST head (``GriffinAlphaFASTPolicy``) and the shared Qwen3-VL backbone
plumbing: attention-backend resolution, the tied embedding table, ``image_grid_thw`` handling, the
FAST decode helpers, and save/load round trips.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from lerobot.utils.constants import ACTION
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from lerobot_policy_griffin_alpha.backbone import (
    ATTN_FALLBACK,
    FLASH_ATTENTION_2,
    MODEL_INPUT_KEYS,
    resolve_attn_implementation,
)
from lerobot_policy_griffin_alpha.model_utils import DEFAULT_MAX_NEW_TOKENS, resolve_dtype as _resolve_dtype
from lerobot_policy_griffin_alpha.modeling_griffin_alpha_fast import GriffinAlphaFASTPolicy

_AUTOPROCESSOR = (
    "lerobot_policy_griffin_alpha.modeling_griffin_alpha_fast.AutoProcessor.from_pretrained"
)


@pytest.fixture
def qwen3_policy(griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer):
    with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
        policy = GriffinAlphaFASTPolicy(
            griffin_alpha_fast_config, qwen3vl_model=tiny_qwen3vl_model
        )
    return policy


class TestModelInputKeys:
    def test_filters_unknown_keys_and_none(self):
        batch = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "labels": None,
            "unknown_key": torch.tensor([0]),
        }
        filtered = GriffinAlphaFASTPolicy._filter_model_inputs(batch)
        assert set(filtered.keys()) == {"input_ids", "attention_mask"}

    def test_carries_image_grid_thw(self):
        """Qwen3-VL is dynamic-resolution: the per-image grid travels with the batch."""
        assert "image_grid_thw" in MODEL_INPUT_KEYS
        assert "image_position_ids" not in MODEL_INPUT_KEYS
        assert {"input_ids", "pixel_values", "attention_mask", "labels"} <= MODEL_INPUT_KEYS

    def test_every_key_is_accepted_by_forward(self):
        """Guards against a transformers bump renaming a forward kwarg out from under us."""
        import inspect

        accepted = set(inspect.signature(Qwen3VLForConditionalGeneration.forward).parameters)
        assert MODEL_INPUT_KEYS <= accepted, MODEL_INPUT_KEYS - accepted


class TestResolveAttnImplementation:
    """FlashAttention 2 resolution.

    These run identically with or without flash-attn installed: every test that depends on
    availability patches it explicitly rather than reading the host. They cover the resolution logic
    and the plumbing, never a real FA2 forward pass — that needs an Ampere-or-newer GPU.
    """

    _MODULE = "lerobot_policy_griffin_alpha.backbone"

    def test_explicit_backends_pass_through_untouched(self):
        assert resolve_attn_implementation("sdpa") == "sdpa"
        assert resolve_attn_implementation("eager") == "eager"
        # Even with require=True, a non-FA2 request is never second-guessed.
        assert resolve_attn_implementation("eager", require=True) == "eager"

    def test_flash_attention_2_used_when_available(self):
        with patch(f"{self._MODULE}.is_flash_attn_2_available", return_value=True):
            assert resolve_attn_implementation(FLASH_ATTENTION_2) == FLASH_ATTENTION_2

    def test_falls_back_to_sdpa_when_unavailable(self, caplog):
        with patch(f"{self._MODULE}.is_flash_attn_2_available", return_value=False):
            with caplog.at_level("WARNING"):
                assert resolve_attn_implementation(FLASH_ATTENTION_2) == ATTN_FALLBACK
        # The fallback must be loud: a run silently training at SDPA speed is the failure mode.
        assert any("FlashAttention 2 requested but unavailable" in r.message for r in caplog.records)

    def test_require_raises_instead_of_falling_back(self):
        with patch(f"{self._MODULE}.is_flash_attn_2_available", return_value=False):
            with pytest.raises(RuntimeError, match="flash-attn is not available"):
                resolve_attn_implementation(FLASH_ATTENTION_2, require=True)

    def test_config_default_requests_flash_attention_2(self):
        from lerobot_policy_griffin_alpha import GriffinAlphaFASTConfig

        config = GriffinAlphaFASTConfig()
        assert config.attn_implementation == FLASH_ATTENTION_2
        # Off by default so the package stays importable without flash-attn.
        assert config.require_attn_implementation is False


class TestAttnImplementationPlumbing:
    def test_policy_records_the_backend_actually_in_use(self, qwen3_policy):
        """The config is rewritten to the resolved value, so a checkpoint never over-claims FA2."""
        resolved = qwen3_policy.config.attn_implementation
        assert resolved == getattr(qwen3_policy.model.config, "_attn_implementation", None)
        # Environment-agnostic: FA2 where flash-attn is installed (the H100 boxes), the fallback
        # where it is not (CPU runners). Hardcoding either makes this test lie on the other.
        assert resolved == resolve_attn_implementation(FLASH_ATTENTION_2)
        assert resolved in {FLASH_ATTENTION_2, ATTN_FALLBACK}

    def test_explicit_sdpa_reaches_the_inner_model(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        config = replace(griffin_alpha_fast_config, attn_implementation="sdpa")
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(config, qwen3vl_model=tiny_qwen3vl_model)
        assert policy.model.config._attn_implementation == "sdpa"
        assert policy.config.attn_implementation == "sdpa"

    def test_require_flag_fails_construction_when_unavailable(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        """Forces FA2 unavailable rather than relying on the host lacking flash-attn — otherwise
        this passes vacuously on a CPU runner and fails outright on a box that has it."""
        config = replace(
            griffin_alpha_fast_config,
            attn_implementation=FLASH_ATTENTION_2,
            require_attn_implementation=True,
        )
        module = "lerobot_policy_griffin_alpha.backbone"
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch(f"{module}.is_flash_attn_2_available", return_value=False):
                with pytest.raises(RuntimeError, match="flash-attn is not available"):
                    GriffinAlphaFASTPolicy(config, qwen3vl_model=tiny_qwen3vl_model)

    def test_require_flag_succeeds_when_flash_attn_is_present(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        """The other half: with FA2 available, require=True must be a no-op, not an obstacle."""
        config = replace(
            griffin_alpha_fast_config,
            attn_implementation=FLASH_ATTENTION_2,
            require_attn_implementation=True,
        )
        module = "lerobot_policy_griffin_alpha.backbone"
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with patch(f"{module}.is_flash_attn_2_available", return_value=True):
                # Only the resolution is under test; building the model with a real FA2 kernel is
                # covered by the on-GPU run, not here.
                assert (
                    resolve_attn_implementation(FLASH_ATTENTION_2, require=True)
                    == FLASH_ATTENTION_2
                )

    def test_reload_path_applies_the_backend_before_construction(
        self, griffin_alpha_fast_config, fast_action_tokenizer
    ):
        """No model handed in: the backend must be set on the inner config, not switched after."""
        config = replace(griffin_alpha_fast_config, attn_implementation="sdpa")
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(config)
        assert policy.model.config._attn_implementation == "sdpa"

    def test_backend_reaches_both_text_and_vision_stacks(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        """What makes "FA2 covers the whole model" true rather than aspirational.

        Qwen3-VL has two sub-configs (text_config, vision_config) and each drives its own attention
        modules. If the backend only reached the text stack, the vision tower would silently stay on
        another kernel and the config would over-claim.
        """
        config = replace(griffin_alpha_fast_config, attn_implementation="sdpa")
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(config, qwen3vl_model=tiny_qwen3vl_model)

        inner = policy.model.config
        assert inner.text_config._attn_implementation == "sdpa"
        assert inner.vision_config._attn_implementation == "sdpa"
        assert policy.model.model.visual.config._attn_implementation == "sdpa"
        assert policy.model.model.language_model.config._attn_implementation == "sdpa"

    def test_backend_survives_save_load_round_trip(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer, tmp_path
    ):
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy.from_qwen3vl_model(
                tiny_qwen3vl_model, config=griffin_alpha_fast_config
            )
            policy.save_pretrained(tmp_path)
            reloaded = GriffinAlphaFASTPolicy.from_pretrained(tmp_path)
        assert reloaded.config.attn_implementation == policy.config.attn_implementation


class TestResolveDtype:
    def test_passthrough_and_none(self):
        assert _resolve_dtype(None) is None
        assert _resolve_dtype(torch.bfloat16) is torch.bfloat16

    def test_string_forms(self):
        assert _resolve_dtype("bfloat16") is torch.bfloat16
        assert _resolve_dtype("float32") is torch.float32

    def test_unknown_string_is_none(self):
        assert _resolve_dtype("not_a_dtype") is None


class TestGriffinAlphaFASTPolicy:
    def test_reset_clears_action_queue(self, qwen3_policy):
        qwen3_policy._action_queue.append(torch.zeros(7))
        qwen3_policy.reset()
        assert len(qwen3_policy._action_queue) == 0

    def test_no_audio_tower(self, qwen3_policy):
        """Qwen3-VL has no audio stack; nothing may assume one."""
        assert not hasattr(qwen3_policy.model.model, "audio_tower")

    def test_no_per_layer_embedding_table(self, qwen3_policy):
        """Qwen3-VL has a single embedding table; resize_token_embeddings covers it fully."""
        language_model = qwen3_policy.model.model.language_model
        assert not hasattr(language_model, "embed_tokens_per_layer")

    def test_default_trains_everything(self, qwen3_policy):
        optim_params = qwen3_policy.get_optim_params()
        assert len(optim_params) > 0
        assert all(p.requires_grad for p in optim_params)

    def test_freeze_vision_tower(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        config = replace(griffin_alpha_fast_config, freeze_vision_tower=True)
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(config, qwen3vl_model=tiny_qwen3vl_model)

        visual_params = list(policy.model.model.visual.parameters())
        assert len(visual_params) > 0
        assert not any(p.requires_grad for p in visual_params)
        # The language model stays trainable.
        assert any(p.requires_grad for p in policy.model.model.language_model.parameters())
        frozen_ids = {id(p) for p in visual_params}
        assert {id(p) for p in policy.get_optim_params()}.isdisjoint(frozen_ids)

    def test_freeze_embeddings_also_freezes_tied_head(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        config = replace(griffin_alpha_fast_config, freeze_embeddings=True)
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(config, qwen3vl_model=tiny_qwen3vl_model)

        assert not policy.model.model.language_model.embed_tokens.weight.requires_grad
        assert not policy.model.lm_head.weight.requires_grad

    def test_decode_action_tokens_empty_sequence(self, qwen3_policy):
        generated_ids = torch.tensor([[1, 2, 3, 4]])
        actions = qwen3_policy._decode_action_tokens(generated_ids)
        horizon = qwen3_policy.config.horizon
        action_dim = qwen3_policy.config.output_features[ACTION].shape[0]
        assert actions.shape == (1, horizon, action_dim)
        assert torch.all(actions == 0)

    def test_decode_action_tokens_with_action_tokens(self, qwen3_policy, fast_action_tokenizer):
        """The FAST decode helper reused from GriffinAlphaPolicy must behave identically here."""
        horizon = qwen3_policy.config.horizon
        action_dim = qwen3_policy.config.output_features[ACTION].shape[0]
        sample_action = torch.randn(1, horizon, action_dim)
        fast_action_tokenizer.action_dim = action_dim
        fast_action_tokenizer.time_horizon = horizon
        fast_ids = fast_action_tokenizer(sample_action)[0]
        reference_decoded = torch.as_tensor(
            fast_action_tokenizer.decode([fast_ids])[0], dtype=torch.float32
        )

        token_min = qwen3_policy.config.action_token_min
        generated_ids = torch.tensor([[0, 1, 2] + [token_min + int(t) for t in fast_ids]])

        decoded = qwen3_policy._decode_action_tokens(generated_ids)
        assert decoded.shape == (1, horizon, action_dim)
        assert not torch.all(decoded == 0)
        assert torch.allclose(decoded, reference_decoded)

    def test_forward_returns_loss(self, qwen3_policy):
        fake_loss = torch.tensor(1.5, requires_grad=True)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        with patch.object(qwen3_policy.model, "forward", return_value=SimpleNamespace(loss=fake_loss)):
            loss, metrics = qwen3_policy.forward(batch)
        assert loss == fake_loss
        assert loss.requires_grad
        assert metrics["loss"] == pytest.approx(1.5)

    def test_select_action_drains_queue(self, qwen3_policy):
        n_steps = qwen3_policy.config.n_action_steps
        horizon = qwen3_policy.config.horizon
        action_dim = qwen3_policy.config.max_action_dim
        qwen3_policy.predict_action_chunk = MagicMock(
            return_value=torch.randn(1, horizon, action_dim)
        )

        actions = [qwen3_policy.select_action({"input_ids": torch.tensor([[1]])}) for _ in range(n_steps)]

        assert qwen3_policy.predict_action_chunk.call_count == 1
        for action in actions:
            assert action.shape == (1, action_dim)

    def test_predict_action_chunk_resamples_to_native_chunk_size(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        native_chunk_size = griffin_alpha_fast_config.horizon + 2
        config = replace(griffin_alpha_fast_config, resample_action_chunk_size=native_chunk_size)
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(config, qwen3vl_model=tiny_qwen3vl_model)

        action_dim = policy.config.max_action_dim
        decoded_chunk = torch.randn(1, policy.config.horizon, action_dim)
        with patch.object(policy.model, "generate", return_value=torch.tensor([[1, 2, 3]])):
            with patch.object(policy, "_decode_action_tokens", return_value=decoded_chunk):
                chunk = policy.predict_action_chunk({"input_ids": torch.tensor([[1]])})

        assert chunk.shape == (1, native_chunk_size, action_dim)

    def test_save_and_load_round_trip(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer, tmp_path
    ):
        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy.from_qwen3vl_model(
                tiny_qwen3vl_model, config=griffin_alpha_fast_config
            )
            expected_vocab = policy.config.qwen3vl_config.text_config.vocab_size
            policy.save_pretrained(tmp_path)
            reloaded = GriffinAlphaFASTPolicy.from_pretrained(tmp_path)

        assert isinstance(reloaded, GriffinAlphaFASTPolicy)
        assert reloaded.config.qwen3vl_config.text_config.vocab_size == expected_vocab

        assert (tmp_path / "generation_config.json").is_file()
        assert reloaded.model.generation_config.do_sample is False
        assert reloaded.model.generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS

        # Reload rebuilds the model with the plain (float32) constructor, so this guards the cast
        # back to the config dtype.
        assert all(p.dtype == torch.bfloat16 for p in policy.parameters())
        assert all(p.dtype == torch.bfloat16 for p in reloaded.parameters())

        orig_sd = policy.state_dict()
        reloaded_sd = reloaded.state_dict()
        assert orig_sd.keys() == reloaded_sd.keys()
        for key in orig_sd:
            assert torch.equal(orig_sd[key].cpu(), reloaded_sd[key].cpu()), key

    def test_save_dedups_tied_lm_head(
        self, griffin_alpha_fast_config, tiny_qwen3vl_model, fast_action_tokenizer, tmp_path
    ):
        """Qwen3-VL ties lm_head to embed_tokens; the saver must not write the alias twice."""
        from safetensors import safe_open

        assert (
            tiny_qwen3vl_model.lm_head.weight.data_ptr()
            == tiny_qwen3vl_model.model.language_model.embed_tokens.weight.data_ptr()
        ), "fixture should mirror the real 4B's tied embeddings"

        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy(
                griffin_alpha_fast_config, qwen3vl_model=tiny_qwen3vl_model
            )
            policy.save_pretrained(tmp_path)

        with safe_open(tmp_path / "model.safetensors", framework="pt") as f:
            keys = set(f.keys())
        embed_key = "model.model.language_model.embed_tokens.weight"
        head_key = "model.lm_head.weight"
        # Exactly one of the tied pair is persisted (the saver keeps the lm_head-style name).
        assert (embed_key in keys) ^ (head_key in keys), sorted(k for k in keys if "embed_tokens" in k or "lm_head" in k)

    def test_from_qwen3vl_model_resizes_and_derives_token_range(
        self, tiny_qwen3vl_model, tiny_qwen3vl_config, fast_action_tokenizer
    ):
        base_vocab = tiny_qwen3vl_config.text_config.vocab_size
        target_vocab_size = base_vocab + 2048 + 256

        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy.from_qwen3vl_model(tiny_qwen3vl_model)

        assert isinstance(policy, GriffinAlphaFASTPolicy)
        assert policy.config.qwen3vl_config is tiny_qwen3vl_config
        assert policy.config.qwen3vl_config.text_config.vocab_size == target_vocab_size
        assert policy.model is tiny_qwen3vl_model
        assert policy.fast_tokenizer is fast_action_tokenizer
        # Derived from the base vocab, not the 151936 default.
        assert policy.config.action_token_min == base_vocab
        assert policy.config.action_token_max == base_vocab + 2048 - 1
        # The embedding table really grew (Qwen has a single tied table — no separate PLE resize).
        assert (
            policy.model.model.language_model.embed_tokens.weight.shape[0] == target_vocab_size
        )
        assert all(p.dtype == torch.bfloat16 for p in policy.parameters())
        assert policy.model.generation_config.do_sample is False
        assert policy.model.generation_config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS

    def test_from_qwen3vl_model_no_resize_keeps_vocab_and_derives_token_range(
        self, tiny_qwen3vl_model, fast_action_tokenizer
    ):
        action_vocab_size, proprio_vocab_size = 2048, 256
        base_vocab = tiny_qwen3vl_model.config.text_config.vocab_size
        # Simulate a checkpoint whose embeddings were already resized.
        adapted_vocab = base_vocab + action_vocab_size + proprio_vocab_size
        tiny_qwen3vl_model.resize_token_embeddings(adapted_vocab)

        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            policy = GriffinAlphaFASTPolicy.from_qwen3vl_model(
                tiny_qwen3vl_model, resize_embeddings=False
            )

        assert policy.model is tiny_qwen3vl_model
        assert policy.config.qwen3vl_config.text_config.vocab_size == adapted_vocab
        assert policy.config.action_token_min == base_vocab
        assert policy.config.action_token_max == base_vocab + action_vocab_size - 1

    def test_from_qwen3vl_model_no_resize_rejects_unadapted_model(
        self, tiny_qwen3vl_config, fast_action_tokenizer
    ):
        small_config = Qwen3VLConfig.from_dict(tiny_qwen3vl_config.to_dict())
        small_config.text_config.vocab_size = 100
        small_model = Qwen3VLForConditionalGeneration(small_config).to(torch.bfloat16)

        with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
            with pytest.raises(ValueError, match="does not appear to be an adapted"):
                GriffinAlphaFASTPolicy.from_qwen3vl_model(small_model, resize_embeddings=False)

    def test_from_qwen3vl_pretrained_converts(
        self, tiny_qwen3vl_model, tiny_qwen3vl_config, fast_action_tokenizer
    ):
        target_vocab_size = tiny_qwen3vl_config.text_config.vocab_size + 2048 + 256
        module = "lerobot_policy_griffin_alpha.backbone"

        with patch(f"{module}.AutoConfig.from_pretrained", return_value=tiny_qwen3vl_config):
            with patch(
                f"{module}.Qwen3VLForConditionalGeneration.from_pretrained",
                return_value=tiny_qwen3vl_model,
            ):
                with patch(_AUTOPROCESSOR, return_value=fast_action_tokenizer):
                    policy = GriffinAlphaFASTPolicy.from_qwen3vl_pretrained("fake/path")

        assert isinstance(policy, GriffinAlphaFASTPolicy)
        assert policy.config.qwen3vl_config is tiny_qwen3vl_config
        assert policy.config.qwen3vl_config.text_config.vocab_size == target_vocab_size
        assert policy.model is tiny_qwen3vl_model
