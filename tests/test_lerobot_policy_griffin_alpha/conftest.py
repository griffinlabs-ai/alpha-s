import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import EnvTransition
from lerobot.utils.constants import ACTION
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from lerobot_policy_griffin_alpha.backbone import QWEN_PATCH_MERGE
from lerobot_policy_griffin_alpha.configuration_griffin_alpha import GriffinAlphaConfig
from lerobot_policy_griffin_alpha.configuration_griffin_alpha_fast import GriffinAlphaFASTConfig
from lerobot_policy_griffin_alpha.prompt_utils import make_proprio_state_tokens

from .helpers import make_sample_env_transition


@pytest.fixture
def tiny_qwen3vl_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        architectures=["Qwen3VLForConditionalGeneration"],
        text_config={
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 1000,
            "head_dim": 16,
            "dtype": "bfloat16",
        },
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "intermediate_size": 128,
            "depth": 4,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            # Qwen3-VL injects visual features at these text layers (the real 4B uses [5, 11, 17]).
            "deepstack_visual_indexes": [1, 2],
            "dtype": "bfloat16",
        },
        # The real Qwen3-VL-4B ties lm_head to embed_tokens; mirror that so the tied-weight
        # deduplicating safetensors save is actually exercised.
        tie_word_embeddings=True,
        dtype="bfloat16",
    )


@pytest.fixture
def tiny_qwen3vl_model(tiny_qwen3vl_config: Qwen3VLConfig) -> Qwen3VLForConditionalGeneration:
    # Mirror a real checkpoint loaded via HF from_pretrained, which is fully bfloat16 (the plain
    # constructor would otherwise leave some projection layers in float32).
    return Qwen3VLForConditionalGeneration(tiny_qwen3vl_config).to(torch.bfloat16)


_TINY_POLICY_KWARGS = dict(
    output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    image_keys=("observation.images.head",),
    gradient_checkpointing=False,
    device="cpu",
    horizon=4,
    n_action_steps=4,
    apply_image_augmentation=False,
)


@pytest.fixture
def griffin_alpha_fast_config(tiny_qwen3vl_config: Qwen3VLConfig) -> GriffinAlphaFASTConfig:
    """Tiny FAST config for the modeling tests (no real processor involved)."""
    return GriffinAlphaFASTConfig(qwen3vl_config=tiny_qwen3vl_config, **_TINY_POLICY_KWARGS)


@pytest.fixture
def griffin_alpha_config() -> GriffinAlphaConfig:
    """Tiny flow config for tests that only need a config (action steps, transitions)."""
    return GriffinAlphaConfig(**_TINY_POLICY_KWARGS)


@pytest.fixture
def griffin_alpha_fast_processor_config() -> GriffinAlphaFASTConfig:
    """Config for the processor tests.

    Deliberately keeps the *real* ``action_token_min`` (151936), because the processor step asserts
    that the injected action tokens line up with the model's base vocab -- the single highest-risk
    invariant of the vocabulary layout. A tiny stand-in vocab would defeat that check.
    """
    return GriffinAlphaFASTConfig(**_TINY_POLICY_KWARGS)


@pytest.fixture(scope="session")
def fast_action_tokenizer():
    return AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer", trust_remote_code=True)


@pytest.fixture(scope="session")
def qwen3_vl_vlm_processor():
    """The real Qwen3-VL processor, configured exactly as the policy's processor step does.

    Built once per session and injected in place of ``_make_vla_processor`` so the tests exercise
    the genuine tokenizer id layout and the genuine dynamic-resolution image pipeline.
    """
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", trust_remote_code=True)
    # Mirror GriffinAlphaPromptStep._make_vla_processor: a token *band*, not a single value. min must
    # stay strictly below max or undersized frames overshoot the cap (ceil branch).
    processor.image_processor.size = {
        "shortest_edge": 60 * QWEN_PATCH_MERGE**2,
        "longest_edge": 72 * QWEN_PATCH_MERGE**2,
    }
    processor.image_processor.do_rescale = False

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    # Pad up to the model's base vocab (151936) so the action tokens start exactly there.
    n_pad = 151936 - len(processor.tokenizer)
    if n_pad > 0:
        processor.tokenizer.add_tokens([f"<|vla_unused_{i}|>" for i in range(n_pad)], special_tokens=True)
    action_tokens = [f"<robot_action_{i}>" for i in range(2048)]
    proprio_tokens = make_proprio_state_tokens(256)
    processor.tokenizer.add_tokens(action_tokens + proprio_tokens, special_tokens=True)
    return processor


@pytest.fixture
def sample_env_transition(griffin_alpha_config: GriffinAlphaConfig) -> EnvTransition:
    return make_sample_env_transition(griffin_alpha_config, batched=True, with_action=False)


@pytest.fixture
def qwen3_sample_env_transition(griffin_alpha_fast_processor_config: GriffinAlphaFASTConfig) -> EnvTransition:
    return make_sample_env_transition(griffin_alpha_fast_processor_config, batched=True, with_action=False)


@pytest.fixture
def qwen3_sample_training_transition(griffin_alpha_fast_processor_config: GriffinAlphaFASTConfig) -> EnvTransition:
    return make_sample_env_transition(griffin_alpha_fast_processor_config, batched=True, with_action=True)
