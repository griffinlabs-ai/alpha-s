# Checkpoints

Two Hugging Face model repositories, each with two branches. The default head is the flow-matching
expert (`main`); the FAST-token head is the `fast` branch of the same repository.

| repository | `main` | `fast` |
|---|---|---|
| [`griffinlabs/Griffin-Alpha-S`](https://huggingface.co/griffinlabs/Griffin-Alpha-S) | pre-trained base, `griffin_alpha` | pre-trained base, `griffin_alpha_fast` |
| [`griffinlabs/Griffin-Alpha-S-LIBERO`](https://huggingface.co/griffinlabs/Griffin-Alpha-S-LIBERO) | LIBERO fine-tune, `griffin_alpha` | LIBERO fine-tune, `griffin_alpha_fast` |

Select the FAST head with `revision="fast"` in `from_pretrained`, or by downloading the branch for the
lerobot CLIs (see the note below). Weights are released under CC BY-NC-SA 4.0 (see each repository's LICENSE); the
code in this repository is Apache-2.0.

> **Selecting the `fast` branch.** The lerobot CLIs (`lerobot-eval`, `lerobot-train`) read `config.json`
> from `--policy.path` and apply `--policy.pretrained_revision` only to the weights and processors, so
> pointing them at a Hub id plus a branch mixes the two heads. Download the branch first and pass the
> directory:
>
> ```bash
> hf download griffinlabs/Griffin-Alpha-S-LIBERO --revision fast --local-dir ckpts/libero-fast
> lerobot-train --policy.path=ckpts/libero-fast ...
> ```
>
> The Python API is unaffected: `from_pretrained(repo, revision="fast")` and
> `make_pre_post_processors(cfg, pretrained_path=repo, pretrained_revision="fast")` load the right head.

## What is in a checkpoint

A checkpoint is a standard lerobot policy directory:

| file | contents |
|---|---|
| `config.json` | the policy config; its `type` field selects `griffin_alpha` or `griffin_alpha_fast` |
| `model.safetensors` | backbone (+ expert) weights, bf16 |
| `generation_config.json` | greedy decoding settings (used by the FAST head) |
| `policy_preprocessor.json` + `..._normalizer_processor.safetensors` | the input pipeline and its normalization stats |
| `policy_postprocessor.json` + `..._unnormalizer_processor.safetensors` | the output pipeline and its stats |

The processor JSON records every step by registry name (`griffinlabs/griffin_alpha_input`,
`relative_actions_processor`, ...) plus that step's config, so a checkpoint carries its own camera
keys, prompt fields, and action-space transform. See `docs/finetuning.md` for how to change them.

### The bases

- Trained on a multi-embodiment mixture with a canonical 32-dimensional action vector (each
  embodiment's actions occupy the leading slots, the rest is zero padding). The flow head is 32 wide
  (`action_dim_override=32`); the FAST tokenizer decodes at the width your dataset supplies.
- `use_relative_actions=true` with `relative_exclude_joints=["gripper"]`: arm dimensions are
  predicted relative to the current `observation.state`, grippers absolute. The normalization stats
  shipped with a base are placeholders; they are replaced by your dataset's when you fine-tune.
- `image_keys` is empty, meaning "every visual feature of the dataset, in order". The prompt order
  matters: pre-training put the primary (exocentric) camera first, then wrist cameras.
- The flow base's expert was pre-trained on a frozen backbone; the public default for fine-tuning
  from it is joint training (`freeze_backbone=false`).

### The LIBERO fine-tunes

- Cameras `observation.images.image` (agent view) and `observation.images.image2` (wrist), 256x256;
  8-D state; 7-D OSC_POSE deltas, `use_relative_actions=false` because those deltas are already
  relative commands; `n_action_steps=10`; embodiment prompt "LIBERO simulated Franka Emika Panda,
  1 gripper", `arm_control_mode="eef_pose"`.
- The flow fine-tune has `include_proprio=true`, `condition_on_subtask=true` and `num_inference_steps=1`
  (one Euler step matched ten on LIBERO; the base keeps 10); the FAST fine-tune has `include_proprio=false`.

## The config contract

`config.json` is decoded strictly: a key the installed plugin's config dataclass does not define is an
error, not a warning. Consequences:

- A checkpoint written by a newer plugin version that added a field will not load in an older
  plugin. Upgrade the plugin.
- Conversely, every field this plugin adds carries a default, so older checkpoints of the same
  major version keep loading.
- If you edit `config.json` by hand, only use keys that exist on the config class.

`tests/test_lerobot_policy_griffin_alpha/test_checkpoint_contract.py` pins this behaviour, and pins
that the processor pipelines a fresh factory builds are the ones the checkpoints carry.

## Attention backend

The configs request `attn_implementation="flash_attention_2"`. When flash-attn is not installed the
policy falls back to SDPA and logs a warning; the field is rewritten to what is actually in use before
a checkpoint is saved. Set `--policy.require_attn_implementation=true` on a training job to fail
instead of falling back.

## Transformers version pin

The flow head builds its expert from transformers' own Qwen3-VL pieces and captures the backbone's
per-layer K/V by plugging into transformers' attention registries. Those are private APIs that can move
between minor versions:

- `transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS` and
  `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`
- `transformers.models.qwen3_vl.modeling_qwen3_vl.{Qwen3VLTextDecoderLayer, Qwen3VLTextRMSNorm,
  apply_rotary_pos_emb, eager_attention_forward}`

This is why `pyproject.toml` pins `transformers>=5.5.4,<5.6`. Raising the ceiling means re-checking those
four imports in `modeling_griffin_alpha.py` and re-running the test suite.
