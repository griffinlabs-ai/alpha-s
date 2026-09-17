# Fine-tuning

Griffin Alpha-S is a LeRobot policy plugin, so fine-tuning is stock `lerobot-train`. Install the package
with its `train` extra (`pip install 'lerobot_policy_griffin_alpha[train] @ git+https://github.com/griffinlabs-ai/alpha-s.git'`,
which adds lerobot's own training dependencies such as `accelerate`) and lerobot finds the two policy
types (`griffin_alpha`, `griffin_alpha_fast`) on its own.

Video decoding: lerobot decodes dataset videos with `torchcodec`, which needs an FFmpeg build it can
link against (LeRobot's docs recommend the conda-forge `ffmpeg`). When it cannot load, lerobot falls
back to `pyav` automatically; training works either way, only slower.

## From a LIBERO fine-tune, on a LIBERO-like dataset

If your dataset has the same cameras and action space as the published LIBERO checkpoints (keys
`observation.images.image` / `observation.images.image2`, 8-D state, 7-D per-step deltas), you can
start directly from the checkpoint:

```bash
lerobot-train \
  --policy.path=griffinlabs/Griffin-Alpha-S-LIBERO \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.device=cuda --batch_size=16 --steps=6000 \
  --output_dir=outputs/train/alpha-s-libero-ft
```

`lerobot-train` reloads the checkpoint's processors and replaces their normalization statistics
with your dataset's. Everything else in the processors (camera keys, prompt, action transform) stays
as saved. For the FAST head, download the `fast` branch first (see the note below) and pass the directory.

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

## From the base, on LIBERO

`examples/libero/make_libero_base.sh` is the exact `make_finetune_base.py` invocation behind the published
LIBERO checkpoints (cameras, prompt, `eef_pose`, native deltas, replan stride 10, one Euler step for the
flow head). Run it, then `lerobot-train --policy.path=outputs/bases/libero-flow --dataset.repo_id=HuggingFaceVLA/libero ...`.

## From the base, on your own robot

The saved processors freeze the camera keys, the embodiment prompt, the control-mode string, and
the proprio switch. To change any of them, rebuild the processors first with
`scripts/make_finetune_base.py`, then train from the directory it writes:

```bash
python scripts/make_finetune_base.py \
  --base griffinlabs/Griffin-Alpha-S --revision main \
  --dataset_repo_id my-org/my-robot-dataset \
  --output_dir outputs/bases/my-robot \
  --image_keys observation.images.front observation.images.wrist \
  --embodiment_prompt "MyBot 6-DoF arm, 1 gripper" \
  --arm_control_mode joint

lerobot-train --policy.path=outputs/bases/my-robot --dataset.repo_id=my-org/my-robot-dataset \
  --policy.device=cuda --batch_size=16 --steps=20000 --output_dir=outputs/train/my-robot
```

What the script does, in order: loads the base config and applies your overrides; reads the
dataset's features and `action` names; recomputes normalization statistics in relative-action space
(see below); loads the weights; saves weights, config and both processor pipelines. Useful flags:

| flag | meaning |
|---|---|
| `--image_keys A B ...` | cameras in PROMPT ORDER: primary (exocentric) camera first, wrist cameras after, as in pre-training |
| `--embodiment_prompt`, `--arm_control_mode` | the two strings in the prompt's header. `arm_control_mode` is required (the released checkpoints use `eef_pose` and `joint`) |
| `--no-include_proprio` | drop the proprio tokens from the prompt |
| `--n_action_steps N` | replan stride at inference (the chunk is always 50 steps) |
| `--no-use_relative_actions` | for datasets whose actions are already per-step deltas |
| `--relative_exclude_joints gripper ...` | action-name fragments kept absolute |
| `--freeze_backbone` | flow head only: train the expert alone on a frozen backbone |
| `--revision fast` | start from the FAST base instead |

## Relative actions

The default action space is relative: every action dimension whose dataset name does not match one
of `relative_exclude_joints` (default `["gripper"]`) is predicted as an offset from the current
`observation.state`, and the postprocessor adds the state back. This is how the bases were
pre-trained. Two requirements follow:

1. **Your dataset's `action` feature needs `names`**, and at least one must contain a fragment from
   `relative_exclude_joints`. lerobot builds the relative mask from those names; without names every
   dimension, gripper included, becomes relative, silently. `make_finetune_base.py` refuses to run in
   that situation, and the factory logs a warning if it ever builds such a pipeline.
2. **Normalization statistics must be computed in the relative space.** A dataset's `stats.json`
   describes absolute actions. `make_finetune_base.py` recomputes relative stats by default
   (`--no-recompute_relative_stats` to skip). Equivalently, run lerobot's own tool once on the dataset:

   ```bash
   lerobot-edit-dataset --repo_id my-org/my-robot-dataset --new_repo_id my-org/my-robot-dataset \
     --operation.type recompute_stats --operation.overwrite true \
     --operation.relative_action true --operation.relative_exclude_joints "['gripper']" --operation.chunk_size 50
   ```

   (Without `--new_repo_id` equal to the source and `--operation.overwrite true`, lerobot writes the
   result to a new `<repo_id>_recomputed_stats` dataset; train on that one instead.)

   Bare `lerobot-train --policy.path=griffinlabs/Griffin-Alpha-S` normalizes with whatever `stats.json` the
   dataset carries, so do one of the two before training from a base.

For an environment whose actions are already deltas (LIBERO's OSC_POSE, many teleop datasets), set
`--no-use_relative_actions`; subtracting the state from a delta command would be wrong.

## Learning-rate schedule

The presets are a linear warmup into a cosine decay: `scheduler_warmup_steps=2000`,
`scheduler_decay_steps=30000`, from `optimizer_lr=3e-5` (and `expert_optimizer_lr=5e-5` for the flow
expert) down to `scheduler_decay_lr=1e-6`. lerobot auto-scales both step counts when `--steps` is smaller
than the decay length, so for any run under 30k steps the warmup is effectively 6.7% of the run. For a
run longer than 30k steps the learning rate reaches the floor at step 30k and stays there; pass
`--policy.scheduler_decay_steps=<your --steps>` (and scale the warmup to taste) so the decay spans the
whole run.

## The two heads

`griffin_alpha` (flow) trains one objective, the flow-matching regression, with two parameter groups:
the backbone at `optimizer_lr` (3e-5) and the expert at `expert_optimizer_lr` (5e-5). With
`freeze_backbone=true` only the expert trains. Inference integrates `num_inference_steps` Euler
steps (10 on the base, 1 on the LIBERO checkpoint); a checkpoint trained at one step count can be run
at another, and `make_finetune_base.py --num_inference_steps` bakes a different value.

`griffin_alpha_fast` trains the backbone's next-token cross-entropy over FAST action tokens
appended to the vocabulary, and decodes with greedy generation. The FAST tokens sit at the END of
the sequence, so an over-long prompt truncates the training target; the input step warns when that
happens. Raise `max_sequence_length` if you see the warning.

Both heads pad the action chunk to `max_action_dim` (32 on the bases) and mask the padding out of
the loss per sample, so the same base serves embodiments of any width up to 32.

## The subtask line

`condition_on_subtask=true` (flow head) ends the prompt with `assistant\n[subtask: <text>\n]action: `,
including a ground-truth subtask string when the batch carries one. Stock `lerobot-train` does not
forward a dataset `subtask` column to the policy, and inference never has one, so in practice the
prompt always takes the no-subtask branch (`action: ` only). That branch is a trained form. The
flag is kept because the released flow checkpoints were trained with it and the prompt must match.

## The SE(3) action path

For absolute end-effector-pose action spaces that embed 4x4 homogeneous matrices in the action and
state vectors, the plugin has an opt-in path: `se3_segment_start_idxs` marks where each flattened
matrix starts, `relative_action_mask` marks which dims are made relative to the chunk-start state
(matrix segments via `state^-1 @ action`), and the model sees each matrix as xyz + a 6-D rotation.
It requires `use_relative_actions=false`. lerobot does not re-pair these steps after loading a
checkpoint, so a harness that loads an SE(3) checkpoint must call
`lerobot_policy_griffin_alpha.reconnect_se3_steps(preprocessor, postprocessor)` once. The released
checkpoints do not use this path.
