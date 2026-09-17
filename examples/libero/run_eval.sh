#!/usr/bin/env bash
# Evaluate a Griffin Alpha-S checkpoint on LIBERO (closed loop, 256x256 observations).
#
# Prerequisites (see docs/libero_eval.md):
#   pip install 'lerobot_policy_griffin_alpha[libero]'      (plus the egl_probe build note)
#
# Usage:
#   bash examples/libero/run_eval.sh [extra lerobot-eval / eval_libero.py flags...]
#
# Environment overrides (defaults in parentheses):
#   POLICY_PATH     Hub id or local dir            (griffinlabs/Griffin-Alpha-S-LIBERO)
#   REVISION        Hub branch: main = flow head, fast = FAST head   (main); a non-main branch is
#                   downloaded to LOCAL_CKPT_DIR (~/.cache/griffin-alpha-s) first, see the note below
#   SUITE           LIBERO suite or comma list     (libero_spatial,libero_object,libero_goal,libero_10)
#   TASK_IDS        optional JSON list, e.g. "[0,1]" (unset = every task in the suite)
#   N_EPISODES      episodes per task              (10)
#   BATCH_SIZE      parallel envs per task         (1)
#   EPISODE_LENGTH  optional max steps per episode (unset = LIBERO default; 60 makes a quick smoke test)
#   OUTPUT_DIR      metrics + videos               (outputs/eval/alpha-s-libero)
#   MUJOCO_GL       egl (default) or osmesa        -- see docs/libero_eval.md if EGL is unavailable
#
# The replan stride is read from the checkpoint (n_action_steps=10). Override with
# --griffin.n_action_steps=N; override the flow head's Euler steps with --griffin.fm_num_steps=K.
set -euo pipefail

POLICY_PATH="${POLICY_PATH:-griffinlabs/Griffin-Alpha-S-LIBERO}"
REVISION="${REVISION:-main}"
SUITE="${SUITE:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${N_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval/alpha-s-libero}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"

EXTRA=()
if [[ -n "${TASK_IDS:-}" ]]; then EXTRA+=(--env.task_ids="${TASK_IDS}"); fi
if [[ -n "${EPISODE_LENGTH:-}" ]]; then EXTRA+=(--env.episode_length="${EPISODE_LENGTH}"); fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# lerobot's CLIs read config.json from --policy.path WITHOUT applying --policy.pretrained_revision (only the
# weights and processors follow the revision), and the two branches hold different policy types. So a
# non-main branch of a Hub repo is downloaded to a local directory first and evaluated from there.
if [[ "${REVISION}" != "main" && ! -d "${POLICY_PATH}" ]]; then
  LOCAL_DIR="${LOCAL_CKPT_DIR:-${HOME}/.cache/griffin-alpha-s}/$(echo "${POLICY_PATH}" | tr '/' '_')@${REVISION}"
  echo ">>> downloading ${POLICY_PATH}@${REVISION} to ${LOCAL_DIR}"
  hf download "${POLICY_PATH}" --revision "${REVISION}" --local-dir "${LOCAL_DIR}" >/dev/null
  POLICY_PATH="${LOCAL_DIR}"
fi

echo ">>> Evaluating ${POLICY_PATH} (${REVISION}) on LIBERO suite=${SUITE} (${N_EPISODES} eps/task, batch=${BATCH_SIZE})"
python "${SCRIPT_DIR}/eval_libero.py" \
  --policy.path="${POLICY_PATH}" \
  --policy.device=cuda \
  --env.type=libero \
  --env.task="${SUITE}" \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --env.max_parallel_tasks=1 \
  --eval.n_episodes="${N_EPISODES}" \
  --eval.batch_size="${BATCH_SIZE}" \
  --output_dir="${OUTPUT_DIR}" \
  "${EXTRA[@]}" \
  "$@"

echo ">>> Done. Metrics: ${OUTPUT_DIR}/eval_info.json ; videos: ${OUTPUT_DIR}/videos/"
