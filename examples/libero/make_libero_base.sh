#!/usr/bin/env bash
# Rebuild a Griffin Alpha-S base for LIBERO fine-tuning: the exact make_finetune_base.py invocation behind
# the published griffinlabs/Griffin-Alpha-S-LIBERO checkpoints (cameras, prompt, control mode, native
# per-step deltas, replan stride, 1 Euler step for the flow head).
#
#   bash examples/libero/make_libero_base.sh                       # flow head  -> outputs/bases/libero-flow
#   REVISION=fast bash examples/libero/make_libero_base.sh         # FAST head  -> outputs/bases/libero-fast
#   lerobot-train --policy.path=outputs/bases/libero-flow --dataset.repo_id=HuggingFaceVLA/libero ...
set -euo pipefail
BASE="${BASE:-griffinlabs/Griffin-Alpha-S}"
REVISION="${REVISION:-main}"
OUT="${OUT:-outputs/bases/libero-$([[ "$REVISION" == "fast" ]] && echo fast || echo flow)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA=()
[[ "$REVISION" == "fast" ]] || EXTRA+=(--num_inference_steps 1)   # flow head only; see docs/libero_eval.md
python "${SCRIPT_DIR}/../../scripts/make_finetune_base.py" \
  --base "${BASE}" --revision "${REVISION}" \
  --dataset_repo_id HuggingFaceVLA/libero \
  --output_dir "${OUT}" \
  --image_keys observation.images.image observation.images.image2 \
  --embodiment_prompt "LIBERO simulated Franka Emika Panda, 1 gripper" \
  --arm_control_mode eef_pose \
  --no-use_relative_actions \
  --n_action_steps 10 \
  "${EXTRA[@]}" "$@"
echo ">>> LIBERO base written to ${OUT}; next: lerobot-train --policy.path=${OUT} --dataset.repo_id=HuggingFaceVLA/libero --policy.device=cuda ..."
