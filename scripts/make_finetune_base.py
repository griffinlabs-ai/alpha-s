#!/usr/bin/env python
"""Re-bake a published Griffin Alpha-S checkpoint for YOUR dataset, so ``lerobot-train`` can fine-tune it.

Why this exists: ``lerobot-train --policy.path=<checkpoint>`` reloads the checkpoint's saved
pre/post-processors and only overrides their normalization statistics from your dataset. The camera
keys, the embodiment prompt, the control-mode string, and whether proprio tokens are in the prompt are
frozen inside that saved processor JSON. This script rebuilds the processors from a config you control
and saves a new policy directory (weights + config + processors) that ``lerobot-train`` can start from::

    python scripts/make_finetune_base.py \
        --base griffinlabs/Griffin-Alpha-S --revision main \
        --dataset_repo_id my-org/my-robot-dataset \
        --output_dir outputs/bases/my-robot \
        --image_keys observation.images.front observation.images.wrist \
        --embodiment_prompt "MyBot 6-DoF arm, 1 gripper" --arm_control_mode joint

    lerobot-train --policy.path=outputs/bases/my-robot --dataset.repo_id=my-org/my-robot-dataset ...

Relative actions are ON by default (the bases were pre-trained that way: arm dimensions relative to
the current state, grippers absolute). That needs two things from your dataset, both checked here:
``action`` feature ``names`` (so the gripper can be told apart) and normalization statistics computed
in the RELATIVE space, which this script recomputes by default (``--no-recompute_relative_stats`` to
skip). Pass ``--no-use_relative_actions`` for a dataset whose actions are already per-step deltas.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

import lerobot_policy_griffin_alpha  # noqa: F401  (registers the policy types)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.feature_utils import dataset_to_policy_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("make_finetune_base")


def _bool_flag(parser: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    parser.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help + f" (default: {default})")
    parser.add_argument(f"--no-{name}", dest=name, action="store_false")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="Hub repo id or local directory of a Griffin Alpha-S checkpoint")
    p.add_argument("--revision", default=None, help="Hub branch/tag (e.g. `fast` for the FAST head)")
    p.add_argument("--dataset_repo_id", required=True, help="LeRobot dataset to fine-tune on (features + stats come from it)")
    p.add_argument("--dataset_root", default=None, help="Local root of the dataset if not in the HF cache")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_keys", nargs="*", default=None,
                   help="Camera keys IN PROMPT ORDER (primary camera first). Default: every VISUAL feature of the dataset in its order")
    p.add_argument("--embodiment_prompt", default=None, help='Free text describing the robot, e.g. "MyBot 6-DoF arm, 1 gripper"')
    p.add_argument("--arm_control_mode", default=None, help='Control-mode string in the prompt, e.g. "joint" or "eef_pose" (required unless the base has one)')
    _bool_flag(p, "include_proprio", True, "Put discretized proprio-state tokens in the prompt")
    p.add_argument("--n_action_steps", type=int, default=None, help="Actions executed per predicted chunk (replan stride)")
    _bool_flag(p, "use_relative_actions", True, "Predict arm dims relative to the current state (grippers absolute)")
    p.add_argument("--relative_exclude_joints", nargs="*", default=None, help='Action names kept absolute (default: ["gripper"])')
    _bool_flag(p, "recompute_relative_stats", True, "Recompute normalization stats in relative space (needs the full dataset)")
    _bool_flag(p, "freeze_backbone", False, "Flow head only: train the expert alone on a frozen backbone")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="Flow head only: Euler steps at inference (the base ships 10; 1 matched 10 on LIBERO but not on a bimanual robot)")
    p.add_argument("--device", default="cuda", help="Device recorded in the saved config (the weights are handled on CPU here)")
    p.add_argument("--num_workers", type=int, default=4, help="Threads for the relative-stats pass")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Config from the base, then the overrides.
    cfg = PreTrainedConfig.from_pretrained(args.base, revision=args.revision)
    log.info("loaded %s config (type=%s)", type(cfg).__name__, type(cfg).get_choice_name(type(cfg)))
    if args.embodiment_prompt is not None:
        cfg.embodiment_prompt = args.embodiment_prompt
    if args.arm_control_mode is not None:
        cfg.arm_control_mode = args.arm_control_mode
    if cfg.arm_control_mode is None:
        raise SystemExit("--arm_control_mode is required: the prompt names the control mode and the plugin refuses to guess it")
    cfg.include_proprio = args.include_proprio
    cfg.use_relative_actions = args.use_relative_actions
    if args.relative_exclude_joints is not None:
        cfg.relative_exclude_joints = list(args.relative_exclude_joints)
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    if hasattr(cfg, "freeze_backbone"):
        cfg.freeze_backbone = args.freeze_backbone
    if args.num_inference_steps is not None and hasattr(cfg, "num_inference_steps"):
        cfg.num_inference_steps = args.num_inference_steps
    cfg.device = args.device
    cfg.pretrained_path = None
    cfg.push_to_hub = False
    cfg.repo_id = None

    # 2. Features and action names from the dataset.
    meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(meta.features)
    cfg.output_features = {k: ft for k, ft in features.items() if ft.type.name == "ACTION"}
    cfg.input_features = {k: ft for k, ft in features.items() if k not in cfg.output_features}
    action_names = (meta.features.get(ACTION) or {}).get("names")
    cfg.action_feature_names = list(action_names) if action_names else None
    if args.image_keys is not None:
        cfg.image_keys = tuple(args.image_keys)
        missing = [k for k in cfg.image_keys if k not in cfg.input_features]
        if missing:
            raise SystemExit(f"--image_keys not in the dataset's features: {missing}; available: {list(cfg.input_features)}")
    else:
        cfg.image_keys = ()
    log.info("cameras (prompt order): %s", cfg.resolved_image_keys)

    if cfg.use_relative_actions:
        if not cfg.action_feature_names:
            raise SystemExit(
                "use_relative_actions=True but the dataset's `action` feature has no `names`. Without names every "
                "action dimension, gripper included, would be made relative. Add names to the dataset "
                "(see lerobot's dataset docs) or pass --no-use_relative_actions."
            )
        excluded = [n for n in cfg.action_feature_names if any(tok.lower() in str(n).lower() for tok in cfg.relative_exclude_joints)]
        if not excluded:
            raise SystemExit(
                f"none of the action names {cfg.action_feature_names} match relative_exclude_joints="
                f"{cfg.relative_exclude_joints}; every dimension would become relative. Pass --relative_exclude_joints "
                "<name fragments> for the dimensions that must stay absolute (grippers), or --no-use_relative_actions."
            )
        log.info("relative actions ON; kept absolute: %s", excluded)

    # 3. Normalization statistics.
    if cfg.use_relative_actions and args.recompute_relative_stats:
        from lerobot.datasets.dataset_tools import recompute_stats

        log.info("recomputing stats in relative space (chunk=%d); this reads the whole dataset", cfg.horizon)
        dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
        dataset = recompute_stats(
            dataset, relative_action=True, relative_exclude_joints=cfg.relative_exclude_joints,
            chunk_size=cfg.horizon, num_workers=args.num_workers,
        )
        stats = dataset.meta.stats
    else:
        stats = meta.stats
        if cfg.use_relative_actions:
            log.warning("using the dataset's RAW stats with relative actions; pass --recompute_relative_stats unless stats.json is already relative")

    # 4. Weights + processors, saved as one lerobot policy directory.
    policy_cls = get_policy_class(type(cfg).get_choice_name(type(cfg)))
    log.info("loading weights from %s (this takes a while on CPU)", args.base)
    policy = policy_cls.from_pretrained(args.base, config=cfg, revision=args.revision)
    policy.save_pretrained(out)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(out)
    post.save_pretrained(out)

    summary = {
        "type": type(cfg).get_choice_name(type(cfg)),
        "base": args.base, "revision": args.revision, "dataset": args.dataset_repo_id,
        "image_keys": list(cfg.resolved_image_keys), "use_relative_actions": cfg.use_relative_actions,
        "relative_exclude_joints": cfg.relative_exclude_joints, "action_feature_names": cfg.action_feature_names,
        "relative_stats_recomputed": bool(cfg.use_relative_actions and args.recompute_relative_stats),
    }
    (out / "finetune_base.json").write_text(json.dumps(summary, indent=2))
    log.info("done. Next:\n  lerobot-train --policy.path=%s --dataset.repo_id=%s --policy.device=cuda ...", out, args.dataset_repo_id)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
