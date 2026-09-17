#!/usr/bin/env python
"""Evaluate a Griffin Alpha-S checkpoint on LIBERO with ``lerobot-eval``, using a chunk-level rollout.

This is a thin wrapper around ``lerobot.scripts.lerobot_eval``. It changes exactly one thing: the
rollout loop. LeRobot's stock rollout re-runs the policy preprocessor on every environment step and
pops one action at a time from the policy's queue. The rollout below instead predicts a whole action
chunk once, post-processes the chunk in one go, executes ``n_action_steps`` of it, and then replans.
This is the loop the published LIBERO numbers were produced with. Every ``lerobot-eval`` flag is
forwarded unchanged.

Two knobs of our own, stripped from ``argv`` before lerobot's parser sees them (environment variables
of the same meaning are accepted as a fallback):

    --griffin.n_action_steps=N   replan stride; default = the checkpoint's ``n_action_steps``
                                 (env: GRIFFIN_N_ACTION_STEPS)
    --griffin.fm_num_steps=K     Euler steps for the flow head; default = the checkpoint's
                                 ``num_inference_steps`` (env: GRIFFIN_FM_NUM_STEPS)

Example (see run_eval.sh for the full recipe)::

    python examples/libero/eval_libero.py \\
        --policy.path=griffinlabs/Griffin-Alpha-S-LIBERO --policy.device=cuda \\
        --env.type=libero --env.task=libero_spatial \\
        --env.observation_height=256 --env.observation_width=256 --env.max_parallel_tasks=1 \\
        --eval.n_episodes=10 --eval.batch_size=1 --output_dir=outputs/eval/alpha-s-libero
"""

import os
import sys
from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from tqdm import trange

import lerobot_policy_griffin_alpha  # noqa: F401  (registers the griffin_alpha / griffin_alpha_fast policy types)
from lerobot_policy_griffin_alpha import GriffinAlphaPolicy

import lerobot.scripts.lerobot_eval as E  # noqa: E402
from lerobot.envs.utils import check_env_attributes_and_types, preprocess_observation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.utils import inside_slurm

_GRIFFIN_FLAGS = {"n_action_steps": "GRIFFIN_N_ACTION_STEPS", "fm_num_steps": "GRIFFIN_FM_NUM_STEPS"}


def _pop_griffin_flags() -> dict[str, int]:
    """Remove ``--griffin.<name>=<int>`` from argv (draccus rejects unknown flags) and return them."""
    values: dict[str, int] = {}
    kept = []
    for arg in sys.argv:
        if arg.startswith("--griffin."):
            name, _, raw = arg[len("--griffin.") :].partition("=")
            if name not in _GRIFFIN_FLAGS or not raw:
                raise SystemExit(f"unknown or valueless flag {arg!r}; known: --griffin.{{{', '.join(_GRIFFIN_FLAGS)}}}=<int>")
            values[name] = int(raw)
        else:
            kept.append(arg)
    sys.argv[:] = kept
    for name, env in _GRIFFIN_FLAGS.items():
        if name not in values and os.environ.get(env):
            values[name] = int(os.environ[env])
    return values


GRIFFIN = _pop_griffin_flags()


def _chunk_kwargs(policy) -> dict:
    """``num_steps`` for the flow head when overridden; nothing otherwise (the FAST policy takes no such argument)."""
    if isinstance(policy, GriffinAlphaPolicy) and GRIFFIN.get("fm_num_steps"):
        return {"num_steps": GRIFFIN["fm_num_steps"]}
    return {}


def chunk_rollout(
    env,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    seeds=None,
    return_observations=False,
    render_callback=None,
    **_ignored,
):
    """Drop-in replacement for ``lerobot_eval.rollout``: predict a chunk, execute ``n_action_steps``, replan.

    Returns the same dict shape as the stock rollout. Extra keyword arguments newer lerobot versions pass
    (recording options) are accepted and ignored.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations, all_actions, all_rewards, all_successes, all_dones = [], [], [], [], []
    step = 0
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    n_action_steps = GRIFFIN.get("n_action_steps") or int(getattr(policy.config, "n_action_steps", 1))
    progbar = trange(
        max_steps,
        desc=f"Chunk rollout ({n_action_steps}/replan), at most {max_steps} steps",
        disable=inside_slurm(),
        leave=False,
    )
    check_env_attributes_and_types(env)
    action_queue: deque = deque()  # env-ready actions, numpy (B, action_dim)

    while not np.all(done) and step < max_steps:
        proc_obs = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(proc_obs))

        if len(action_queue) == 0:
            try:
                proc_obs["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    proc_obs["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    proc_obs["task"] = [""] * env.num_envs

            # Preprocess ONCE per chunk (this also caches the reference state for relative actions).
            obs_for_policy = preprocessor(env_preprocessor(proc_obs))
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(obs_for_policy, **_chunk_kwargs(policy))  # (B, horizon, dim)
            chunk = chunk[:, :n_action_steps]
            # Postprocess the WHOLE chunk against the single cached chunk-start state.
            env_chunk = env_postprocessor({ACTION: postprocessor(chunk)})[ACTION]  # (B, chunk, env_dim)
            for k in range(env_chunk.shape[1]):
                action_queue.append(env_chunk[:, k, :].to("cpu").numpy())

        action_numpy = action_queue.popleft()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError("Unsupported `final_info` format; upgrade gymnasium to >= 1.0.")
            successes = final_info["is_success"].tolist()
        elif "is_success" in info:
            is_success = info["is_success"]
            successes = is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * env.num_envs
        else:
            successes = [False] * env.num_envs

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_success_rate = torch.stack(all_successes, dim=1).any(dim=1).float().mean().item()
        progbar.set_postfix({"running_success_rate": f"{running_success_rate * 100:.1f}%"})
        progbar.update()

    if return_observations:
        all_observations.append(deepcopy(preprocess_observation(observation)))

    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        ret[OBS_STR] = {key: torch.stack([obs[key] for obs in all_observations], dim=1) for key in all_observations[0]}

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()
    return ret


E.rollout = chunk_rollout


if __name__ == "__main__":
    E.main()
