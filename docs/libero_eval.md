# LIBERO evaluation

`examples/libero/eval_libero.py` wraps `lerobot-eval` with one change: a chunk-level rollout that
predicts a whole action chunk, executes `n_action_steps` of it, and replans. It is the loop the
numbers below were produced with. Every `lerobot-eval` flag is forwarded unchanged.

## Install

```bash
pip install 'lerobot_policy_griffin_alpha[libero]'
```

Two things LIBERO's dependency chain needs that pip does not handle by itself:

1. **`egl_probe` / `hf-egl-probe` build.** Their `setup.py` shells out to a `cmake` binary that pip's
   isolated build environment cannot see, and their CMake files predate CMake 4. Install them first,
   without build isolation:

   ```bash
   CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation egl_probe hf-egl-probe
   ```

2. **LIBERO's first-import prompt.** `import libero.libero` asks an interactive question about its
   config directory on first use. Answer it once, non-interactively:

   ```bash
   printf 'N\n' | python -c "import libero.libero"
   ```

Rendering: the launcher defaults to `MUJOCO_GL=egl`. If your machine has no GLVND `libEGL.so.0`, or
MuJoCo's EGL backend crashes under load, use software rendering instead:
`MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (needs `libosmesa6` on Debian/Ubuntu). It is slower but
deterministic.

LIBERO downloads its task assets on first use. Its downloader treats the cache as complete as soon
as the four top-level asset directories exist, so an interrupted download can leave a partial cache
that then fails inside the simulator with a missing-file error. If that happens, delete the LIBERO
asset directory (see `~/.libero/config.yaml`) and let it download again.

## Run

```bash
bash examples/libero/run_eval.sh                       # flow head, all four suites, 10 episodes/task
REVISION=fast bash examples/libero/run_eval.sh         # FAST head (the launcher downloads the branch locally first)
SUITE=libero_spatial EPISODE_LENGTH=60 N_EPISODES=1 bash examples/libero/run_eval.sh   # smoke test
```

The launcher fixes what the published numbers depend on: 256x256 observations,
`--env.max_parallel_tasks=1`, and the checkpoint's own replan stride (`n_action_steps=10`). Two knobs
of the script's own: `--griffin.n_action_steps=N` (replan stride) and `--griffin.fm_num_steps=K`
(Euler steps of the flow head, default = the checkpoint's `num_inference_steps`). Metrics land in
`<OUTPUT_DIR>/eval_info.json`, videos under `<OUTPUT_DIR>/videos/`.

Loading the model the first time takes a few minutes (4.4B backbone + 0.9B expert). FlashAttention 2
is used when installed and falls back to SDPA with a warning otherwise.

## Results

Four suites x 10 tasks x 10 episodes = 400 episodes per configuration, single seed, 256x256 RGB
from the agent-view and wrist cameras, native per-step OSC deltas (`use_relative_actions=false`).

| suite | `griffin_alpha` (flow, 1 Euler step, the checkpoint default) | `griffin_alpha` (flow, 10 Euler steps) | `griffin_alpha_fast` |
|---|---|---|---|
| libero_spatial | 98.0 | 92.0 | 96.0 |
| libero_object | 100.0 | 100.0 | 96.0 |
| libero_goal | 95.0 | 97.0 | 97.0 |
| libero_10 | 94.0 | 95.0 | 87.0 |
| **four-suite** | **96.8** (387/400) | **96.0** (384/400) | **94.0** (376/400) |

The LIBERO flow checkpoint ships with `num_inference_steps=1`: on this benchmark one Euler step matched
ten within noise and costs about a tenth of the inference time. Pass `--griffin.fm_num_steps=10` to
reproduce the 10-step column. The general-purpose base keeps 10.

One caveat: **the two heads were not trained identically.** The flow checkpoint has proprio tokens in
its prompt (`include_proprio=true`); the FAST checkpoint does not. The gap between the columns mixes the
head change with the prompt change.
