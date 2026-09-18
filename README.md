# Griffin Alpha-S

Griffin Alpha-S is a vision-language-action (VLA) model by [Griffin Labs](https://griffinlabs.ai):
a Qwen3-VL-4B backbone extended with a robot-action head and pre-trained on a multi-embodiment
mixture of robot data. This repository is the [LeRobot](https://github.com/huggingface/lerobot)
policy plugin that runs, evaluates and fine-tunes the released weights, plus the LIBERO evaluation
recipe.

| 📝 Blog | 🤗 Hugging Face |
|:---:|:---:|
| [Read the Blog](https://griffinlabs.ai/blog/griffin-alpha-s) | [Access Models Weights](https://huggingface.co/collections/griffinlabs/alpha-s) |

Two heads ship, as two LeRobot policy types selected by a checkpoint's `config.json`:

| policy type | head | class |
|---|---|---|
| `griffin_alpha` | flow-matching action expert (910M) over the backbone; the default | `GriffinAlphaPolicy` |
| `griffin_alpha_fast` | FAST action tokens appended to the vocabulary, autoregressive | `GriffinAlphaFASTPolicy` |

Both predict a 50-step action chunk from up to three camera images, the proprioceptive state, and a
language instruction, and both share the same prompt, image pipeline and action-space transform.

## Checkpoints

| repository | `main` branch | `fast` branch |
|---|---|---|
| [`griffinlabs/Griffin-Alpha-S`](https://huggingface.co/griffinlabs/Griffin-Alpha-S) | pre-trained base, flow head | pre-trained base, FAST head |
| [`griffinlabs/Griffin-Alpha-S-LIBERO`](https://huggingface.co/griffinlabs/Griffin-Alpha-S-LIBERO) | LIBERO fine-tune, flow head | LIBERO fine-tune, FAST head |

Weights are released under **CC BY-NC-SA 4.0**. The code in this repository is **Apache-2.0**.
Details of what each checkpoint contains: [docs/checkpoints.md](docs/checkpoints.md).

## Install

Python 3.12 or newer.

```bash
pip install git+https://github.com/griffinlabs-ai/alpha-s.git            # the plugin
pip install 'lerobot_policy_griffin_alpha[libero] @ git+https://github.com/griffinlabs-ai/alpha-s.git'   # + LIBERO sim
pip install 'lerobot_policy_griffin_alpha[train] @ git+https://github.com/griffinlabs-ai/alpha-s.git'    # + lerobot-train deps (accelerate)
```

LeRobot discovers the plugin automatically (any installed distribution named `lerobot_policy_*`), so
`lerobot-train`, `lerobot-eval` and the async inference server accept `griffin_alpha` and
`griffin_alpha_fast` with no extra flags. FlashAttention 2 is used when installed
(`pip install flash-attn`, see the `flash` extra in `pyproject.toml`) and falls back to SDPA with a
warning otherwise. The LIBERO extra has two install gotchas covered in
[docs/libero_eval.md](docs/libero_eval.md).

## Quickstart

**Evaluate on LIBERO** (closed loop, 4 suites x 10 tasks x 10 episodes):

```bash
bash examples/libero/run_eval.sh                 # flow head
REVISION=fast bash examples/libero/run_eval.sh   # FAST head
```

**Load a policy in Python:**

```python
import lerobot_policy_griffin_alpha  # registers the policy types with lerobot
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_griffin_alpha import GriffinAlphaPolicy

policy = GriffinAlphaPolicy.from_pretrained("griffinlabs/Griffin-Alpha-S-LIBERO")  # GriffinAlphaFASTPolicy + revision="fast" for the FAST head
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path="griffinlabs/Griffin-Alpha-S-LIBERO",
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
)
batch = preprocessor({
    "observation.images.image": image,     # (3, 256, 256) float in [0, 1]
    "observation.images.image2": wrist,    # (3, 256, 256)
    "observation.state": state,            # (8,)
    "task": "put the bowl on the plate",
})
chunk = postprocessor(policy.predict_action_chunk(batch))  # (1, 50, 7) actions for the environment
```

**Fine-tune with `lerobot-train`:**

```bash
lerobot-train --policy.path=griffinlabs/Griffin-Alpha-S-LIBERO --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.device=cuda --batch_size=16 --steps=6000 --output_dir=outputs/train/libero-ft
```

To fine-tune the base on your own robot (different cameras, prompt, or action space), first rebuild
the checkpoint's processors for your dataset with `scripts/make_finetune_base.py`, then train from
the directory it writes. For the FAST head, download the `fast` branch to a directory first (`hf download
<repo> --revision fast --local-dir ...`): the lerobot CLIs do not apply a revision to `config.json`. The full
recipe, including how relative actions and their normalization statistics work, is in
[docs/finetuning.md](docs/finetuning.md).

## Results

LIBERO, 400 episodes per configuration (4 suites x 10 tasks x 10 episodes), single seed:

| suite | flow head, 1 Euler step (checkpoint default) | flow head, 10 Euler steps | FAST head |
|---|---|---|---|
| libero_spatial | 98.0 | 92.0 | 96.0 |
| libero_object | 100.0 | 100.0 | 96.0 |
| libero_goal | 95.0 | 97.0 | 97.0 |
| libero_10 | 94.0 | 95.0 | 87.0 |
| **four-suite** | **96.8** | **96.0** | **94.0** |

Ten episodes per task is a standard error of about 1.2 points on the four-suite number, and the flow
and FAST checkpoints differ in more than the head (the flow one has proprio in its prompt), so read
the columns as three good checkpoints rather than a ranking. See
[docs/libero_eval.md](docs/libero_eval.md) for the protocol and caveats.

## Repository layout

```
src/lerobot_policy_griffin_alpha/
  backbone.py                       shared Qwen3-VL construction, loading, saving, freeze flags; the shared config fields
  configuration_griffin_alpha.py    griffin_alpha (flow) config      modeling_griffin_alpha.py       flow expert + policy
  configuration_griffin_alpha_fast.py  griffin_alpha_fast config     modeling_griffin_alpha_fast.py  FAST policy
  processor_griffin_alpha[_fast].py the two input steps + pre/post-processor factories
  input_step.py                     the shared prompt builder        pipeline_common.py   the shared pipeline layout
  action_steps.py                   action-space processor steps     fast_decoding.py     FAST inverse DCT decode
examples/libero/                    eval_libero.py (chunk-level lerobot-eval rollout), run_eval.sh, make_libero_base.sh
scripts/make_finetune_base.py       rebuild a checkpoint's processors for your dataset
docs/                               finetuning.md, libero_eval.md, checkpoints.md
tests/                              pytest suite (CPU; downloads the public Qwen3-VL processor on first run)
```

## Tests

```bash
pip install -e '.[test]'
pytest
```

## Citation

```bibtex
@misc{griffinalpha2026,
  title  = {Griffin Alpha-S: an open-weights vision-language-action model},
  author = {Griffin Labs},
  year   = {2026},
  url    = {https://github.com/griffinlabs-ai/alpha-s}
}
```

## License

Code: Apache License 2.0 (see `LICENSE`, `NOTICE`). Model weights: CC BY-NC-SA 4.0, see the LICENSE
file in each Hugging Face repository.
