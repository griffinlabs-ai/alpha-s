import torch
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from lerobot_policy_griffin_alpha.backbone import GriffinAlphaBackboneConfig


def make_sample_env_transition(
    # Only reads image_keys / horizon / max_action_dim, which both heads share.
    config: GriffinAlphaBackboneConfig,
    *,
    batched: bool = True,
    with_action: bool = False,
) -> EnvTransition:
    state = torch.randn(7) if not batched else torch.randn(1, 7)

    observation = {OBS_STATE: state}
    for image_key in config.image_keys:
        if batched:
            observation[image_key] = [torch.randint(0, 255, (3, 64, 64))]
        else:
            observation[image_key] = torch.randint(0, 255, (3, 64, 64))

    info = {
        "arm_control_mode": ["joint"] if batched else "joint",
        "embodiment_prompt": ["test_robot"] if batched else "test_robot",
        "n_action_dims": [7] if batched else 7,
    }

    transition: EnvTransition = {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: None,
        TransitionKey.REWARD: None,
        TransitionKey.DONE: None,
        TransitionKey.TRUNCATED: None,
        TransitionKey.INFO: info,
        TransitionKey.COMPLEMENTARY_DATA: {
            "task": ["pick up the cup"] if batched else "pick up the cup",
            "subtask": [""] if batched else "",
        },
    }

    if with_action:
        horizon = config.horizon
        action_dim = config.max_action_dim
        if batched:
            transition[TransitionKey.ACTION] = torch.randn(1, horizon, action_dim)
        else:
            transition[TransitionKey.ACTION] = torch.randn(horizon, action_dim)

    return transition
