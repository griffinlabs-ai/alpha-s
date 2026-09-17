"""Prompt-side helpers shared by both input processor steps: action/proprio token strings and the
training-time image augmentation."""

from functools import cache
from typing import TypeVar

import torch
import torchvision.transforms as T_v2

T = TypeVar("T")


def index_optional_list(values: list[T] | None, index: int) -> T | None:
    """``values[index]`` when a per-sample list is present, else ``None``."""
    return values[index] if values is not None else None


def map_fast_token_to_vlm_action(tokens: list[str]) -> str:
    return "".join(f"<robot_action_{token}>" for token in tokens)


def make_proprio_state_tokens(vocab_size: int) -> list[str]:
    return [f"<proprio_state_{i}>" for i in range(vocab_size)]


def map_normalized_state_to_vlm_proprio(state: torch.Tensor, vocab_size: int) -> str:
    if vocab_size <= 0:
        raise ValueError(f"proprio_vocab_size must be positive, got {vocab_size}")

    clipped = state.clamp(-1.0, 1.0)
    bucket_ids = torch.floor((clipped + 1.0) * (vocab_size / 2.0)).to(torch.long)
    bucket_ids = bucket_ids.clamp(0, vocab_size - 1)
    return "".join(f"<proprio_state_{bucket_id}>" for bucket_id in bucket_ids.reshape(-1).tolist())


class GriffinAlphaImageTransform:
    BRIGHTNESS_FACTOR = 0.2
    CONTRAST_FACTOR = 0.2
    SATURATION_FACTOR = 0.2
    HUE_FACTOR = 0.05
    CROP_SCALE = 0.9

    def __init__(self) -> None:
        self.color_jitter = T_v2.ColorJitter(
            brightness=self.BRIGHTNESS_FACTOR,
            contrast=self.CONTRAST_FACTOR,
            saturation=self.SATURATION_FACTOR,
            hue=self.HUE_FACTOR,
        )

    @staticmethod
    @cache
    def get_random_crop_transform(source_size: tuple[int, int], scale: float) -> T_v2.RandomCrop:
        linear_scale = scale**0.5
        target_size = round(source_size[0] * linear_scale), round(source_size[1] * linear_scale)
        return T_v2.RandomCrop(target_size)

    @staticmethod
    @cache
    def get_center_crop_transform(source_size: tuple[int, int], scale: float) -> T_v2.CenterCrop:
        linear_scale = scale**0.5
        target_size = round(source_size[0] * linear_scale), round(source_size[1] * linear_scale)
        return T_v2.CenterCrop(target_size)

    def center_crop(self, image: torch.Tensor) -> torch.Tensor:
        return self.get_center_crop_transform(image.shape[-2:], self.CROP_SCALE)(image)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        image = self.get_random_crop_transform(image.shape[-2:], self.CROP_SCALE)(image)
        return self.color_jitter(image)
