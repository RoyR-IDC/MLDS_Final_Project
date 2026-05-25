"""Batch-level augmentation strategies for Dogs vs Cats experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.preprocessing.tile_transforms import split_into_tiles, reconstruct_from_tiles


class BatchAugmentation(Protocol):
    """Callable strategy that can modify a training batch."""

    name: str

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return augmented images and targets."""


@dataclass(frozen=True)
class NoBatchAugmentation:
    """No-op batch augmentation."""

    name: str = "none"

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return images, targets


@dataclass(frozen=True)
class RandomPatchShuffle:
    """Randomly shuffle square patches independently for each image."""

    tiles_per_side: int = 3
    probability: float = 1.0
    name: str = "patch_shuffle"

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")
        if self.tiles_per_side < 2:
            return images, targets

        shuffled = images.clone()
        for index in range(images.shape[0]):
            if torch.rand((), device=images.device).item() > self.probability:
                continue
            tiles = split_into_tiles(images[index], self.tiles_per_side)
            order = torch.randperm(tiles.shape[0], device=images.device)
            shuffled[index] = reconstruct_from_tiles(tiles[order], self.tiles_per_side)
        return shuffled, targets
