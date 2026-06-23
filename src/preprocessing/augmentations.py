"""Batch-level augmentation strategies for Dogs vs Cats experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from src.preprocessing.image_transforms import make_tile_compatible_image_size
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

    def _tile_compatible_image(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, height, width = image.shape
        if height != width:
            raise ValueError(f"Expected square image tensor with shape [C, N, N], got H={height}, W={width}")
        compatible_size = make_tile_compatible_image_size(height, self.tiles_per_side)
        if compatible_size == height:
            return image, (height, width)

        total_padding = compatible_size - height
        before = total_padding // 2
        after = total_padding - before
        padded = F.pad(
            image.unsqueeze(0),
            (before, after, before, after),
            mode="replicate",
        ).squeeze(0)
        return padded, (height, width)

    @staticmethod
    def _center_crop(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        target_height, target_width = size
        _, height, width = image.shape
        top = max(0, (height - target_height) // 2)
        left = max(0, (width - target_width) // 2)
        return image[:, top : top + target_height, left : left + target_width]

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")
        if self.tiles_per_side < 2:
            return images, targets

        shuffled = images.clone()
        for index in range(images.shape[0]):
            if torch.rand((), device=images.device).item() > self.probability:
                continue
            compatible_image, original_size = self._tile_compatible_image(images[index])
            tiles = split_into_tiles(compatible_image, self.tiles_per_side)
            order = torch.randperm(tiles.shape[0], device=images.device)
            reconstructed = reconstruct_from_tiles(tiles[order], self.tiles_per_side)
            shuffled[index] = self._center_crop(reconstructed, original_size)
        return shuffled, targets
