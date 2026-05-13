"""Batch-level augmentation strategies for Dogs vs Cats experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

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
class SameLabelCutMix:
    """CutMix that only mixes examples from the same class, preserving hard labels."""

    alpha: float = 1.0
    probability: float = 1.0
    name: str = "same_label_cutmix"

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[0] < 2 or torch.rand((), device=images.device).item() > self.probability:
            return images, targets

        mixed = images.clone()
        batch_size, _, height, width = images.shape
        lam = self._sample_lambda(images.device)
        cut_ratio = (1.0 - lam) ** 0.5
        cut_w = max(1, int(width * cut_ratio))
        cut_h = max(1, int(height * cut_ratio))

        for index in range(batch_size):
            same_label = torch.nonzero(targets == targets[index], as_tuple=False).flatten()
            same_label = same_label[same_label != index]
            if same_label.numel() == 0:
                continue
            partner = same_label[torch.randint(same_label.numel(), (1,), device=images.device)].item()
            center_x = torch.randint(width, (1,), device=images.device).item()
            center_y = torch.randint(height, (1,), device=images.device).item()
            x1 = max(0, center_x - cut_w // 2)
            x2 = min(width, x1 + cut_w)
            y1 = max(0, center_y - cut_h // 2)
            y2 = min(height, y1 + cut_h)
            mixed[index, :, y1:y2, x1:x2] = images[int(partner), :, y1:y2, x1:x2]
        return mixed, targets

    def _sample_lambda(self, device: torch.device) -> float:
        if self.alpha <= 0:
            return 0.5
        concentration = torch.tensor([self.alpha], device=device)
        beta = torch.distributions.Beta(concentration, concentration)
        return float(beta.sample().item())


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


@dataclass(frozen=True)
class CompositeBatchAugmentation:
    """Apply multiple batch augmentation strategies in order."""

    augmentations: Sequence[BatchAugmentation]
    name: str = "combined_augmentations"

    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for augmentation in self.augmentations:
            images, targets = augmentation(images, targets)
        return images, targets
