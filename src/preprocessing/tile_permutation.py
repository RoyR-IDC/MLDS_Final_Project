"""Tensor-based tile permutation utilities for image classification."""

from __future__ import annotations

import random
from typing import Callable, Optional, Protocol, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.preprocessing.permutations import (
    identity_permutation,
    random_permutation,
)

ImageLabel = tuple[torch.Tensor, int]


class TensorClassificationDataset(Protocol):
    """Dataset protocol for tensor image classification samples."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, idx: int) -> ImageLabel:
        """Return one ``(image, label)`` sample."""
        ...


def validate_grid_size(image: torch.Tensor, grid_size: int) -> None:
    """Validate that an image tensor can be split into a square tile grid.

    Args:
        image: Image tensor with shape ``[C, H, W]``.
        grid_size: Number of tiles along each image side.

    Raises:
        ValueError: If the tensor shape or grid size is invalid.
    """

    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C, H, W], got {tuple(image.shape)}")
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    _, height, width = image.shape
    if height % grid_size != 0 or width % grid_size != 0:
        raise ValueError(
            f"Image height and width must be divisible by grid_size={grid_size}; "
            f"got H={height}, W={width}."
        )


def split_into_tiles(image: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Split an image tensor into row-major tiles.

    Args:
        image: Tensor with shape ``[C, H, W]``.
        grid_size: Number of tiles along each side.

    Returns:
        Tensor with shape ``[grid_size * grid_size, C, tile_h, tile_w]``.
    """

    validate_grid_size(image, grid_size)
    channels, height, width = image.shape
    tile_h = height // grid_size
    tile_w = width // grid_size
    tiles = (
        image.reshape(channels, grid_size, tile_h, grid_size, tile_w)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_size * grid_size, channels, tile_h, tile_w)
    )
    return tiles


def reconstruct_from_tiles(tiles: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Reconstruct an image tensor from row-major tiles.

    Args:
        tiles: Tensor with shape ``[grid_size * grid_size, C, tile_h, tile_w]``.
        grid_size: Number of tiles along each side.

    Returns:
        Tensor with shape ``[C, H, W]``.
    """

    if tiles.ndim != 4:
        raise ValueError(f"Expected tiles with shape [N, C, tile_h, tile_w], got {tuple(tiles.shape)}")
    expected_tiles = grid_size * grid_size
    if tiles.shape[0] != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} tiles for grid_size={grid_size}, got {tiles.shape[0]}")
    _, channels, tile_h, tile_w = tiles.shape
    image = (
        tiles.reshape(grid_size, grid_size, channels, tile_h, tile_w)
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, grid_size * tile_h, grid_size * tile_w)
    )
    return image


def apply_tile_permutation(image: torch.Tensor, grid_size: int, permutation: Sequence[int]) -> torch.Tensor:
    """Apply a tile permutation to an image tensor.

    The permutation maps output tile positions to source tile indices. For example,
    ``permutation[0] == 3`` places the original tile 3 in output position 0.

    Args:
        image: Tensor with shape ``[C, H, W]``.
        grid_size: Number of tiles along each side.
        permutation: Sequence of length ``grid_size * grid_size``.

    Returns:
        Permuted image tensor with the same shape as ``image``.
    """

    tiles = split_into_tiles(image, grid_size)
    permutation_tensor = torch.as_tensor(permutation, dtype=torch.long, device=tiles.device)
    if permutation_tensor.numel() != tiles.shape[0]:
        raise ValueError(f"Permutation length must be {tiles.shape[0]}, got {permutation_tensor.numel()}")
    if sorted(permutation_tensor.cpu().tolist()) != list(range(tiles.shape[0])):
        raise ValueError("permutation must contain each tile index exactly once")
    permuted_tiles = tiles.index_select(0, permutation_tensor)
    permuted_image = reconstruct_from_tiles(permuted_tiles, grid_size)
    return permuted_image


class TilePermutationDataset(Dataset[ImageLabel]):
    """Dataset wrapper that applies fixed or sampled tile permutations.

    Args:
        base_dataset: Dataset returning ``(image_tensor, label)``.
        grid_size: Number of tiles along each image side.
        permutation: Optional fixed permutation mapping output positions to source indices.
        random_permutations: Optional list of permutations to sample per item.
        seed: Seed for deterministic per-item sampled permutations.
    """

    def __init__(
        self,
        base_dataset: TensorClassificationDataset,
        grid_size: int = 1,
        permutation: Optional[Sequence[int]] = None,
        random_permutations: Optional[Sequence[Sequence[int]]] = None,
        seed: int = 0,
    ) -> None:
        self.base_dataset = base_dataset
        self.grid_size = grid_size
        self.permutation = list(permutation) if permutation is not None else None
        self.random_permutations = [list(p) for p in random_permutations] if random_permutations else None
        self.seed = seed

        if self.permutation is not None and self.random_permutations is not None:
            raise ValueError("Use either permutation or random_permutations, not both")

    def __len__(self) -> int:
        """Return the number of examples."""

        dataset_length = len(self.base_dataset)
        return dataset_length

    def _sample_permutation(self, idx: int) -> Optional[list[int]]:
        if self.permutation is not None:
            return self.permutation
        if self.random_permutations:
            rng = random.Random(self.seed + idx)
            permutation = list(rng.choice(self.random_permutations))
            return permutation
        return None

    def __getitem__(self, idx: int) -> ImageLabel:
        """Return one possibly permuted ``(image, label)`` example."""

        image, label = self.base_dataset[idx]
        if self.grid_size == 1:
            return image, int(label)
        permutation = self._sample_permutation(idx)
        if permutation is None:
            permutation = random_permutation(self.grid_size, self.seed + idx)
        permuted_image = apply_tile_permutation(image, self.grid_size, permutation)
        sample = (permuted_image, int(label))
        return sample


class ImageFileDataset(Dataset[ImageLabel]):
    """Simple image-file dataset returning transformed tensors.

    Args:
        samples: Sequence of ``(path, label)`` pairs.
        transform: Transform applied to each loaded RGB PIL image.
    """

    def __init__(self, samples: Sequence[tuple[str, int]], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples."""

        sample_count = len(self.samples)
        return sample_count

    def __getitem__(self, idx: int) -> ImageLabel:
        """Load and transform one image sample."""

        path, label = self.samples[idx]
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        sample = (image_tensor, int(label))
        return sample
