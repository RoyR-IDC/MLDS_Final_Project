"""Tensor-based tile-order utilities for image classification."""

from __future__ import annotations

import random
from typing import Callable, Optional, Protocol, Sequence, TypeAlias

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.preprocessing.tile_orders import (
    GridSideLength,
    OutputTileOrder,
    random_tile_order,
)

ImageLabel = tuple[torch.Tensor, int]
TileBatch: TypeAlias = torch.Tensor


class TensorClassificationDataset(Protocol):
    """Dataset protocol for tensor image classification samples."""

    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    def __getitem__(self, idx: int) -> ImageLabel:
        """Return one ``(image, label)`` sample."""
        ...


def validate_grid_side_length(image: torch.Tensor, grid_side_length: GridSideLength) -> None:
    """Validate that an image tensor can be split into a square tile grid.

    Args:
        image: Image tensor with shape ``[C, H, W]``.
        grid_side_length: Number of tiles along each image side.

    Raises:
        ValueError: If the tensor shape or grid size is invalid.
    """

    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C, H, W], got {tuple(image.shape)}")
    if grid_side_length < 1:
        raise ValueError("grid_side_length must be at least 1")
    _, height, width = image.shape
    if height % grid_side_length != 0 or width % grid_side_length != 0:
        raise ValueError(
            f"Image height and width must be divisible by grid_side_length={grid_side_length}; "
            f"got H={height}, W={width}."
        )


def split_into_tiles(image: torch.Tensor, grid_side_length: GridSideLength) -> TileBatch:
    """Split an image tensor into row-major tiles.

    Args:
        image: Tensor with shape ``[C, H, W]``.
        grid_side_length: Number of tiles along each side.

    Returns:
        Tensor with shape ``[tile_count, C, tile_h, tile_w]``.
    """

    validate_grid_side_length(image, grid_side_length)
    channels, height, width = image.shape
    tile_h = height // grid_side_length
    tile_w = width // grid_side_length
    tile_batch = (
        image.reshape(channels, grid_side_length, tile_h, grid_side_length, tile_w)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_side_length * grid_side_length, channels, tile_h, tile_w)
    )
    return tile_batch


def reconstruct_from_tiles(tile_batch: TileBatch, grid_side_length: GridSideLength) -> torch.Tensor:
    """Reconstruct an image tensor from row-major tiles.

    Args:
        tile_batch: Tensor with shape ``[tile_count, C, tile_h, tile_w]``.
        grid_side_length: Number of tiles along each side.

    Returns:
        Tensor with shape ``[C, H, W]``.
    """

    if tile_batch.ndim != 4:
        raise ValueError(f"Expected tile_batch with shape [N, C, tile_h, tile_w], got {tuple(tile_batch.shape)}")
    expected_tiles = grid_side_length * grid_side_length
    if tile_batch.shape[0] != expected_tiles:
        raise ValueError(
            f"Expected {expected_tiles} tiles for grid_side_length={grid_side_length}, got {tile_batch.shape[0]}"
        )
    _, channels, tile_h, tile_w = tile_batch.shape
    image = (
        tile_batch.reshape(grid_side_length, grid_side_length, channels, tile_h, tile_w)
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, grid_side_length * tile_h, grid_side_length * tile_w)
    )
    return image


def apply_tile_order(
    image: torch.Tensor,
    grid_side_length: GridSideLength,
    output_tile_order: Sequence[int],
) -> torch.Tensor:
    """Apply an output tile order to an image tensor.

    The order maps output tile positions to source tile indices. For example,
    ``output_tile_order[0] == 3`` places the original tile 3 in output position 0.

    Args:
        image: Tensor with shape ``[C, H, W]``.
        grid_side_length: Number of tiles along each side.
        output_tile_order: Sequence of length ``grid_side_length * grid_side_length``.

    Returns:
        Reordered image tensor with the same shape as ``image``.
    """

    tile_batch = split_into_tiles(image, grid_side_length)
    tile_order_tensor = torch.as_tensor(output_tile_order, dtype=torch.long, device=tile_batch.device)
    if tile_order_tensor.numel() != tile_batch.shape[0]:
        raise ValueError(f"Output tile order length must be {tile_batch.shape[0]}, got {tile_order_tensor.numel()}")
    if sorted(tile_order_tensor.cpu().tolist()) != list(range(tile_batch.shape[0])):
        raise ValueError("output_tile_order must contain each tile index exactly once")
    reordered_tile_batch = tile_batch.index_select(0, tile_order_tensor)
    reordered_image = reconstruct_from_tiles(reordered_tile_batch, grid_side_length)
    return reordered_image


class TileOrderDataset(Dataset[ImageLabel]):
    """Dataset wrapper that applies fixed or sampled tile orders.

    Args:
        base_dataset: Dataset returning ``(image_tensor, label)``.
        grid_side_length: Number of tiles along each image side.
        output_tile_order: Optional fixed tile order mapping output positions to source indices.
        random_tile_orders: Optional list of tile orders to sample per item.
        seed: Seed for deterministic per-item sampled tile orders.
    """

    def __init__(
        self,
        base_dataset: TensorClassificationDataset,
        grid_side_length: GridSideLength = 1,
        output_tile_order: Optional[Sequence[int]] = None,
        random_tile_orders: Optional[Sequence[Sequence[int]]] = None,
        seed: int = 0,
    ) -> None:
        self.base_dataset = base_dataset
        self.grid_side_length = grid_side_length
        self.output_tile_order = list(output_tile_order) if output_tile_order is not None else None
        self.random_tile_orders = [list(tile_order) for tile_order in random_tile_orders] if random_tile_orders else None
        self.seed = seed

        if self.output_tile_order is not None and self.random_tile_orders is not None:
            raise ValueError("Use either output_tile_order or random_tile_orders, not both")

    def __len__(self) -> int:
        """Return the number of examples."""

        dataset_length = len(self.base_dataset)
        return dataset_length

    def _sample_tile_order(self, idx: int) -> Optional[OutputTileOrder]:
        if self.output_tile_order is not None:
            return self.output_tile_order
        if self.random_tile_orders:
            rng = random.Random(self.seed + idx)
            output_tile_order = list(rng.choice(self.random_tile_orders))
            return output_tile_order
        return None

    def __getitem__(self, idx: int) -> ImageLabel:
        """Return one possibly permuted ``(image, label)`` example."""

        image, label = self.base_dataset[idx]
        if self.grid_side_length == 1:
            return image, int(label)
        output_tile_order = self._sample_tile_order(idx)
        if output_tile_order is None:
            output_tile_order = random_tile_order(self.grid_side_length, self.seed + idx)
        reordered_image = apply_tile_order(image, self.grid_side_length, output_tile_order)
        sample = (reordered_image, int(label))
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
