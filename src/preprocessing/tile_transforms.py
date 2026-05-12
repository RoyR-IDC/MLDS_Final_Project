"""Tensor transforms for square tile permutations."""

from __future__ import annotations

import torch

from src.preprocessing.tile_permutations import TilePermutation


def validate_square_tile_image(image: torch.Tensor, tiles_per_side: int) -> None:
    """Validate that an image tensor can be split into square tiles."""

    if image.ndim != 3:
        raise ValueError(f"Expected image tensor with shape [C, N, N], got {tuple(image.shape)}")
    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    _, height, width = image.shape
    if height != width:
        raise ValueError(f"Expected square image tensor with shape [C, N, N], got H={height}, W={width}")
    if height % tiles_per_side != 0:
        raise ValueError(f"Image side length N={height} must be divisible by tiles_per_side={tiles_per_side}")


def split_into_tiles(image: torch.Tensor, tiles_per_side: int) -> torch.Tensor:
    """Split an image tensor into row-major square tiles."""

    validate_square_tile_image(image, tiles_per_side)
    channels, image_size, _ = image.shape
    tile_size = image_size // tiles_per_side
    return (
        image.reshape(channels, tiles_per_side, tile_size, tiles_per_side, tile_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(tiles_per_side * tiles_per_side, channels, tile_size, tile_size)
    )


def reconstruct_from_tiles(tile_batch: torch.Tensor, tiles_per_side: int) -> torch.Tensor:
    """Reconstruct an image tensor from row-major square tiles."""

    if tile_batch.ndim != 4:
        raise ValueError(f"Expected tile_batch with shape [N, C, tile_size, tile_size], got {tuple(tile_batch.shape)}")
    expected_tiles = tiles_per_side * tiles_per_side
    if tile_batch.shape[0] != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} tiles for tiles_per_side={tiles_per_side}, got {tile_batch.shape[0]}")
    _, channels, tile_h, tile_w = tile_batch.shape
    if tile_h != tile_w:
        raise ValueError(f"Expected square tiles, got tile_h={tile_h}, tile_w={tile_w}")
    return (
        tile_batch.reshape(tiles_per_side, tiles_per_side, channels, tile_h, tile_w)
        .permute(2, 0, 3, 1, 4)
        .reshape(channels, tiles_per_side * tile_h, tiles_per_side * tile_w)
    )


class TilePermutationTransform:
    """PyTorch-style transform that applies a square tile permutation."""

    def __init__(self, tile_permutation: TilePermutation) -> None:
        self.tile_permutation = tile_permutation

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        validate_square_tile_image(image, self.tile_permutation.tiles_per_side)
        _, image_size, _ = image.shape
        tiles_per_side = self.tile_permutation.tiles_per_side
        tile_size = image_size // tiles_per_side
        output = image.new_empty(image.shape)

        for new_row, row in enumerate(self.tile_permutation.order):
            for new_col, (old_row, old_col) in enumerate(row):
                old_tile = image[
                    :,
                    old_row * tile_size : (old_row + 1) * tile_size,
                    old_col * tile_size : (old_col + 1) * tile_size,
                ]
                output[
                    :,
                    new_row * tile_size : (new_row + 1) * tile_size,
                    new_col * tile_size : (new_col + 1) * tile_size,
                ] = old_tile
        return output


def apply_tile_permutation(image: torch.Tensor, tile_permutation: TilePermutation) -> torch.Tensor:
    """Apply a tile permutation to an image tensor."""

    return TilePermutationTransform(tile_permutation)(image)
