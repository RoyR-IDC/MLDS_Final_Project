"""Dependency-light tile-order generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, TypeAlias


TileIndex: TypeAlias = int
GridSideLength: TypeAlias = int
TileCount: TypeAlias = int
OutputTileOrder: TypeAlias = list[TileIndex]
OutputTileOrderPool: TypeAlias = list[OutputTileOrder]


@dataclass(frozen=True)
class TileOrderRecord:
    """Metadata for one reusable output tile order.

    Attributes:
        grid_side_length: Number of tiles along each image side.
        tile_order_id: Stable ID within this grid side length.
        tile_order_seed: Unified experiment seed used for this tile-order record.
        output_tile_order: Mapping from output tile position to source tile index.
    """

    grid_side_length: GridSideLength
    tile_order_id: int
    tile_order_seed: int
    output_tile_order: OutputTileOrder


def identity_tile_order(grid_side_length: GridSideLength) -> OutputTileOrder:
    """Return the identity output tile order for a square tile grid.

    Args:
        grid_side_length: Number of tiles along each image side.

    Returns:
        Row-major identity tile order.
    """

    if grid_side_length < 1:
        raise ValueError("grid_side_length must be at least 1")
    output_tile_order = list(range(grid_side_length * grid_side_length))
    return output_tile_order


def random_tile_order(grid_side_length: GridSideLength, seed: int) -> OutputTileOrder:
    """Generate one seeded random output tile order.

    Args:
        grid_side_length: Number of tiles along each image side.
        seed: Random seed.

    Returns:
        A tile order mapping output tile position to source tile index.
    """

    output_tile_order = identity_tile_order(grid_side_length)
    random.Random(seed).shuffle(output_tile_order)
    return output_tile_order


def generate_tile_orders(
    grid_side_length: GridSideLength,
    n: int,
    seed: int = 0,
) -> OutputTileOrderPool:
    """Generate reusable seeded random output tile orders for a grid.

    Args:
        grid_side_length: Number of tiles along each image side.
        n: Number of random tile orders to generate.
        seed: Seed for the tile-order generator.

    Returns:
        List of tile orders, each mapping output position to source tile index.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    output_tile_orders: OutputTileOrderPool = []
    for _ in range(n):
        output_tile_order = identity_tile_order(grid_side_length)
        rng.shuffle(output_tile_order)
        output_tile_orders.append(output_tile_order)
    return output_tile_orders


def build_tile_order_records(
    grid_side_lengths: Iterable[GridSideLength],
    num_tile_orders: int,
    seed: int = 42,
    include_identity: bool = True,
) -> list[TileOrderRecord]:
    """Build stable tile-order records for experiment reuse.

    Args:
        grid_side_lengths: Grid side lengths to include.
        num_tile_orders: Number of random tile orders per non-identity grid.
        seed: Unified experiment seed used to generate random tile orders.
        include_identity: Whether to include identity at ``tile_order_id=0``.

    Returns:
        List of tile-order records.
    """

    records: list[TileOrderRecord] = []
    for grid_side_length in grid_side_lengths:
        next_id = 0
        if include_identity:
            records.append(
                TileOrderRecord(
                    grid_side_length=grid_side_length,
                    tile_order_id=next_id,
                    tile_order_seed=seed,
                    output_tile_order=identity_tile_order(grid_side_length),
                )
            )
            next_id += 1
        for offset, output_tile_order in enumerate(generate_tile_orders(grid_side_length, num_tile_orders, seed)):
            records.append(
                TileOrderRecord(
                    grid_side_length=grid_side_length,
                    tile_order_id=next_id + offset,
                    tile_order_seed=seed,
                    output_tile_order=output_tile_order,
                )
            )
    return records
