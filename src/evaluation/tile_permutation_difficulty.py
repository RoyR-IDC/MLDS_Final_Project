"""Model-agnostic Part 3 hardness metrics for image tile permutations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from src.preprocessing.tile_permutations import TilePermutation, matrix_to_flat_order


def compute_global_displacement(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: int | None,
) -> float:
    """Return normalized global Manhattan tile displacement."""

    if tile_permutation is None:
        return 0.0
    if isinstance(tile_permutation, TilePermutation):
        tiles_per_side = tile_permutation.tiles_per_side
    tile_permutation_values = _validate_tile_permutation(tile_permutation, tiles_per_side)
    assert tiles_per_side is not None
    maximum = _max_global_displacement(tiles_per_side)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        tile_permutation_values,
        tiles_per_side,
    )
    total_displacement = _manhattan_distances(source_coordinates, destination_coordinates).sum()
    return _clip_unit(float(total_displacement / maximum))


def compute_adjacency_destruction_hardness(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: int | None,
) -> float:
    """Return strict directional adjacency destruction hardness for a tile grid.

    The metric measures destruction of local spatial structure caused by a tile
    permutation. Natural image recognition depends heavily on local continuity
    and neighboring relationships, so preserving nearby tiles in their original
    relative positions keeps more of the image structure intact. This score is
    model-agnostic because it evaluates only the geometric structure of the
    permutation, not any model outputs or architecture-specific behavior.

    For an ``N x N`` grid, each original rightward and downward adjacency is
    counted once, giving ``M = 2 * N * (N - 1)`` directed relations. A relation is
    preserved only when the same two source tiles remain adjacent after
    permutation and their relative direction is unchanged. This strict
    directional requirement is stronger than undirected adjacency: if two
    neighboring tiles remain next to each other but are reversed, the local
    continuity has changed and the relation is not counted as preserved.

    The returned hardness is ``1 - (A_preserved / M)``. The leading ``1 -``
    converts a structure-preservation score into a hardness score: preserved
    structure gives low hardness, while destroyed structure gives high hardness.
    The score is normalized to ``[0, 1]`` by dividing by the total number of
    original directed adjacency relations. A ``1 x 1`` grid has no adjacencies,
    so its hardness is defined as ``0.0``. ``None`` is treated as the unpermuted
    baseline and also returns ``0.0``.

    Time complexity is ``O(N^2)`` because each tile and each original adjacency
    is visited a constant number of times. Space complexity is ``O(N^2)`` for the
    source-to-destination coordinate lookup.
    """

    if tile_permutation is None:
        return 0.0
    if isinstance(tile_permutation, TilePermutation):
        tiles_per_side = tile_permutation.tiles_per_side
    tile_permutation_values = _validate_tile_permutation(tile_permutation, tiles_per_side)
    assert tiles_per_side is not None
    original_positions = _tile_coordinates(tiles_per_side).astype(np.intp)
    _, permuted_positions = _source_destination_coordinates(tile_permutation_values, tiles_per_side)
    return compute_adjacency_destruction_hardness_from_positions(
        original_positions,
        permuted_positions.astype(np.intp),
        tiles_per_side,
    )


def compute_adjacency_destruction_hardness_from_positions(
    original_positions: Sequence[Sequence[int]] | np.ndarray,
    permuted_positions: Sequence[Sequence[int]] | np.ndarray,
    tiles_per_side: int,
) -> float:
    """Return strict adjacency hardness from explicit source tile positions.

    Args:
        original_positions: Row-column coordinates for each source tile before
            permutation, indexed by source tile id.
        permuted_positions: Row-column coordinates for the same source tiles
            after permutation, indexed by source tile id.
        tiles_per_side: Side length of the square tile grid.

    Returns:
        Normalized adjacency destruction hardness in ``[0, 1]``.

    Raises:
        ValueError: If coordinates do not describe each grid position exactly
            once, if the arrays have the wrong shape, or if ``tiles_per_side`` is
            less than one.
    """

    _validate_tiles_per_side(tiles_per_side)
    maximum = _total_directed_adjacencies(tiles_per_side)
    if maximum == 0:
        return 0.0

    original = _validate_position_coordinates(original_positions, tiles_per_side, "original_positions")
    permuted = _validate_position_coordinates(permuted_positions, tiles_per_side, "permuted_positions")
    source_by_original_position = _source_by_position(original, tiles_per_side)

    preserved = 0
    right_delta = np.asarray([0, 1], dtype=np.intp)
    down_delta = np.asarray([1, 0], dtype=np.intp)
    for row in range(tiles_per_side):
        for col in range(tiles_per_side):
            source = source_by_original_position[row, col]
            source_destination = permuted[source]
            if col + 1 < tiles_per_side:
                right_source = source_by_original_position[row, col + 1]
                # Strict direction means source must remain immediately left of
                # the same neighbor; reversed adjacency is not preserved.
                if np.array_equal(permuted[right_source] - source_destination, right_delta):
                    preserved += 1
            if row + 1 < tiles_per_side:
                down_source = source_by_original_position[row + 1, col]
                # Strict direction means source must remain immediately above
                # the same neighbor; undirected neighbor retention is not enough.
                if np.array_equal(permuted[down_source] - source_destination, down_delta):
                    preserved += 1

    return _clip_unit(1.0 - float(preserved / maximum))


def compute_combined_hardness(
    *,
    adjacency_destruction_hardness: float,
    spatial_permutation_entropy: float,
    global_tile_displacement: float,
    weight_adj: float = 0.5,
    weight_entropy: float = 0.3,
    weight_dist: float = 0.2,
) -> float:
    """Return the weighted Part 3 hardness score from normalized components."""

    _validate_hardness_weights(weight_adj, weight_entropy, weight_dist)
    return _weighted_hardness_score(
        adjacency_destruction_hardness=adjacency_destruction_hardness,
        spatial_permutation_entropy=spatial_permutation_entropy,
        global_tile_displacement=global_tile_displacement,
        weight_adj=weight_adj,
        weight_entropy=weight_entropy,
        weight_dist=weight_dist,
    )


def compute_spatial_permutation_entropy(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: int | None,
) -> float:
    """Return normalized compass/distance-tier entropy for a tile permutation."""

    if tile_permutation is None:
        return 0.0
    if isinstance(tile_permutation, TilePermutation):
        tiles_per_side = tile_permutation.tiles_per_side
    if tiles_per_side is None:
        return 0.0
    tile_permutation_values = _validate_tile_permutation(tile_permutation, tiles_per_side)
    if tiles_per_side == 1:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        tile_permutation_values,
        tiles_per_side,
    )
    displacement = destination_coordinates.astype(np.intp) - source_coordinates.astype(np.intp)
    bins = [_movement_bin(delta, tiles_per_side) for delta in displacement]
    _, counts = np.unique(np.asarray(bins, dtype=object), return_counts=True)
    probabilities = counts.astype(float) / float(tile_permutation_values.size)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum = math.log(tile_permutation_values.size)
    if maximum == 0.0:
        return 0.0
    return _clip_unit(entropy / maximum)


# =============================================================================
# Helper methods for validation, vectorized geometry, normalization, and scoring
# =============================================================================


def _validate_tile_permutation(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: int | None,
) -> np.ndarray:
    """Validate and return an output-position to source-tile permutation."""

    if tile_permutation is None:
        return np.asarray([0], dtype=np.intp)
    if isinstance(tile_permutation, TilePermutation):
        tiles_per_side = tile_permutation.tiles_per_side
        tile_permutation = matrix_to_flat_order(tile_permutation)
    if tiles_per_side is None:
        raise ValueError("tiles_per_side is required when tile_permutation is not None")
    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")

    values = np.asarray(tile_permutation)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[..., 0] * tiles_per_side + values[..., 1]
    if values.ndim == 2:
        values = values.reshape(-1)
    expected = tiles_per_side * tiles_per_side
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"Expected tile_permutation length {expected}, got {values.size}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("tile_permutation must contain integer tile indices")

    values = values.astype(np.intp, copy=False)
    if not np.array_equal(np.sort(values), np.arange(expected, dtype=np.intp)):
        raise ValueError("tile_permutation must contain each tile index exactly once")
    return values


def _tile_coordinates(tiles_per_side: int) -> np.ndarray:
    """Return row-column coordinates for row-major tile indices."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    return np.column_stack(
        np.divmod(np.arange(tiles_per_side * tiles_per_side, dtype=np.intp), tiles_per_side)
    ).astype(float)


def _source_to_destination(tile_permutation_values: np.ndarray) -> np.ndarray:
    """Invert an output-to-source tile permutation into source-to-output indices."""

    destination_by_source = np.empty_like(tile_permutation_values)
    destination_by_source[tile_permutation_values] = np.arange(tile_permutation_values.size, dtype=np.intp)
    return destination_by_source


def _source_destination_coordinates(
    tile_permutation_values: np.ndarray,
    tiles_per_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original and destination coordinates for every source tile."""

    coordinates = _tile_coordinates(tiles_per_side)
    destination_by_source = _source_to_destination(tile_permutation_values)
    return coordinates, coordinates[destination_by_source]


def _manhattan_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return vectorized Manhattan distances between aligned coordinate arrays."""

    return np.abs(left - right).sum(axis=-1)


def _max_global_displacement(tiles_per_side: int) -> float:
    """Return the maximum total Manhattan displacement for a square tile grid."""

    _validate_tiles_per_side(tiles_per_side)
    row_indices = np.arange(tiles_per_side, dtype=float)
    row_max = tiles_per_side * np.abs(row_indices - row_indices[::-1]).sum()
    return float(2.0 * row_max)


def _global_displacement(tile_permutation_values: np.ndarray, tiles_per_side: int) -> float:
    """Compute global displacement for an already validated tile permutation."""

    maximum = _max_global_displacement(tiles_per_side)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        tile_permutation_values,
        tiles_per_side,
    )
    total_displacement = _manhattan_distances(source_coordinates, destination_coordinates).sum()
    return _clip_unit(float(total_displacement / maximum))


def _weighted_hardness_score(
    adjacency_destruction_hardness: float,
    spatial_permutation_entropy: float,
    global_tile_displacement: float,
    weight_adj: float,
    weight_entropy: float,
    weight_dist: float,
) -> float:
    """Fuse normalized component scores into one combined hardness score."""

    for name, value in {
        "adjacency_destruction_hardness": adjacency_destruction_hardness,
        "spatial_permutation_entropy": spatial_permutation_entropy,
        "global_tile_displacement": global_tile_displacement,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    hardness = (
        weight_adj * adjacency_destruction_hardness
        + weight_entropy * spatial_permutation_entropy
        + weight_dist * global_tile_displacement
    )
    return _clip_unit(hardness)


def _movement_bin(displacement: np.ndarray, tiles_per_side: int) -> str:
    """Return the compass direction and distance tier for one tile displacement."""

    row_delta, col_delta = (int(displacement[0]), int(displacement[1]))
    manhattan = abs(row_delta) + abs(col_delta)
    if manhattan == 0:
        return "stationary"

    direction = ""
    if row_delta < 0:
        direction += "N"
    elif row_delta > 0:
        direction += "S"
    if col_delta < 0:
        direction += "W"
    elif col_delta > 0:
        direction += "E"

    normalized_distance = manhattan / float(2 * (tiles_per_side - 1))
    if normalized_distance <= 1.0 / 3.0:
        tier = "near"
    elif normalized_distance <= 2.0 / 3.0:
        tier = "medium"
    else:
        tier = "far"
    return f"{direction}:{tier}"


def _validate_tiles_per_side(tiles_per_side: int) -> None:
    """Validate a tile grid side length."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")


def _total_directed_adjacencies(tiles_per_side: int) -> int:
    """Return the number of canonical right/down adjacency relations."""

    _validate_tiles_per_side(tiles_per_side)
    return 2 * tiles_per_side * (tiles_per_side - 1)


def _validate_position_coordinates(
    positions: Sequence[Sequence[int]] | np.ndarray,
    tiles_per_side: int,
    name: str,
) -> np.ndarray:
    """Validate explicit row-column positions indexed by source tile."""

    values = np.asarray(positions)
    expected = tiles_per_side * tiles_per_side
    if values.shape != (expected, 2):
        raise ValueError(f"{name} must have shape ({expected}, 2)")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must contain integer row-column coordinates")
    values = values.astype(np.intp, copy=False)
    if np.any(values < 0) or np.any(values >= tiles_per_side):
        raise ValueError(f"{name} coordinates must be within [0, {tiles_per_side - 1}]")
    flat_positions = values[:, 0] * tiles_per_side + values[:, 1]
    if not np.array_equal(np.sort(flat_positions), np.arange(expected, dtype=np.intp)):
        raise ValueError(f"{name} must contain each grid position exactly once")
    return values


def _source_by_position(positions: np.ndarray, tiles_per_side: int) -> np.ndarray:
    """Return a grid mapping each row-column position to its source tile index."""

    source_by_position = np.empty((tiles_per_side, tiles_per_side), dtype=np.intp)
    for source, (row, col) in enumerate(positions):
        source_by_position[row, col] = source
    return source_by_position


def _validate_hardness_weights(weight_adj: float, weight_entropy: float, weight_dist: float) -> None:
    """Validate that combined hardness weights form a convex sum."""

    total_weight = weight_adj + weight_entropy + weight_dist
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weight_adj + weight_entropy + weight_dist must equal 1")


def _clip_unit(value: float) -> float:
    """Clip small floating-point drift into the normalized unit interval."""

    return float(min(1.0, max(0.0, value)))
