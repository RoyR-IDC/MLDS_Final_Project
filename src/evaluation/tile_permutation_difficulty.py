"""Model-agnostic Part 3 hardness metrics for image tile permutations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.preprocessing.tile_permutations import TilePermutation, TilesPerSide, matrix_to_flat_order


def compute_global_displacement(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: TilesPerSide | None,
) -> float:
    """Return normalized global Manhattan tile displacement."""

    if tile_permutation is None:
        return 0.0
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


def compute_center_weighted_displacement(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: TilesPerSide | None,
    alpha_center: float,
) -> float:
    """Return normalized center-weighted Manhattan tile displacement."""

    if tile_permutation is None:
        return 0.0
    tile_permutation_values = _validate_tile_permutation(tile_permutation, tiles_per_side)
    assert tiles_per_side is not None
    maximum = _max_center_weighted_displacement(tiles_per_side, alpha_center)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        tile_permutation_values,
        tiles_per_side,
    )
    weighted_distances = _center_weights(tiles_per_side, alpha_center) * _manhattan_distances(
        source_coordinates,
        destination_coordinates,
    )
    return _clip_unit(float(weighted_distances.sum() / maximum))


def compute_combined_hardness(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: TilesPerSide | None,
    alpha_center: float,
    weight_center: float = 0.5,
    weight_dist: float = 0.5,
) -> float:
    """Return the weighted Part 3 hardness score for one tile permutation."""

    if tile_permutation is None:
        return 0.0
    _validate_hardness_weights(weight_center, weight_dist)
    tile_permutation_values = _validate_tile_permutation(tile_permutation, tiles_per_side)
    assert tiles_per_side is not None
    center_weighted_displacement = _center_weighted_displacement(
        tile_permutation_values,
        tiles_per_side,
        alpha_center,
    )
    global_tile_displacement = _global_displacement(tile_permutation_values, tiles_per_side)
    return _weighted_hardness_score(
        center_weighted_displacement=center_weighted_displacement,
        global_tile_displacement=global_tile_displacement,
        weight_center=weight_center,
        weight_dist=weight_dist,
    )


# =============================================================================
# Helper methods for validation, vectorized geometry, normalization, and scoring
# =============================================================================


def _validate_tile_permutation(
    tile_permutation: TilePermutation | Sequence[int] | None,
    tiles_per_side: TilesPerSide | None,
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


def _tile_coordinates(tiles_per_side: TilesPerSide) -> np.ndarray:
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
    tiles_per_side: TilesPerSide,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original and destination coordinates for every source tile."""

    coordinates = _tile_coordinates(tiles_per_side)
    destination_by_source = _source_to_destination(tile_permutation_values)
    return coordinates, coordinates[destination_by_source]


def _manhattan_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return vectorized Manhattan distances between aligned coordinate arrays."""

    return np.abs(left - right).sum(axis=-1)


def _center_weights(tiles_per_side: TilesPerSide, alpha_center: float) -> np.ndarray:
    """Return center-importance weights for every source tile."""

    if alpha_center < 0:
        raise ValueError("alpha_center must be non-negative")

    coordinates = _tile_coordinates(tiles_per_side)
    center = (tiles_per_side - 1) / 2.0
    radii = np.linalg.norm(coordinates - center, axis=1)
    return np.exp(-alpha_center * radii)


def _max_global_displacement(tiles_per_side: TilesPerSide) -> float:
    """Return the maximum total Manhattan displacement for a square tile grid."""

    _validate_tiles_per_side(tiles_per_side)
    row_indices = np.arange(tiles_per_side, dtype=float)
    row_max = tiles_per_side * np.abs(row_indices - row_indices[::-1]).sum()
    return float(2.0 * row_max)


def _max_center_weighted_displacement(tiles_per_side: TilesPerSide, alpha_center: float) -> float:
    """Return the exact maximum weighted Manhattan displacement."""

    _validate_tiles_per_side(tiles_per_side)
    coordinates = _tile_coordinates(tiles_per_side)
    weights = _center_weights(tiles_per_side, alpha_center)
    weighted_distances = weights[:, np.newaxis] * _manhattan_distances(
        coordinates[:, np.newaxis, :],
        coordinates[np.newaxis, :, :],
    )
    source_indices, destination_indices = linear_sum_assignment(weighted_distances, maximize=True)
    return float(weighted_distances[source_indices, destination_indices].sum())


def _global_displacement(tile_permutation_values: np.ndarray, tiles_per_side: TilesPerSide) -> float:
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


def _center_weighted_displacement(
    tile_permutation_values: np.ndarray,
    tiles_per_side: TilesPerSide,
    alpha_center: float,
) -> float:
    """Compute center-weighted displacement for an already validated tile permutation."""

    maximum = _max_center_weighted_displacement(tiles_per_side, alpha_center)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        tile_permutation_values,
        tiles_per_side,
    )
    weighted_distances = _center_weights(tiles_per_side, alpha_center) * _manhattan_distances(
        source_coordinates,
        destination_coordinates,
    )
    return _clip_unit(float(weighted_distances.sum() / maximum))


def _weighted_hardness_score(
    center_weighted_displacement: float,
    global_tile_displacement: float,
    weight_center: float,
    weight_dist: float,
) -> float:
    """Fuse normalized component scores into one combined hardness score."""

    for name, value in {
        "center_weighted_displacement": center_weighted_displacement,
        "global_tile_displacement": global_tile_displacement,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    hardness = weight_center * center_weighted_displacement + weight_dist * global_tile_displacement
    return _clip_unit(hardness)


def _validate_tiles_per_side(tiles_per_side: TilesPerSide) -> None:
    """Validate a tile grid side length."""

    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")


def _validate_hardness_weights(weight_center: float, weight_dist: float) -> None:
    """Validate that combined hardness weights form a convex sum."""

    total_weight = weight_center + weight_dist
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weight_center + weight_dist must equal 1")


def _clip_unit(value: float) -> float:
    """Clip small floating-point drift into the normalized unit interval."""

    return float(min(1.0, max(0.0, value)))
