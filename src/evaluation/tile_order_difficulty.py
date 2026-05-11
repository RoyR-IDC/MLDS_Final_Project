"""Model-agnostic Part 3 hardness metrics for image tile orders."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.preprocessing.tile_orders import GridSideLength, OutputTileOrder


def compute_global_displacement(output_tile_order: OutputTileOrder | Sequence[int], grid_side_length: GridSideLength) -> float:
    """Return normalized global Manhattan tile displacement."""

    output_tile_order_values = _validate_output_tile_order(output_tile_order, grid_side_length)
    maximum = _max_global_displacement(grid_side_length)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        output_tile_order_values,
        grid_side_length,
    )
    total_displacement = _manhattan_distances(source_coordinates, destination_coordinates).sum()
    return _clip_unit(float(total_displacement / maximum))


def compute_center_weighted_displacement(
    output_tile_order: OutputTileOrder | Sequence[int],
    grid_side_length: GridSideLength,
    alpha_center: float,
) -> float:
    """Return normalized center-weighted Manhattan tile displacement."""

    output_tile_order_values = _validate_output_tile_order(output_tile_order, grid_side_length)
    maximum = _max_center_weighted_displacement(grid_side_length, alpha_center)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        output_tile_order_values,
        grid_side_length,
    )
    weighted_distances = _center_weights(grid_side_length, alpha_center) * _manhattan_distances(
        source_coordinates,
        destination_coordinates,
    )
    return _clip_unit(float(weighted_distances.sum() / maximum))


def compute_combined_hardness(
    output_tile_order: OutputTileOrder | Sequence[int],
    grid_side_length: GridSideLength,
    alpha_center: float,
    weight_center: float = 0.5,
    weight_dist: float = 0.5,
) -> float:
    """Return the weighted Part 3 hardness score for one output tile order."""

    _validate_hardness_weights(weight_center, weight_dist)
    output_tile_order_values = _validate_output_tile_order(output_tile_order, grid_side_length)
    center_weighted_displacement = _center_weighted_displacement(
        output_tile_order_values,
        grid_side_length,
        alpha_center,
    )
    global_tile_displacement = _global_displacement(output_tile_order_values, grid_side_length)
    return _weighted_hardness_score(
        center_weighted_displacement=center_weighted_displacement,
        global_tile_displacement=global_tile_displacement,
        weight_center=weight_center,
        weight_dist=weight_dist,
    )


# =============================================================================
# Helper methods for validation, vectorized geometry, normalization, and scoring
# =============================================================================


def _validate_output_tile_order(output_tile_order: Sequence[int], grid_side_length: GridSideLength) -> np.ndarray:
    """Validate and return an output-position to source-tile order."""

    if grid_side_length < 1:
        raise ValueError("grid_side_length must be at least 1")

    values = np.asarray(output_tile_order)
    expected = grid_side_length * grid_side_length
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"Expected output_tile_order length {expected}, got {values.size}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("output_tile_order must contain integer tile indices")

    values = values.astype(np.intp, copy=False)
    if not np.array_equal(np.sort(values), np.arange(expected, dtype=np.intp)):
        raise ValueError("output_tile_order must contain each tile index exactly once")
    return values


def _tile_coordinates(grid_side_length: GridSideLength) -> np.ndarray:
    """Return row-column coordinates for row-major tile indices."""

    if grid_side_length < 1:
        raise ValueError("grid_side_length must be at least 1")
    return np.column_stack(
        np.divmod(np.arange(grid_side_length * grid_side_length, dtype=np.intp), grid_side_length)
    ).astype(float)


def _source_to_destination(output_tile_order_values: np.ndarray) -> np.ndarray:
    """Invert an output-to-source tile order into source-to-output indices."""

    destination_by_source = np.empty_like(output_tile_order_values)
    destination_by_source[output_tile_order_values] = np.arange(output_tile_order_values.size, dtype=np.intp)
    return destination_by_source


def _source_destination_coordinates(
    output_tile_order_values: np.ndarray,
    grid_side_length: GridSideLength,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original and destination coordinates for every source tile."""

    coordinates = _tile_coordinates(grid_side_length)
    destination_by_source = _source_to_destination(output_tile_order_values)
    return coordinates, coordinates[destination_by_source]


def _manhattan_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return vectorized Manhattan distances between aligned coordinate arrays."""

    return np.abs(left - right).sum(axis=-1)


def _center_weights(grid_side_length: GridSideLength, alpha_center: float) -> np.ndarray:
    """Return center-importance weights for every source tile."""

    if alpha_center < 0:
        raise ValueError("alpha_center must be non-negative")

    coordinates = _tile_coordinates(grid_side_length)
    center = (grid_side_length - 1) / 2.0
    radii = np.linalg.norm(coordinates - center, axis=1)
    return np.exp(-alpha_center * radii)


def _max_global_displacement(grid_side_length: GridSideLength) -> float:
    """Return the maximum total Manhattan displacement for a square tile grid."""

    _validate_grid_side_length(grid_side_length)
    row_indices = np.arange(grid_side_length, dtype=float)
    row_max = grid_side_length * np.abs(row_indices - row_indices[::-1]).sum()
    return float(2.0 * row_max)


def _max_center_weighted_displacement(grid_side_length: GridSideLength, alpha_center: float) -> float:
    """Return the exact maximum weighted Manhattan displacement."""

    _validate_grid_side_length(grid_side_length)
    coordinates = _tile_coordinates(grid_side_length)
    weights = _center_weights(grid_side_length, alpha_center)
    weighted_distances = weights[:, np.newaxis] * _manhattan_distances(
        coordinates[:, np.newaxis, :],
        coordinates[np.newaxis, :, :],
    )
    source_indices, destination_indices = linear_sum_assignment(weighted_distances, maximize=True)
    return float(weighted_distances[source_indices, destination_indices].sum())


def _global_displacement(output_tile_order_values: np.ndarray, grid_side_length: GridSideLength) -> float:
    """Compute global displacement for an already validated output tile order."""

    maximum = _max_global_displacement(grid_side_length)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        output_tile_order_values,
        grid_side_length,
    )
    total_displacement = _manhattan_distances(source_coordinates, destination_coordinates).sum()
    return _clip_unit(float(total_displacement / maximum))


def _center_weighted_displacement(
    output_tile_order_values: np.ndarray,
    grid_side_length: GridSideLength,
    alpha_center: float,
) -> float:
    """Compute center-weighted displacement for an already validated output tile order."""

    maximum = _max_center_weighted_displacement(grid_side_length, alpha_center)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(
        output_tile_order_values,
        grid_side_length,
    )
    weighted_distances = _center_weights(grid_side_length, alpha_center) * _manhattan_distances(
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


def _validate_grid_side_length(grid_side_length: GridSideLength) -> None:
    """Validate a tile grid side length."""

    if grid_side_length < 1:
        raise ValueError("grid_side_length must be at least 1")


def _validate_hardness_weights(weight_center: float, weight_dist: float) -> None:
    """Validate that combined hardness weights form a convex sum."""

    total_weight = weight_center + weight_dist
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weight_center + weight_dist must equal 1")


def _clip_unit(value: float) -> float:
    """Clip small floating-point drift into the normalized unit interval."""

    return float(min(1.0, max(0.0, value)))
