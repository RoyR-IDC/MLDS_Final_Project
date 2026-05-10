"""Model-agnostic Part 3 hardness metrics for image tile permutations."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_global_displacement(permutation: Sequence[int], N: int) -> float:
    """Return normalized global Manhattan tile displacement."""

    permutation_values = _validate_permutation(permutation, N)
    maximum = _max_global_displacement(N)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(permutation_values, N)
    total_displacement = _manhattan_distances(source_coordinates, destination_coordinates).sum()
    return _clip_unit(float(total_displacement / maximum))


def compute_center_weighted_displacement(
    permutation: Sequence[int],
    N: int,
    alpha_center: float,
) -> float:
    """Return normalized center-weighted Manhattan tile displacement."""

    permutation_values = _validate_permutation(permutation, N)
    maximum = _max_center_weighted_displacement(N, alpha_center)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(permutation_values, N)
    weighted_distances = _center_weights(N, alpha_center) * _manhattan_distances(
        source_coordinates,
        destination_coordinates,
    )
    return _clip_unit(float(weighted_distances.sum() / maximum))


def compute_adjacency_preservation_loss(permutation: Sequence[int], N: int) -> float:
    """Return one minus the fraction of original 4-neighbor adjacencies preserved."""

    permutation_values = _validate_permutation(permutation, N)
    adjacency_pairs = _original_adjacency_pairs(N)
    if len(adjacency_pairs) == 0:
        return 0.0

    coordinates = _tile_coordinates(N)
    destination_by_source = _source_to_destination(permutation_values)
    destination_pairs = coordinates[destination_by_source[adjacency_pairs]]
    pair_distances = _manhattan_distances(destination_pairs[:, 0, :], destination_pairs[:, 1, :])
    preserved_count = np.count_nonzero(pair_distances == 1)
    return _clip_unit(1.0 - float(preserved_count / len(adjacency_pairs)))


def compute_combined_hardness(
    permutation: Sequence[int],
    N: int,
    alpha_center: float,
    weight_adj: float = 0.5,
    weight_center: float = 0.3,
    weight_dist: float = 0.2,
) -> float:
    """Return the weighted Part 3 hardness score for one permutation."""

    _validate_hardness_weights(weight_adj, weight_center, weight_dist)
    permutation_values = _validate_permutation(permutation, N)
    adjacency_preservation_loss = _adjacency_preservation_loss(permutation_values, N)
    center_weighted_displacement = _center_weighted_displacement(permutation_values, N, alpha_center)
    global_tile_displacement = _global_displacement(permutation_values, N)
    return _weighted_hardness_score(
        adjacency_preservation_loss=adjacency_preservation_loss,
        center_weighted_displacement=center_weighted_displacement,
        global_tile_displacement=global_tile_displacement,
        weight_adj=weight_adj,
        weight_center=weight_center,
        weight_dist=weight_dist,
    )


# =============================================================================
# Helper methods for validation, vectorized geometry, normalization, and scoring
# =============================================================================


def _validate_permutation(permutation: Sequence[int], grid_size: int) -> np.ndarray:
    """Validate and return an output-position to source-tile permutation."""

    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")

    values = np.asarray(permutation)
    expected = grid_size * grid_size
    if values.ndim != 1 or values.size != expected:
        raise ValueError(f"Expected permutation length {expected}, got {values.size}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("permutation must contain integer tile indices")

    values = values.astype(np.intp, copy=False)
    if not np.array_equal(np.sort(values), np.arange(expected, dtype=np.intp)):
        raise ValueError("permutation must contain each tile index exactly once")
    return values


def _tile_coordinates(N: int) -> np.ndarray:
    """Return row-column coordinates for row-major tile indices."""

    if N < 1:
        raise ValueError("N must be at least 1")
    return np.column_stack(np.divmod(np.arange(N * N, dtype=np.intp), N)).astype(float)


def _source_to_destination(permutation_values: np.ndarray) -> np.ndarray:
    """Invert an output-to-source permutation into source-to-output indices."""

    destination_by_source = np.empty_like(permutation_values)
    destination_by_source[permutation_values] = np.arange(permutation_values.size, dtype=np.intp)
    return destination_by_source


def _source_destination_coordinates(permutation_values: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return original and destination coordinates for every source tile."""

    coordinates = _tile_coordinates(N)
    destination_by_source = _source_to_destination(permutation_values)
    return coordinates, coordinates[destination_by_source]


def _manhattan_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return vectorized Manhattan distances between aligned coordinate arrays."""

    return np.abs(left - right).sum(axis=-1)


def _center_weights(N: int, alpha_center: float) -> np.ndarray:
    """Return center-importance weights for every source tile."""

    if alpha_center < 0:
        raise ValueError("alpha_center must be non-negative")

    coordinates = _tile_coordinates(N)
    center = (N - 1) / 2.0
    radii = np.linalg.norm(coordinates - center, axis=1)
    return np.exp(-alpha_center * radii)


def _original_adjacency_pairs(N: int) -> np.ndarray:
    """Return each original horizontal and vertical tile adjacency once."""

    if N < 1:
        raise ValueError("N must be at least 1")

    tile_indices = np.arange(N * N, dtype=np.intp).reshape(N, N)
    horizontal_pairs = np.column_stack((tile_indices[:, :-1].ravel(), tile_indices[:, 1:].ravel()))
    vertical_pairs = np.column_stack((tile_indices[:-1, :].ravel(), tile_indices[1:, :].ravel()))
    if horizontal_pairs.size == 0:
        return vertical_pairs
    if vertical_pairs.size == 0:
        return horizontal_pairs
    return np.vstack((horizontal_pairs, vertical_pairs))


def _max_global_displacement(N: int) -> float:
    """Return the maximum total Manhattan displacement for an ``N x N`` grid."""

    _validate_grid_size(N)
    row_indices = np.arange(N, dtype=float)
    row_max = N * np.abs(row_indices - row_indices[::-1]).sum()
    return float(2.0 * row_max)


def _max_center_weighted_displacement(N: int, alpha_center: float) -> float:
    """Return the exact maximum weighted Manhattan displacement."""

    _validate_grid_size(N)
    coordinates = _tile_coordinates(N)
    weights = _center_weights(N, alpha_center)
    weighted_distances = weights[:, np.newaxis] * _manhattan_distances(
        coordinates[:, np.newaxis, :],
        coordinates[np.newaxis, :, :],
    )
    source_indices, destination_indices = linear_sum_assignment(weighted_distances, maximize=True)
    return float(weighted_distances[source_indices, destination_indices].sum())


def _global_displacement(permutation_values: np.ndarray, N: int) -> float:
    """Compute global displacement for an already validated permutation."""

    maximum = _max_global_displacement(N)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(permutation_values, N)
    total_displacement = _manhattan_distances(source_coordinates, destination_coordinates).sum()
    return _clip_unit(float(total_displacement / maximum))


def _center_weighted_displacement(permutation_values: np.ndarray, N: int, alpha_center: float) -> float:
    """Compute center-weighted displacement for an already validated permutation."""

    maximum = _max_center_weighted_displacement(N, alpha_center)
    if maximum == 0:
        return 0.0

    source_coordinates, destination_coordinates = _source_destination_coordinates(permutation_values, N)
    weighted_distances = _center_weights(N, alpha_center) * _manhattan_distances(
        source_coordinates,
        destination_coordinates,
    )
    return _clip_unit(float(weighted_distances.sum() / maximum))


def _adjacency_preservation_loss(permutation_values: np.ndarray, N: int) -> float:
    """Compute adjacency loss for an already validated permutation."""

    adjacency_pairs = _original_adjacency_pairs(N)
    if len(adjacency_pairs) == 0:
        return 0.0

    coordinates = _tile_coordinates(N)
    destination_by_source = _source_to_destination(permutation_values)
    destination_pairs = coordinates[destination_by_source[adjacency_pairs]]
    pair_distances = _manhattan_distances(destination_pairs[:, 0, :], destination_pairs[:, 1, :])
    preserved_count = np.count_nonzero(pair_distances == 1)
    return _clip_unit(1.0 - float(preserved_count / len(adjacency_pairs)))


def _weighted_hardness_score(
    adjacency_preservation_loss: float,
    center_weighted_displacement: float,
    global_tile_displacement: float,
    weight_adj: float,
    weight_center: float,
    weight_dist: float,
) -> float:
    """Fuse normalized component scores into one combined hardness score."""

    for name, value in {
        "adjacency_preservation_loss": adjacency_preservation_loss,
        "center_weighted_displacement": center_weighted_displacement,
        "global_tile_displacement": global_tile_displacement,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    hardness = (
        weight_adj * adjacency_preservation_loss
        + weight_center * center_weighted_displacement
        + weight_dist * global_tile_displacement
    )
    return _clip_unit(hardness)


def _validate_grid_size(N: int) -> None:
    """Validate a tile grid side length."""

    if N < 1:
        raise ValueError("N must be at least 1")


def _validate_hardness_weights(weight_adj: float, weight_center: float, weight_dist: float) -> None:
    """Validate that combined hardness weights form a convex sum."""

    total_weight = weight_adj + weight_center + weight_dist
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weight_adj + weight_center + weight_dist must equal 1")


def _clip_unit(value: float) -> float:
    """Clip small floating-point drift into the normalized unit interval."""

    return float(min(1.0, max(0.0, value)))
