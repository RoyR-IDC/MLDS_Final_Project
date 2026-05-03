"""Model-agnostic tile permutation difficulty metrics."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def validate_permutation(permutation: Sequence[int], grid_size: int) -> List[int]:
    """Validate and return a tile permutation as a list.

    Args:
        permutation: Mapping from output tile positions to source tile indices.
        grid_size: Number of tiles along each image side.

    Returns:
        Validated permutation list.

    Raises:
        ValueError: If the permutation is malformed.
    """

    expected = grid_size * grid_size
    values = list(permutation)
    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    if len(values) != expected:
        raise ValueError(f"Expected permutation length {expected}, got {len(values)}")
    if sorted(values) != list(range(expected)):
        raise ValueError("permutation must contain each tile index exactly once")
    return values


def tile_coordinates(grid_size: int) -> np.ndarray:
    """Return row-column coordinates for row-major tile indices."""

    return np.array([(idx // grid_size, idx % grid_size) for idx in range(grid_size * grid_size)], dtype=float)


def _source_to_destination(permutation: Sequence[int], grid_size: int) -> np.ndarray:
    permutation = validate_permutation(permutation, grid_size)
    destination_by_source = np.empty(grid_size * grid_size, dtype=int)
    for destination, source in enumerate(permutation):
        destination_by_source[source] = destination
    return destination_by_source


def average_displacement(permutation: Sequence[int], grid_size: int) -> float:
    """Compute average Euclidean tile displacement in grid units.

    Args:
        permutation: Mapping from output tile positions to source tile indices.
        grid_size: Number of tiles along each image side.

    Returns:
        Mean displacement before normalization.
    """

    destination_by_source = _source_to_destination(permutation, grid_size)
    coords = tile_coordinates(grid_size)
    distances = np.linalg.norm(coords - coords[destination_by_source], axis=1)
    return float(distances.mean())


def normalized_average_displacement(permutation: Sequence[int], grid_size: int) -> float:
    """Compute average displacement normalized by the grid diagonal."""

    if grid_size == 1:
        return 0.0
    diagonal = math.sqrt(2.0) * (grid_size - 1)
    return float(average_displacement(permutation, grid_size) / diagonal)


def adjacency_preservation(permutation: Sequence[int], grid_size: int) -> float:
    """Compute the fraction of original 4-neighbor tile adjacencies preserved."""

    destination_by_source = _source_to_destination(permutation, grid_size)
    coords = tile_coordinates(grid_size)
    adjacent_pairs: List[Tuple[int, int]] = []
    for source in range(grid_size * grid_size):
        row, col = divmod(source, grid_size)
        if col + 1 < grid_size:
            adjacent_pairs.append((source, source + 1))
        if row + 1 < grid_size:
            adjacent_pairs.append((source, source + grid_size))
    if not adjacent_pairs:
        return 1.0
    kept = 0
    for left, right in adjacent_pairs:
        left_coord = coords[destination_by_source[left]]
        right_coord = coords[destination_by_source[right]]
        kept += int(np.abs(left_coord - right_coord).sum() == 1)
    return float(kept / len(adjacent_pairs))


def locality_disruption(permutation: Sequence[int], grid_size: int) -> float:
    """Compute one minus adjacency preservation."""

    return float(1.0 - adjacency_preservation(permutation, grid_size))


def displacement_entropy(permutation: Sequence[int], grid_size: int) -> float:
    """Compute Shannon entropy of tile displacement magnitudes."""

    destination_by_source = _source_to_destination(permutation, grid_size)
    coords = tile_coordinates(grid_size)
    distances = np.linalg.norm(coords - coords[destination_by_source], axis=1)
    unique, counts = np.unique(distances, return_counts=True)
    if len(unique) <= 1:
        return 0.0
    probabilities = counts.astype(float) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def combined_difficulty_score(permutation: Sequence[int], grid_size: int) -> float:
    """Compute a compact difficulty score from displacement and locality disruption."""

    return float(0.5 * normalized_average_displacement(permutation, grid_size) + 0.5 * locality_disruption(permutation, grid_size))


def permutation_metric_row(permutation: Sequence[int], grid_size: int) -> dict:
    """Return all implemented permutation metrics as one dictionary."""

    return {
        "average_displacement": average_displacement(permutation, grid_size),
        "normalized_average_displacement": normalized_average_displacement(permutation, grid_size),
        "adjacency_preservation": adjacency_preservation(permutation, grid_size),
        "locality_disruption": locality_disruption(permutation, grid_size),
        "displacement_entropy": displacement_entropy(permutation, grid_size),
        "combined_difficulty": combined_difficulty_score(permutation, grid_size),
    }

