"""Model-agnostic hardness metrics for image tile permutations."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


Position = Tuple[int, int]


def validate_permutation(permutation: Sequence[int], grid_size: int) -> List[int]:
    """Validate and return an output-to-source tile permutation.

    Args:
        permutation: Mapping from output tile positions to source tile indices.
        grid_size: Number of tiles along each image side.

    Returns:
        Validated permutation as a list.

    Raises:
        ValueError: If the grid size or permutation is malformed.
    """

    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")

    expected = grid_size * grid_size
    values = list(permutation)
    if len(values) != expected:
        raise ValueError(f"Expected permutation length {expected}, got {len(values)}")
    if sorted(values) != list(range(expected)):
        raise ValueError("permutation must contain each tile index exactly once")
    return values


def get_tile_positions(N: int) -> list[Position]:
    """Return row-major ``(x, y)`` coordinates for an ``N x N`` tile grid."""

    if N < 1:
        raise ValueError("N must be at least 1")
    return [(idx // N, idx % N) for idx in range(N * N)]


def tile_coordinates(grid_size: int) -> np.ndarray:
    """Return row-column coordinates for row-major tile indices."""

    return np.array(get_tile_positions(grid_size), dtype=float)


def compute_manhattan_distance(pos_a: Position, pos_b: Position) -> int:
    """Return the Manhattan distance between two grid positions."""

    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])


def _source_to_destination(permutation: Sequence[int], grid_size: int) -> np.ndarray:
    """Invert an output-to-source permutation into source-to-output indices."""

    values = validate_permutation(permutation, grid_size)
    destination_by_source = np.empty(grid_size * grid_size, dtype=int)
    for destination, source in enumerate(values):
        destination_by_source[source] = destination
    return destination_by_source


def _source_destination_positions(permutation: Sequence[int], N: int) -> list[tuple[Position, Position]]:
    """Return original and permuted positions for each source tile."""

    positions = get_tile_positions(N)
    destination_by_source = _source_to_destination(permutation, N)
    return [(positions[source], positions[int(destination_by_source[source])]) for source in range(N * N)]


def _clip_unit(value: float) -> float:
    """Clip small floating-point drift into the normalized unit interval."""

    return float(min(1.0, max(0.0, value)))


def _hungarian_minimize(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a square minimization assignment problem with the Hungarian method."""

    cost = cost_matrix.tolist()
    size = len(cost)
    potentials_rows = [0.0] * (size + 1)
    potentials_cols = [0.0] * (size + 1)
    matching = [0] * (size + 1)
    predecessor = [0] * (size + 1)

    for row in range(1, size + 1):
        matching[0] = row
        column = 0
        min_values = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = matching[column]
            delta = float("inf")
            next_column = 0
            for candidate_column in range(1, size + 1):
                if used[candidate_column]:
                    continue
                current = (
                    cost[current_row - 1][candidate_column - 1]
                    - potentials_rows[current_row]
                    - potentials_cols[candidate_column]
                )
                if current < min_values[candidate_column]:
                    min_values[candidate_column] = current
                    predecessor[candidate_column] = column
                if min_values[candidate_column] < delta:
                    delta = min_values[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(size + 1):
                if used[candidate_column]:
                    potentials_rows[matching[candidate_column]] += delta
                    potentials_cols[candidate_column] -= delta
                else:
                    min_values[candidate_column] -= delta
            column = next_column
            if matching[column] == 0:
                break
        while True:
            next_column = predecessor[column]
            matching[column] = matching[next_column]
            column = next_column
            if column == 0:
                break

    columns_by_row = [0] * size
    for column in range(1, size + 1):
        if matching[column] != 0:
            columns_by_row[matching[column] - 1] = column - 1
    return np.arange(size), np.array(columns_by_row, dtype=int)


def _linear_sum_assignment_max(weighted_distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a maximum-weight assignment, using SciPy when it imports cleanly."""

    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(weighted_distances, maximize=True)
    except Exception:
        maximum = float(np.max(weighted_distances))
        return _hungarian_minimize(maximum - weighted_distances)


def _validate_hardness_weights(weight_adj: float, weight_center: float, weight_dist: float) -> None:
    """Validate that combined hardness weights form a convex sum."""

    total_weight = weight_adj + weight_center + weight_dist
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("weight_adj + weight_center + weight_dist must equal 1")


def compute_max_global_displacement(N: int) -> int:
    """Return the maximum total Manhattan displacement for an ``N x N`` grid."""

    if N < 1:
        raise ValueError("N must be at least 1")
    row_max = N * sum(abs(row - (N - 1 - row)) for row in range(N))
    return 2 * row_max


def compute_global_displacement(permutation: Sequence[int], N: int) -> float:
    """Return normalized global tile displacement.

    The repository stores permutations as output-position to source-tile
    mappings. This function inverts that mapping so every source tile can be
    compared with its destination position.
    """

    validate_permutation(permutation, N)
    maximum = compute_max_global_displacement(N)
    if maximum == 0:
        return 0.0

    total_displacement = sum(
        compute_manhattan_distance(original, permuted)
        for original, permuted in _source_destination_positions(permutation, N)
    )
    return _clip_unit(total_displacement / maximum)


def _center_weights(N: int, alpha_center: float) -> np.ndarray:
    """Return center-importance weights for every source tile."""

    if alpha_center < 0:
        raise ValueError("alpha_center must be non-negative")

    center = (N - 1) / 2.0
    weights = []
    for x, y in get_tile_positions(N):
        radius = math.sqrt((x - center) ** 2 + (y - center) ** 2)
        weights.append(math.exp(-alpha_center * radius))
    return np.array(weights, dtype=float)


def compute_max_center_weighted_displacement(N: int, alpha_center: float) -> float:
    """Return the exact maximum weighted Manhattan displacement.

    The weighted maximum is an assignment problem: each source tile must be
    assigned to one unique destination tile while maximizing weighted distance.
    """

    validate_permutation(list(range(N * N)), N)
    positions = get_tile_positions(N)
    weights = _center_weights(N, alpha_center)
    weighted_distances = np.array(
        [
            [weights[source] * compute_manhattan_distance(src_pos, dst_pos) for dst_pos in positions]
            for source, src_pos in enumerate(positions)
        ],
        dtype=float,
    )
    source_indices, destination_indices = _linear_sum_assignment_max(weighted_distances)
    return float(weighted_distances[source_indices, destination_indices].sum())


def compute_center_weighted_displacement(
    permutation: Sequence[int],
    N: int,
    alpha_center: float,
) -> float:
    """Return normalized center-weighted tile displacement."""

    validate_permutation(permutation, N)
    maximum = compute_max_center_weighted_displacement(N, alpha_center)
    if maximum == 0:
        return 0.0

    weights = _center_weights(N, alpha_center)
    total = 0.0
    for source, (original, permuted) in enumerate(_source_destination_positions(permutation, N)):
        total += float(weights[source]) * compute_manhattan_distance(original, permuted)
    return _clip_unit(total / maximum)


def get_original_adjacency_pairs(N: int) -> list[tuple[int, int]]:
    """Return each original horizontal and vertical tile adjacency once."""

    if N < 1:
        raise ValueError("N must be at least 1")

    pairs: list[tuple[int, int]] = []
    for source in range(N * N):
        row, col = divmod(source, N)
        if col + 1 < N:
            pairs.append((source, source + 1))
        if row + 1 < N:
            pairs.append((source, source + N))
    return pairs


def are_positions_adjacent(pos_a: Position, pos_b: Position) -> bool:
    """Return whether two positions are 4-neighbors on the grid."""

    return compute_manhattan_distance(pos_a, pos_b) == 1


def compute_adjacency_preservation_loss(permutation: Sequence[int], N: int) -> float:
    """Return the loss of original preserved tile adjacencies."""

    validate_permutation(permutation, N)
    adjacency_pairs = get_original_adjacency_pairs(N)
    if not adjacency_pairs:
        return 0.0

    positions = get_tile_positions(N)
    destination_by_source = _source_to_destination(permutation, N)
    preserved = 0
    for left, right in adjacency_pairs:
        left_pos = positions[int(destination_by_source[left])]
        right_pos = positions[int(destination_by_source[right])]
        preserved += int(are_positions_adjacent(left_pos, right_pos))
    return _clip_unit(1.0 - preserved / len(adjacency_pairs))


def combine_hardness_scores(
    adjacency_preservation_loss: float,
    center_weighted_displacement: float,
    global_tile_displacement: float,
    weight_adj: float = 0.5,
    weight_center: float = 0.3,
    weight_dist: float = 0.2,
) -> float:
    """Fuse normalized component scores into one combined hardness score."""

    _validate_hardness_weights(weight_adj, weight_center, weight_dist)
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


def compute_combined_hardness(
    permutation: Sequence[int],
    N: int,
    alpha_center: float,
    weight_adj: float = 0.5,
    weight_center: float = 0.3,
    weight_dist: float = 0.2,
) -> float:
    """Return the final weighted hardness score ``H`` for one permutation."""

    _validate_hardness_weights(weight_adj, weight_center, weight_dist)
    adjacency_preservation_loss = compute_adjacency_preservation_loss(permutation, N)
    center_weighted_displacement = compute_center_weighted_displacement(permutation, N, alpha_center)
    global_tile_displacement = compute_global_displacement(permutation, N)
    return combine_hardness_scores(
        adjacency_preservation_loss=adjacency_preservation_loss,
        center_weighted_displacement=center_weighted_displacement,
        global_tile_displacement=global_tile_displacement,
        weight_adj=weight_adj,
        weight_center=weight_center,
        weight_dist=weight_dist,
    )


def permutation_metric_row(
    permutation: Sequence[int],
    grid_size: int,
    alpha_center: float = 1.0,
    weight_adj: float = 0.5,
    weight_center: float = 0.3,
    weight_dist: float = 0.2,
) -> dict[str, float]:
    """Return the Part 3 hardness metrics for one permutation."""

    adjacency_preservation_loss = compute_adjacency_preservation_loss(permutation, grid_size)
    center_weighted_displacement = compute_center_weighted_displacement(permutation, grid_size, alpha_center)
    global_tile_displacement = compute_global_displacement(permutation, grid_size)
    combined_hardness_score = combine_hardness_scores(
        adjacency_preservation_loss=adjacency_preservation_loss,
        center_weighted_displacement=center_weighted_displacement,
        global_tile_displacement=global_tile_displacement,
        weight_adj=weight_adj,
        weight_center=weight_center,
        weight_dist=weight_dist,
    )
    return {
        "global_tile_displacement": global_tile_displacement,
        "center_weighted_displacement": center_weighted_displacement,
        "adjacency_preservation_loss": adjacency_preservation_loss,
        "combined_hardness_score": combined_hardness_score,
    }


def average_displacement(permutation: Sequence[int], grid_size: int) -> float:
    """Compute average Euclidean tile displacement in grid units.

    Kept as a compatibility wrapper for earlier notebooks and tests.
    """

    destination_by_source = _source_to_destination(permutation, grid_size)
    coords = tile_coordinates(grid_size)
    distances = np.linalg.norm(coords - coords[destination_by_source], axis=1)
    return float(distances.mean())


def normalized_average_displacement(permutation: Sequence[int], grid_size: int) -> float:
    """Compute average Euclidean displacement normalized by the grid diagonal."""

    if grid_size == 1:
        return 0.0
    diagonal = math.sqrt(2.0) * (grid_size - 1)
    return _clip_unit(average_displacement(permutation, grid_size) / diagonal)


def adjacency_preservation(permutation: Sequence[int], grid_size: int) -> float:
    """Compute the fraction of original 4-neighbor tile adjacencies preserved."""

    return _clip_unit(1.0 - compute_adjacency_preservation_loss(permutation, grid_size))


def locality_disruption(permutation: Sequence[int], grid_size: int) -> float:
    """Compute one minus adjacency preservation."""

    return compute_adjacency_preservation_loss(permutation, grid_size)


def displacement_entropy(permutation: Sequence[int], grid_size: int) -> float:
    """Compute Shannon entropy of Euclidean tile displacement magnitudes."""

    destination_by_source = _source_to_destination(permutation, grid_size)
    coords = tile_coordinates(grid_size)
    distances = np.linalg.norm(coords - coords[destination_by_source], axis=1)
    unique, counts = np.unique(distances, return_counts=True)
    if len(unique) <= 1:
        return 0.0
    probabilities = counts.astype(float) / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def combined_difficulty_score(permutation: Sequence[int], grid_size: int) -> float:
    """Compute the earlier compact 50/50 difficulty score."""

    return _clip_unit(
        0.5 * normalized_average_displacement(permutation, grid_size)
        + 0.5 * locality_disruption(permutation, grid_size)
    )


def compute_max_center_weighted_corruption(N: int, alpha_center: float) -> float:
    """Deprecated alias for ``compute_max_center_weighted_displacement``."""

    return compute_max_center_weighted_displacement(N, alpha_center)


def compute_center_weighted_corruption(
    permutation: Sequence[int],
    N: int,
    alpha_center: float,
) -> float:
    """Deprecated alias for ``compute_center_weighted_displacement``."""

    return compute_center_weighted_displacement(permutation, N, alpha_center)


def compute_adjacency_destruction(permutation: Sequence[int], N: int) -> float:
    """Deprecated alias for ``compute_adjacency_preservation_loss``."""

    return compute_adjacency_preservation_loss(permutation, N)
