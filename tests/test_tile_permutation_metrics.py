import math

import pytest

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_adjacency_destruction_hardness_from_positions,
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)
from src.preprocessing.tile_permutations import identity_tile_permutation


def test_global_tile_displacement():
    tile_permutation = [[(1, 1), (1, 0)], [(0, 1), (0, 0)]]

    assert compute_global_displacement(tile_permutation, 2) == 1.0


def test_center_weighted_displacement():
    alpha_center = 1.0
    center_weight = math.exp(-alpha_center * 0.0)
    corner_weight = math.exp(-alpha_center * math.sqrt(2.0))
    assert center_weight > corner_weight

    center_swap = [[(row, col) for col in range(5)] for row in range(5)]
    center_swap[2][2], center_swap[2][3] = center_swap[2][3], center_swap[2][2]
    outer_swap = [[(row, col) for col in range(5)] for row in range(5)]
    outer_swap[0][0], outer_swap[0][1] = outer_swap[0][1], outer_swap[0][0]

    center_score = compute_center_weighted_displacement(center_swap, 5, alpha_center)
    outer_score = compute_center_weighted_displacement(outer_swap, 5, alpha_center)

    assert center_score > outer_score
    assert 0.0 <= center_score <= 1.0
    assert 0.0 <= outer_score <= 1.0


def test_combined_hardness_score():
    tile_permutation = [[(1, 1), (1, 0)], [(0, 1), (0, 0)]]
    expected = 0.5 * compute_global_displacement(tile_permutation, 2) + 0.5 * compute_center_weighted_displacement(
        tile_permutation,
        2,
        alpha_center=1.0,
    )
    assert compute_combined_hardness(tile_permutation, 2, alpha_center=1.0) == pytest.approx(expected)
    with pytest.raises(ValueError):
        compute_combined_hardness([0, 1, 2, 3], 2, alpha_center=1.0, weight_center=0.4)


def test_adjacency_destruction_hardness_baseline_and_identity():
    assert compute_adjacency_destruction_hardness(None, None) == 0.0
    assert compute_adjacency_destruction_hardness([0], 1) == 0.0
    assert compute_adjacency_destruction_hardness(identity_tile_permutation(2), 2) == 0.0
    assert compute_adjacency_destruction_hardness(identity_tile_permutation(3), 3) == 0.0


def test_adjacency_destruction_hardness_manual_2x2_examples():
    full_reversal = [3, 2, 1, 0]
    row_swap = [2, 3, 0, 1]
    top_neighbors_reversed = [1, 0, 2, 3]

    assert compute_adjacency_destruction_hardness(full_reversal, 2) == 1.0
    assert compute_adjacency_destruction_hardness(row_swap, 2) == pytest.approx(0.5)
    assert compute_adjacency_destruction_hardness(top_neighbors_reversed, 2) == pytest.approx(0.75)


def test_adjacency_destruction_hardness_accepts_explicit_positions():
    original_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    row_swapped_positions = [(1, 0), (1, 1), (0, 0), (0, 1)]

    assert compute_adjacency_destruction_hardness_from_positions(
        original_positions,
        row_swapped_positions,
        2,
    ) == pytest.approx(0.5)


def test_adjacency_destruction_hardness_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="length 4"):
        compute_adjacency_destruction_hardness([0, 1, 2], 2)
    with pytest.raises(ValueError, match="each tile index exactly once"):
        compute_adjacency_destruction_hardness([0, 1, 1, 2], 2)
    with pytest.raises(ValueError, match="integer"):
        compute_adjacency_destruction_hardness([0.0, 1.0, 2.0, 3.0], 2)
    with pytest.raises(ValueError, match="at least 1"):
        compute_adjacency_destruction_hardness([0], 0)
    with pytest.raises(ValueError, match="each grid position exactly once"):
        compute_adjacency_destruction_hardness_from_positions(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 0), (1, 0), (1, 1)],
            2,
        )
