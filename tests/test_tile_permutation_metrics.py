import pytest

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_adjacency_destruction_hardness_from_positions,
    compute_combined_hardness,
    compute_global_displacement,
    compute_spatial_permutation_entropy,
)
from src.preprocessing.tile_permutations import identity_tile_permutation


def test_global_tile_displacement():
    tile_permutation = [[(1, 1), (1, 0)], [(0, 1), (0, 0)]]

    assert compute_global_displacement(tile_permutation, 2) == 1.0


def test_spatial_permutation_entropy_baseline_and_1x1_are_zero():
    assert compute_spatial_permutation_entropy(None, None) == 0.0
    assert compute_spatial_permutation_entropy([0], 1) == 0.0
    assert compute_spatial_permutation_entropy(identity_tile_permutation(3), 3) == 0.0


def test_spatial_permutation_entropy_is_bounded():
    assert 0.0 <= compute_spatial_permutation_entropy([8, 1, 6, 3, 4, 5, 2, 7, 0], 3) <= 1.0


def test_spatial_permutation_entropy_is_higher_for_mixed_than_structured_movement():
    structured_row_shift = [3, 4, 5, 6, 7, 8, 0, 1, 2]
    mixed_scramble = [8, 1, 6, 3, 4, 5, 2, 7, 0]

    assert compute_spatial_permutation_entropy(structured_row_shift, 3) < compute_spatial_permutation_entropy(
        mixed_scramble,
        3,
    )


def test_combined_hardness_score():
    expected = 0.5 * 0.6 + 0.3 * 0.4 + 0.2 * 0.8
    assert compute_combined_hardness(
        adjacency_destruction_hardness=0.6,
        spatial_permutation_entropy=0.4,
        global_tile_displacement=0.8,
    ) == pytest.approx(expected)
    with pytest.raises(ValueError):
        compute_combined_hardness(
            adjacency_destruction_hardness=0.6,
            spatial_permutation_entropy=0.4,
            global_tile_displacement=0.8,
            weight_adj=0.4,
        )
    with pytest.raises(ValueError):
        compute_combined_hardness(
            adjacency_destruction_hardness=1.2,
            spatial_permutation_entropy=0.4,
            global_tile_displacement=0.8,
        )


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
