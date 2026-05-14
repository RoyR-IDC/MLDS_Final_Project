import pytest

torch = pytest.importorskip("torch")

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_adjacency_destruction_hardness_from_positions,
    compute_combined_hardness,
    compute_edge_continuity_disruption,
    compute_global_displacement,
)
from src.preprocessing.tile_permutations import identity_tile_permutation


def test_global_tile_displacement():
    tile_permutation = [[(1, 1), (1, 0)], [(0, 1), (0, 0)]]

    assert compute_global_displacement(tile_permutation, 2) == 1.0


def test_edge_continuity_disruption_baseline_and_1x1_are_zero():
    image = torch.zeros(3, 4, 4)

    assert compute_edge_continuity_disruption(image, None, None) == 0.0
    assert compute_edge_continuity_disruption(image, [0], 1) == 0.0


def test_edge_continuity_disruption_uses_touching_horizontal_and_vertical_borders():
    image = torch.tensor(
        [
            [
                [1.0, 2.0, 10.0, 11.0],
                [3.0, 4.0, 12.0, 13.0],
                [20.0, 21.0, 30.0, 31.0],
                [22.0, 23.0, 32.0, 33.0],
            ]
        ]
    )

    expected = (
        torch.linalg.vector_norm(torch.tensor([2.0 - 10.0, 4.0 - 12.0]))
        + torch.linalg.vector_norm(torch.tensor([21.0 - 30.0, 23.0 - 32.0]))
        + torch.linalg.vector_norm(torch.tensor([3.0 - 20.0, 4.0 - 21.0]))
        + torch.linalg.vector_norm(torch.tensor([12.0 - 30.0, 13.0 - 31.0]))
    )

    assert compute_edge_continuity_disruption(image, identity_tile_permutation(2), 2) == pytest.approx(
        float(expected)
    )


def test_edge_continuity_disruption_is_lower_for_smooth_identity_than_scrambled_permutation():
    row_gradient = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8).expand(3, 8, 8)
    identity_score = compute_edge_continuity_disruption(row_gradient, identity_tile_permutation(2), 2)
    scrambled_score = compute_edge_continuity_disruption(row_gradient, [3, 2, 1, 0], 2)

    assert identity_score < scrambled_score


def test_combined_hardness_score():
    expected = 0.5 * 0.6 + 0.3 * 0.4 + 0.2 * 0.8
    assert compute_combined_hardness(
        adjacency_destruction_hardness=0.6,
        edge_continuity_disruption=0.4,
        global_tile_displacement=0.8,
    ) == pytest.approx(expected)
    with pytest.raises(ValueError):
        compute_combined_hardness(
            adjacency_destruction_hardness=0.6,
            edge_continuity_disruption=0.4,
            global_tile_displacement=0.8,
            weight_adj=0.4,
        )
    with pytest.raises(ValueError):
        compute_combined_hardness(
            adjacency_destruction_hardness=1.2,
            edge_continuity_disruption=0.4,
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
