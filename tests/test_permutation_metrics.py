import math

import pytest

from src.evaluation.permutation_difficulty import (
    combine_hardness_scores,
    compute_adjacency_preservation_loss,
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)


def test_global_tile_displacement():
    permutation = [3, 2, 1, 0]

    assert compute_global_displacement(permutation, 2) == 1.0


def test_center_weighted_displacement():
    alpha_center = 1.0
    center_weight = math.exp(-alpha_center * 0.0)
    corner_weight = math.exp(-alpha_center * math.sqrt(2.0))
    assert center_weight > corner_weight

    center_swap = list(range(25))
    center_swap[12], center_swap[13] = center_swap[13], center_swap[12]
    outer_swap = list(range(25))
    outer_swap[0], outer_swap[1] = outer_swap[1], outer_swap[0]

    center_score = compute_center_weighted_displacement(center_swap, 5, alpha_center)
    outer_score = compute_center_weighted_displacement(outer_swap, 5, alpha_center)

    assert center_score > outer_score
    assert 0.0 <= center_score <= 1.0
    assert 0.0 <= outer_score <= 1.0


def test_adjacency_preservation_loss():
    assert compute_adjacency_preservation_loss([0, 1, 2, 3], 2) == 0.0

    swap_top_row = [1, 0, 2, 3]

    assert compute_adjacency_preservation_loss(swap_top_row, 2) == 0.5


def test_combined_hardness_score():
    assert combine_hardness_scores(
        adjacency_preservation_loss=0.8,
        center_weighted_displacement=0.5,
        global_tile_displacement=1.0,
        weight_adj=0.5,
        weight_center=0.3,
        weight_dist=0.2,
    ) == pytest.approx(0.75)

    assert 0.0 <= compute_combined_hardness([3, 2, 1, 0], 2, alpha_center=1.0) <= 1.0
    with pytest.raises(ValueError):
        compute_combined_hardness([0, 1, 2, 3], 2, alpha_center=1.0, weight_adj=0.4)
