from src.evaluation.permutation_difficulty import (
    compute_adjacency_preservation_loss,
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)


def test_identity_permutation_has_zero_hardness_metrics():
    grid_size = 3
    permutation = list(range(grid_size * grid_size))

    assert compute_global_displacement(permutation, grid_size) == 0.0
    assert compute_center_weighted_displacement(permutation, grid_size, alpha_center=1.0) == 0.0
    assert compute_adjacency_preservation_loss(permutation, grid_size) == 0.0
    assert compute_combined_hardness(permutation, grid_size, alpha_center=1.0) == 0.0


def test_nontrivial_permutation_has_bounded_hardness_metrics():
    grid_size = 3
    permutation = list(range(grid_size * grid_size))
    permutation[0], permutation[-1] = permutation[-1], permutation[0]

    assert 0.0 < compute_global_displacement(permutation, grid_size) <= 1.0
    assert 0.0 < compute_center_weighted_displacement(permutation, grid_size, alpha_center=1.0) <= 1.0
    assert 0.0 <= compute_adjacency_preservation_loss(permutation, grid_size) <= 1.0
    assert 0.0 < compute_combined_hardness(permutation, grid_size, alpha_center=1.0) <= 1.0
