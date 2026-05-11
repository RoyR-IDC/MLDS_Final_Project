from src.evaluation.tile_order_difficulty import (
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)


def test_identity_tile_order_has_zero_hardness_metrics():
    grid_side_length = 3
    output_tile_order = list(range(grid_side_length * grid_side_length))

    assert compute_global_displacement(output_tile_order, grid_side_length) == 0.0
    assert compute_center_weighted_displacement(output_tile_order, grid_side_length, alpha_center=1.0) == 0.0
    assert compute_combined_hardness(output_tile_order, grid_side_length, alpha_center=1.0) == 0.0


def test_nontrivial_output_tile_order_has_bounded_hardness_metrics():
    grid_side_length = 3
    output_tile_order = list(range(grid_side_length * grid_side_length))
    output_tile_order[0], output_tile_order[-1] = output_tile_order[-1], output_tile_order[0]

    assert 0.0 < compute_global_displacement(output_tile_order, grid_side_length) <= 1.0
    assert 0.0 < compute_center_weighted_displacement(output_tile_order, grid_side_length, alpha_center=1.0) <= 1.0
    assert 0.0 < compute_combined_hardness(output_tile_order, grid_side_length, alpha_center=1.0) <= 1.0
