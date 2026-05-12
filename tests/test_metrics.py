from src.evaluation.tile_permutation_difficulty import (
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)
from src.preprocessing.tile_permutations import TilePermutation, identity_tile_permutation


def test_identity_tile_permutation_has_zero_hardness_metrics():
    tiles_per_side = 3
    tile_permutation = identity_tile_permutation(tiles_per_side)

    assert compute_global_displacement(tile_permutation, tiles_per_side) == 0.0
    assert compute_center_weighted_displacement(tile_permutation, tiles_per_side, alpha_center=1.0) == 0.0
    assert compute_combined_hardness(tile_permutation, tiles_per_side, alpha_center=1.0) == 0.0


def test_nontrivial_tile_permutation_has_bounded_hardness_metrics():
    tiles_per_side = 3
    tile_permutation = TilePermutation(
        tiles_per_side=tiles_per_side,
        order=[
            [(2, 2), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (0, 0)],
        ],
    )

    assert 0.0 < compute_global_displacement(tile_permutation, tiles_per_side) <= 1.0
    assert 0.0 < compute_center_weighted_displacement(tile_permutation, tiles_per_side, alpha_center=1.0) <= 1.0
    assert 0.0 < compute_combined_hardness(tile_permutation, tiles_per_side, alpha_center=1.0) <= 1.0
