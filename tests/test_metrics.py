from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_combined_hardness,
    compute_global_displacement,
    compute_spatial_permutation_entropy,
)
from src.preprocessing.tile_permutations import TilePermutation, identity_tile_permutation


def test_identity_tile_permutation_has_zero_hardness_metrics():
    tiles_per_side = 3
    tile_permutation = identity_tile_permutation(tiles_per_side)

    assert compute_global_displacement(tile_permutation, tiles_per_side) == 0.0
    assert compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side) == 0.0
    assert compute_spatial_permutation_entropy(tile_permutation, tiles_per_side) == 0.0
    assert compute_combined_hardness(
        adjacency_destruction_hardness=0.0,
        spatial_permutation_entropy=0.0,
        global_tile_displacement=0.0,
    ) == 0.0


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
    assert 0.0 < compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side) <= 1.0
    assert 0.0 < compute_spatial_permutation_entropy(tile_permutation, tiles_per_side) <= 1.0
    assert 0.0 < compute_combined_hardness(
        adjacency_destruction_hardness=compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side),
        spatial_permutation_entropy=compute_spatial_permutation_entropy(tile_permutation, tiles_per_side),
        global_tile_displacement=compute_global_displacement(tile_permutation, tiles_per_side),
    ) <= 1.0
