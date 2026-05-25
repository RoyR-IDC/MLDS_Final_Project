import pytest

from src.preprocessing.tile_permutations import (
    TILE_PERMUTATION_NAMES,
    TilePermutation,
    build_tile_permutation_records,
    deterministic_tile_permutation,
    generate_tile_permutations,
    identity_tile_permutation,
    matrix_to_flat_order,
    random_tile_permutation,
)


def test_generate_tile_permutations_length():
    tile_permutations = generate_tile_permutations(3, 5, seed=123)

    assert len(tile_permutations) == 5
    assert all(tile_permutation.tiles_per_side == 3 for tile_permutation in tile_permutations)
    assert all(len(tile_permutation.order) == 3 for tile_permutation in tile_permutations)


def test_identity_tile_permutation():
    tile_permutation = identity_tile_permutation(4)

    assert tile_permutation.order[0] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert matrix_to_flat_order(tile_permutation) == list(range(16))


def test_random_tile_permutation_contains_each_source_once():
    tile_permutation = random_tile_permutation(3, seed=123)

    assert sorted(matrix_to_flat_order(tile_permutation)) == list(range(9))


def test_tile_permutation_rejects_bad_shape():
    with pytest.raises(ValueError, match="rows"):
        TilePermutation(tiles_per_side=2, order=[[(0, 0), (0, 1)]])


def test_tile_permutation_rejects_duplicate_source():
    with pytest.raises(ValueError, match="appears more than once"):
        TilePermutation(tiles_per_side=2, order=[[(0, 0), (0, 0)], [(1, 0), (1, 1)]])


def test_tile_permutation_rejects_out_of_range_coordinate():
    with pytest.raises(ValueError, match="within"):
        TilePermutation(tiles_per_side=2, order=[[(0, 0), (0, 1)], [(1, 0), (2, 0)]])


def test_build_tile_permutation_records_uses_named_deterministic_matrix():
    records = build_tile_permutation_records([1, 4, 7, 10], num_tile_permutations=3, seed=123)

    assert {record.tile_permutation_seed for record in records} == {123}
    assert len(records) == 12
    assert [(record.tile_permutation_id, record.tile_permutation_name) for record in records[:3]] == [
        (1, "easy"),
        (2, "medium"),
        (3, "large"),
    ]
    assert all(record.tiles_per_side is None for record in records[:3])
    assert all(record.tile_permutation is None for record in records[:3])
    assert {record.tiles_per_side for record in records[3:]} == {4, 7, 10}
    assert all(record.tile_permutation_name in TILE_PERMUTATION_NAMES for record in records)
    assert all(record.tile_permutation is not None for record in records[3:])


def test_deterministic_tile_permutations_are_valid_and_repeatable():
    for tiles_per_side in (4, 7, 10):
        for permutation_name in TILE_PERMUTATION_NAMES:
            first = deterministic_tile_permutation(tiles_per_side, permutation_name)
            second = deterministic_tile_permutation(tiles_per_side, permutation_name)

            assert matrix_to_flat_order(first) == matrix_to_flat_order(second)
            assert sorted(matrix_to_flat_order(first)) == list(range(tiles_per_side * tiles_per_side))
            assert matrix_to_flat_order(first) != list(range(tiles_per_side * tiles_per_side))
