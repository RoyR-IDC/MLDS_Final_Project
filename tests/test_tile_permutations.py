import pytest

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_combined_hardness,
    compute_global_displacement,
    compute_spatial_permutation_entropy,
)
from src.preprocessing.tile_permutations import (
    ENHANCED_TILE_PERMUTATION_IDS,
    ENHANCED_TILE_PERMUTATION_NAMES,
    TILE_PERMUTATION_NAMES,
    TilePermutation,
    build_enhanced_tile_permutation_records,
    build_tile_permutation_records,
    build_difficulty_tile_permutation,
    deterministic_enhanced_tile_permutation,
    deterministic_tile_permutation,
    identity_tile_permutation,
    matrix_to_flat_order,
)


def test_identity_tile_permutation():
    tile_permutation = identity_tile_permutation(4)

    assert tile_permutation.order[0] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert matrix_to_flat_order(tile_permutation) == list(range(16))


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
        (3, "hard"),
    ]
    assert all(record.tiles_per_side is None for record in records[:3])
    assert all(record.tile_permutation is None for record in records[:3])
    assert {record.tiles_per_side for record in records[3:]} == {4, 7, 10}
    assert all(record.tile_permutation_name in TILE_PERMUTATION_NAMES for record in records)
    assert all(record.tile_permutation is not None for record in records[3:])


def test_build_enhanced_tile_permutation_records_uses_single_baseline():
    records = build_enhanced_tile_permutation_records([1, 4, 7, 10, 14, 17], seed=42)

    baseline_records = [record for record in records if record.tiles_per_side is None]
    permuted_records = [record for record in records if record.tiles_per_side is not None]

    assert len(records) == 31
    assert [(record.tile_permutation_id, record.tile_permutation_name) for record in baseline_records] == [
        (0, "baseline")
    ]
    assert {record.tiles_per_side for record in permuted_records} == {4, 7, 10, 14, 17}
    assert {record.tile_permutation_name for record in permuted_records} == set(ENHANCED_TILE_PERMUTATION_NAMES)
    assert all(
        record.tile_permutation_id == ENHANCED_TILE_PERMUTATION_IDS[record.tile_permutation_name]
        for record in permuted_records
    )


def test_enhanced_base_permutations_preserve_existing_matrices():
    for tiles_per_side in (4, 7, 10, 14, 17):
        for permutation_name in TILE_PERMUTATION_NAMES:
            existing = deterministic_tile_permutation(tiles_per_side, permutation_name)
            enhanced = deterministic_enhanced_tile_permutation(tiles_per_side, permutation_name, seed=42)

            assert matrix_to_flat_order(enhanced) == matrix_to_flat_order(existing)


def test_enhanced_variant_permutations_are_repeatable_and_non_identity():
    for tiles_per_side in (4, 7, 10, 14, 17):
        identity = list(range(tiles_per_side * tiles_per_side))
        for permutation_name in ("easy2", "medium2", "hard2"):
            first = deterministic_enhanced_tile_permutation(tiles_per_side, permutation_name, seed=42)
            second = deterministic_enhanced_tile_permutation(tiles_per_side, permutation_name, seed=42)

            assert matrix_to_flat_order(first) == matrix_to_flat_order(second)
            assert matrix_to_flat_order(first) != identity


def test_enhanced_variant_three_names_are_not_scheduled():
    records = build_enhanced_tile_permutation_records([1, 4], seed=42)

    assert {record.tile_permutation_name for record in records} == {
        "baseline",
        "easy",
        "easy2",
        "medium",
        "medium2",
        "hard",
        "hard2",
    }
    with pytest.raises(ValueError, match="Unsupported enhanced tile permutation name"):
        deterministic_enhanced_tile_permutation(4, "easy3", seed=42)


def test_deterministic_tile_permutations_are_valid_and_repeatable():
    for tiles_per_side in (4, 7, 10):
        for permutation_name in TILE_PERMUTATION_NAMES:
            first = deterministic_tile_permutation(tiles_per_side, permutation_name)
            second = deterministic_tile_permutation(tiles_per_side, permutation_name)

            assert matrix_to_flat_order(first) == matrix_to_flat_order(second)
            assert sorted(matrix_to_flat_order(first)) == list(range(tiles_per_side * tiles_per_side))
            assert matrix_to_flat_order(first) != list(range(tiles_per_side * tiles_per_side))


def _combined_hardness(tile_permutation):
    return compute_combined_hardness(
        adjacency_destruction_hardness=compute_adjacency_destruction_hardness(tile_permutation, None),
        spatial_permutation_entropy=compute_spatial_permutation_entropy(tile_permutation, None),
        global_tile_displacement=compute_global_displacement(tile_permutation, None),
    )


def test_difficulty_hardness_increases_with_label_for_each_grid():
    for tiles_per_side in (4, 7, 10):
        scores = [
            _combined_hardness(deterministic_tile_permutation(tiles_per_side, permutation_name))
            for permutation_name in TILE_PERMUTATION_NAMES
        ]

        assert scores == sorted(scores)
        assert scores[0] < scores[-1]


def test_difficulty_swap_count_scales_with_tile_count():
    easy_4 = deterministic_tile_permutation(4, "easy")
    easy_10 = deterministic_tile_permutation(10, "easy")

    changed_4 = sum(index != value for index, value in enumerate(matrix_to_flat_order(easy_4)))
    changed_10 = sum(index != value for index, value in enumerate(matrix_to_flat_order(easy_10)))

    assert changed_4 > 0
    assert changed_10 > changed_4


def test_build_difficulty_tile_permutation_rejects_invalid_fractions():
    with pytest.raises(ValueError, match="swap_fraction"):
        build_difficulty_tile_permutation(
            4,
            swap_fraction=1.2,
            max_swap_distance_fraction=0.1,
            row_shift_fraction=0.1,
            col_shift_fraction=0.1,
        )
