"""Tests for final results overview helpers."""

from __future__ import annotations

import pandas as pd

from src.utils.results_overview import _deduplicate_part3_baseline_rows


def test_deduplicate_part3_baseline_rows_keeps_one_baseline_and_all_permutations():
    tile_permutations = pd.DataFrame(
        [
            {"tiles_per_side": None, "tile_permutation_id": 1, "tile_permutation": None},
            {"tiles_per_side": None, "tile_permutation_id": 2, "tile_permutation": None},
            {"tiles_per_side": None, "tile_permutation_id": 3, "tile_permutation": None},
            {"tiles_per_side": 4, "tile_permutation_id": 1, "tile_permutation": "[1, 0, 2, 3]"},
            {"tiles_per_side": 4, "tile_permutation_id": 2, "tile_permutation": "[3, 2, 1, 0]"},
        ]
    )

    deduplicated = _deduplicate_part3_baseline_rows(tile_permutations)

    assert deduplicated["tiles_per_side"].isna().sum() == 1
    assert list(deduplicated["tile_permutation_id"]) == [1, 1, 2]
    assert list(deduplicated["tiles_per_side"].dropna()) == [4, 4]
