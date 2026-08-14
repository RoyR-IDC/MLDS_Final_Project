"""Tests for file I/O helpers."""

from __future__ import annotations

import csv

from src.utils.io import save_csv


def test_save_csv_writes_mapping_rows_without_pandas_constructor(tmp_path):
    rows = [
        {"tiles_per_side": None, "tile_permutation_id": 0, "tile_permutation_seed": 42, "tile_permutation": "null"},
        {
            "tiles_per_side": 2,
            "tile_permutation_id": 1,
            "tile_permutation_seed": 42,
            "tile_permutation": "[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]",
        },
    ]
    output_path = tmp_path / "tile_permutations.csv"

    save_csv(rows, str(output_path))

    with output_path.open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))

    assert saved_rows == [
        {"tiles_per_side": "", "tile_permutation_id": "0", "tile_permutation_seed": "42", "tile_permutation": "null"},
        {
            "tiles_per_side": "2",
            "tile_permutation_id": "1",
            "tile_permutation_seed": "42",
            "tile_permutation": "[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]",
        },
    ]
