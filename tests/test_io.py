"""Tests for file I/O helpers."""

from __future__ import annotations

import csv

from src.utils.io import save_csv


def test_save_csv_writes_mapping_rows_without_pandas_constructor(tmp_path):
    rows = [
        {"grid_size": 1, "permutation_id": 0, "permutation_seed": 42, "permutation": "[0]"},
        {"grid_size": 2, "permutation_id": 1, "permutation_seed": 42, "permutation": "[2, 1, 3, 0]"},
    ]
    output_path = tmp_path / "permutations.csv"

    save_csv(rows, str(output_path))

    with output_path.open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))

    assert saved_rows == [
        {"grid_size": "1", "permutation_id": "0", "permutation_seed": "42", "permutation": "[0]"},
        {"grid_size": "2", "permutation_id": "1", "permutation_seed": "42", "permutation": "[2, 1, 3, 0]"},
    ]
