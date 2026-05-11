"""Tests for file I/O helpers."""

from __future__ import annotations

import csv

from src.utils.io import save_csv


def test_save_csv_writes_mapping_rows_without_pandas_constructor(tmp_path):
    rows = [
        {"grid_side_length": 1, "tile_order_id": 0, "tile_order_seed": 42, "output_tile_order": "[0]"},
        {"grid_side_length": 2, "tile_order_id": 1, "tile_order_seed": 42, "output_tile_order": "[2, 1, 3, 0]"},
    ]
    output_path = tmp_path / "output_tile_orders.csv"

    save_csv(rows, str(output_path))

    with output_path.open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))

    assert saved_rows == [
        {"grid_side_length": "1", "tile_order_id": "0", "tile_order_seed": "42", "output_tile_order": "[0]"},
        {"grid_side_length": "2", "tile_order_id": "1", "tile_order_seed": "42", "output_tile_order": "[2, 1, 3, 0]"},
    ]
