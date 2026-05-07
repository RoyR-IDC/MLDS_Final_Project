"""Tests for experiment result output helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping

import numpy as np

from src.evaluation.experiment_results import save_rows


class ArrayKeyRow(Mapping):
    """Minimal mapping that can expose an unhashable ndarray as a column key."""

    def __init__(self):
        self._items = [
            ("model_name", "resnet18"),
            (np.array([1, 2]), np.array([0.1, 0.2])),
            ("val_accuracy", np.float64(0.75)),
        ]

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        for item_key, value in self._items:
            if item_key is key or item_key == key:
                return value
        raise KeyError(key)

    def items(self):
        return iter(self._items)


def test_save_rows_handles_array_like_keys_and_values(tmp_path):
    rows = [ArrayKeyRow()]
    output_path = tmp_path / "raw_results.csv"

    save_rows(rows, str(output_path))

    with output_path.open(encoding="utf-8", newline="") as handle:
        saved_rows = list(csv.DictReader(handle))

    assert saved_rows == [
        {
            "model_name": "resnet18",
            "[1 2]": "[0.1, 0.2]",
            "val_accuracy": "0.75",
        }
    ]
