"""Tests for experiment result output helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation.experiment_results import (
    aggregate_accuracy,
    compute_part3_permutation_metrics,
    load_part1_model_baseline_raw_rows,
    load_part1_resnet50_results,
    save_rows,
)


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


def test_aggregate_accuracy_averages_permutations_by_tile_count():
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet50",
                "grid_size": 2,
                "num_tiles": 4,
                "permutation_id": 0,
                "val_accuracy": 0.50,
                "best_val_accuracy": 0.60,
            },
            {
                "model_name": "resnet50",
                "grid_size": 2,
                "num_tiles": 4,
                "permutation_id": 1,
                "val_accuracy": 0.70,
                "best_val_accuracy": 0.80,
            },
        ]
    )

    aggregated = aggregate_accuracy(raw_results, group_columns=["model_name", "grid_size", "num_tiles"])

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["model_name"] == "resnet50"
    assert row["num_tiles"] == 4
    assert row["mean_val_accuracy"] == 0.60
    assert row["mean_best_val_accuracy"] == 0.70
    assert row["n_runs"] == 2


def test_part1_baseline_raw_rows_are_retagged_for_part2(tmp_path):
    pd.DataFrame(
        [
            {
                "part": "part1",
                "run_id": "part1_run",
                "config_name": "part1",
                "model_name": "resnet50",
                "grid_size": 1,
                "num_tiles": 1,
                "permutation_id": 0,
                "val_accuracy": 0.75,
                "best_val_accuracy": 0.80,
            },
            {
                "part": "part1",
                "run_id": "part1_run",
                "config_name": "part1",
                "model_name": "deit_small",
                "grid_size": 1,
                "num_tiles": 1,
                "permutation_id": 0,
                "val_accuracy": 0.70,
                "best_val_accuracy": 0.72,
            },
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)
    config = SimpleNamespace(results_dir=str(tmp_path), part="part2", config_name="part2_improvement")

    rows = load_part1_model_baseline_raw_rows(config, "resnet50")

    assert len(rows) == 1
    assert rows[0]["part"] == "part2"
    assert rows[0]["config_name"] == "part2_improvement"
    assert rows[0]["ablation_name"] == "regular_part1"
    assert rows[0]["model_name"] == "resnet50"


def test_aggregate_accuracy_keeps_ablation_groups_separate():
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet50",
                "ablation_name": "augmentation_only",
                "grid_size": 1,
                "num_tiles": 1,
                "val_accuracy": 0.60,
                "best_val_accuracy": 0.65,
            },
            {
                "model_name": "resnet50",
                "ablation_name": "pretrained_finetune",
                "grid_size": 1,
                "num_tiles": 1,
                "val_accuracy": 0.80,
                "best_val_accuracy": 0.85,
            },
        ]
    )

    aggregated = aggregate_accuracy(raw_results, group_columns=["model_name", "ablation_name", "grid_size", "num_tiles"])

    assert set(aggregated["ablation_name"]) == {"augmentation_only", "pretrained_finetune"}
    assert len(aggregated) == 2


def test_part3_helpers_reuse_permutation_csv_filter_resnet50_and_emit_renamed_metrics(tmp_path):
    pd.DataFrame(
        [
            {
                "grid_size": 2,
                "permutation_id": 7,
                "permutation_seed": 99,
                "permutation": "[3, 2, 1, 0]",
            }
        ]
    ).to_csv(tmp_path / "part1_permutations.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "resnet50",
                "grid_size": 2,
                "num_tiles": 4,
                "permutation_id": 7,
                "best_val_accuracy": 0.75,
            },
            {
                "model_name": "deit_small",
                "grid_size": 2,
                "num_tiles": 4,
                "permutation_id": 7,
                "best_val_accuracy": 0.80,
            },
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)

    metrics = compute_part3_permutation_metrics(
        permutation_csv=str(tmp_path / "part1_permutations.csv"),
        grid_sizes=[1],
        num_permutations=0,
        seed=42,
    )
    resnet50_results = load_part1_resnet50_results(str(tmp_path / "part1_raw_results.csv"))

    assert metrics["permutation_id"].tolist() == [7]
    assert set(
        [
            "global_tile_displacement",
            "center_weighted_displacement",
            "adjacency_preservation_loss",
            "combined_hardness_score",
        ]
    ).issubset(metrics.columns)
    assert resnet50_results["model_name"].tolist() == ["resnet50"]
