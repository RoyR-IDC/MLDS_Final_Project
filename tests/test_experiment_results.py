"""Tests for experiment result output helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.evaluation.experiment_results import (
    aggregate_accuracy,
    compute_part3_metric_correlations,
    compute_part3_tile_permutation_metrics,
    load_part1_model_baseline_aggregated,
    load_part1_model_baseline_raw_rows,
    load_part1_model_results,
    part3_output_paths,
    plot_accuracy_vs_tiles,
    plot_part3_metrics_vs_accuracy,
    run_part3_hardness_analysis,
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


def _validation_samples(tmp_path):
    first = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    second = np.flipud(first)
    first_path = tmp_path / "cat.0.jpg"
    second_path = tmp_path / "dog.0.jpg"
    Image.fromarray(first).save(first_path)
    Image.fromarray(second).save(second_path)
    return [(str(first_path), 0), (str(second_path), 1)]


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


def test_aggregate_accuracy_averages_tile_permutations_by_num_tiles():
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 0,
                "val_accuracy": 0.50,
                "best_val_accuracy": 0.60,
            },
            {
                "model_name": "resnet18",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 1,
                "val_accuracy": 0.70,
                "best_val_accuracy": 0.80,
            },
        ]
    )

    aggregated = aggregate_accuracy(raw_results, group_columns=["model_name", "tiles_per_side", "num_tiles"])

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["model_name"] == "resnet18"
    assert row["num_tiles"] == 4
    assert row["mean_final_epoch_val_accuracy"] == 0.60
    assert row["mean_best_epoch_val_accuracy"] == 0.70
    assert row["n_runs"] == 2


def test_plot_accuracy_vs_tiles_uses_best_epoch_aggregate_column(tmp_path):
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_final_epoch_val_accuracy": 0.40,
                "std_final_epoch_val_accuracy": 0.01,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.02,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 9,
                "mean_final_epoch_val_accuracy": 0.45,
                "std_final_epoch_val_accuracy": 0.01,
                "mean_best_epoch_val_accuracy": 0.65,
                "std_best_epoch_val_accuracy": 0.03,
            },
        ]
    )
    output_path = tmp_path / "accuracy_vs_tiles.png"

    plot_accuracy_vs_tiles(aggregated, str(output_path))

    assert output_path.exists()


def test_part1_baseline_raw_rows_are_retagged_for_part2(tmp_path):
    pd.DataFrame(
        [
            {
                "part": "part1",
                "run_id": "part1_run",
                "config_name": "part1",
                "model_name": "resnet18",
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 0,
                "tile_permutation": None,
                "val_accuracy": 0.75,
                "best_val_accuracy": 0.80,
            },
            {
                "part": "part1",
                "run_id": "part1_run",
                "config_name": "part1",
                "model_name": "deit_tiny",
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 0,
                "tile_permutation": None,
                "val_accuracy": 0.70,
                "best_val_accuracy": 0.72,
            },
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)
    config = SimpleNamespace(results_dir=str(tmp_path), part="part2", config_name="part2_improvement")

    rows = load_part1_model_baseline_raw_rows(config, "resnet18")

    assert len(rows) == 1
    assert rows[0]["part"] == "part2"
    assert rows[0]["config_name"] == "part2_improvement"
    assert rows[0]["ablation_name"] == "regular_part1"
    assert rows[0]["model_name"] == "resnet18"


def test_part1_baseline_aggregated_rejects_unsupported_accuracy_column_names(tmp_path):
    pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "tiles_per_side": None,
                "num_tiles": 1,
                "mean_val_accuracy": 0.75,
                "std_val_accuracy": 0.0,
                "mean_best_val_accuracy": 0.80,
                "std_best_val_accuracy": 0.0,
                "n_runs": 1,
            },
        ]
    ).to_csv(tmp_path / "part1_aggregated_results.csv", index=False)
    config = SimpleNamespace(results_dir=str(tmp_path), part="part2", config_name="part2_improvement")

    with pytest.raises(ValueError, match="unsupported schema"):
        load_part1_model_baseline_aggregated(config, "resnet18")


def test_aggregate_accuracy_keeps_ablation_groups_separate():
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "augmentation_only",
                "tiles_per_side": None,
                "num_tiles": 1,
                "val_accuracy": 0.60,
                "best_val_accuracy": 0.65,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "finetune_only",
                "tiles_per_side": None,
                "num_tiles": 1,
                "val_accuracy": 0.80,
                "best_val_accuracy": 0.85,
            },
        ]
    )

    aggregated = aggregate_accuracy(raw_results, group_columns=["model_name", "ablation_name", "tiles_per_side", "num_tiles"])

    assert set(aggregated["ablation_name"]) == {"augmentation_only", "finetune_only"}
    assert len(aggregated) == 2


def test_part3_helpers_reuse_tile_permutation_csv_filter_model_and_emit_renamed_metrics(tmp_path):
    pd.DataFrame(
        [
            {
                "tiles_per_side": 2,
                "tile_permutation_id": 7,
                "tile_permutation_seed": 99,
                "tile_permutation": "[[[1, 1], [1, 0]], [[0, 1], [0, 0]]]",
            }
        ]
    ).to_csv(tmp_path / "part1_tile_permutations.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 7,
                "best_val_accuracy": 0.75,
            },
            {
                "model_name": "deit_tiny",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 7,
                "best_val_accuracy": 0.80,
            },
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)

    metrics = compute_part3_tile_permutation_metrics(
        tile_permutation_csv=str(tmp_path / "part1_tile_permutations.csv"),
        tiles_per_side_values=[1],
        num_tile_permutations=0,
        seed=42,
        validation_samples=_validation_samples(tmp_path),
        image_size=4,
    )
    model_results = load_part1_model_results(str(tmp_path / "part1_raw_results.csv"), "resnet18")

    assert metrics["tile_permutation_id"].tolist() == [7]
    assert set(
        [
            "global_tile_displacement",
            "adjacency_destruction_hardness",
            "spatial_permutation_entropy",
            "combined_hardness_score",
        ]
    ).issubset(metrics.columns)
    assert "edge_continuity_disruption_raw" not in metrics.columns
    assert model_results["model_name"].tolist() == ["resnet18"]

def test_part3_metrics_include_none_baseline_and_tiled_permutations(tmp_path):
    pd.DataFrame(
        [
            {"tiles_per_side": None, "tile_permutation_id": 0, "tile_permutation_seed": 42, "tile_permutation": "null"},
            {
                "tiles_per_side": 2,
                "tile_permutation_id": 1,
                "tile_permutation_seed": 42,
                "tile_permutation": "[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]",
            },
        ]
    ).to_csv(tmp_path / "part1_tile_permutations.csv", index=False)

    metrics = compute_part3_tile_permutation_metrics(
        tile_permutation_csv=str(tmp_path / "part1_tile_permutations.csv"),
        tiles_per_side_values=[1, 2],
        num_tile_permutations=1,
        seed=42,
        validation_samples=[],
        image_size=4,
    )

    assert pd.isna(metrics.loc[0, "tiles_per_side"])
    assert metrics.loc[0, "tile_permutation_id"] == 0
    assert metrics.loc[1, ["tiles_per_side", "tile_permutation_id"]].to_dict() == {
        "tiles_per_side": 2,
        "tile_permutation_id": 1,
    }


def test_part3_non_identity_2x2_tile_permutation_has_nonzero_metric(tmp_path):
    pd.DataFrame(
        [
            {
                "tiles_per_side": 2,
                "tile_permutation_id": 1,
                "tile_permutation_seed": 42,
                "tile_permutation": "[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]",
            }
        ]
    ).to_csv(tmp_path / "part1_tile_permutations.csv", index=False)

    metrics = compute_part3_tile_permutation_metrics(
        tile_permutation_csv=str(tmp_path / "part1_tile_permutations.csv"),
        tiles_per_side_values=[2],
        num_tile_permutations=1,
        seed=42,
        validation_samples=_validation_samples(tmp_path),
        image_size=4,
    )

    metric_columns = [
        "global_tile_displacement",
        "adjacency_destruction_hardness",
        "spatial_permutation_entropy",
        "combined_hardness_score",
    ]
    assert metrics.loc[0, metric_columns].gt(0.0).any()
    assert 0.0 <= metrics.loc[0, "spatial_permutation_entropy"] <= 1.0


def test_part3_spatial_permutation_entropy_has_no_raw_edge_column(tmp_path):
    pd.DataFrame(
        [
            {
                "tiles_per_side": 2,
                "tile_permutation_id": 1,
                "tile_permutation_seed": 42,
                "tile_permutation": "[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]",
            }
        ]
    ).to_csv(tmp_path / "part1_tile_permutations.csv", index=False)

    metrics = compute_part3_tile_permutation_metrics(
        tile_permutation_csv=str(tmp_path / "part1_tile_permutations.csv"),
        tiles_per_side_values=[2],
        num_tile_permutations=1,
        seed=42,
        validation_samples=_validation_samples(tmp_path),
        image_size=4,
    )

    assert "edge_continuity_disruption_raw" not in metrics.columns
    assert "spatial_permutation_entropy" in metrics.columns


def test_part3_hardness_analysis_raises_for_all_zero_tiled_non_identity_metrics(tmp_path):
    pd.DataFrame(
        [
            {
                "tiles_per_side": 2,
                "tile_permutation_id": 1,
                "tile_permutation_seed": 42,
                "tile_permutation": "[[[0, 0], [0, 1]], [[1, 0], [1, 1]]]",
            }
        ]
    ).to_csv(tmp_path / "part1_tile_permutations.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.75,
            }
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)

    with pytest.raises(ValueError, match="metrics are all zero"):
        run_part3_hardness_analysis(
            results_dir=str(tmp_path),
            figures_dir=str(tmp_path),
            part1_results_csv=str(tmp_path / "part1_raw_results.csv"),
            tile_permutation_csv=str(tmp_path / "part1_tile_permutations.csv"),
            tiles_per_side_values=[2],
            num_tile_permutations=1,
            seed=42,
            validation_samples=_validation_samples(tmp_path),
            image_size=4,
            model_name="resnet18",
            verbose=False,
            show_progress=False,
        )


def test_part3_combined_plot_is_reported_by_output_paths(tmp_path):
    joined = pd.DataFrame(
        [
            {
                "best_val_accuracy": 0.70,
                "global_tile_displacement": 0.10,
                "adjacency_destruction_hardness": 0.30,
                "spatial_permutation_entropy": 0.20,
                "combined_hardness_score": 0.25,
            },
            {
                "best_val_accuracy": 0.60,
                "global_tile_displacement": 0.80,
                "adjacency_destruction_hardness": 0.90,
                "spatial_permutation_entropy": 0.70,
                "combined_hardness_score": 0.70,
            },
        ]
    )

    plot_part3_metrics_vs_accuracy(joined, str(tmp_path))
    paths = part3_output_paths(str(tmp_path), str(tmp_path))

    assert paths["plots"] == [str(tmp_path / "part3_metrics_vs_accuracy.png")]
    assert (tmp_path / "part3_metrics_vs_accuracy.png").exists()


def test_part3_correlations_are_nan_for_constant_accuracy():
    joined = pd.DataFrame(
        [
            {
                "best_val_accuracy": 0.50,
                "global_tile_displacement": 0.00,
                "adjacency_destruction_hardness": 0.00,
                "spatial_permutation_entropy": 0.00,
                "combined_hardness_score": 0.00,
            },
            {
                "best_val_accuracy": 0.50,
                "global_tile_displacement": 0.50,
                "adjacency_destruction_hardness": 0.60,
                "spatial_permutation_entropy": 0.40,
                "combined_hardness_score": 0.45,
            },
        ]
    )

    correlations = compute_part3_metric_correlations(joined)

    assert correlations["pearson"].isna().all()
    assert correlations["spearman"].isna().all()


def test_part3_correlations_are_finite_for_non_constant_accuracy():
    joined = pd.DataFrame(
        [
            {
                "best_val_accuracy": 0.80,
                "global_tile_displacement": 0.00,
                "adjacency_destruction_hardness": 0.00,
                "spatial_permutation_entropy": 0.10,
                "combined_hardness_score": 0.15,
            },
            {
                "best_val_accuracy": 0.70,
                "global_tile_displacement": 0.50,
                "adjacency_destruction_hardness": 0.60,
                "spatial_permutation_entropy": 0.40,
                "combined_hardness_score": 0.45,
            },
            {
                "best_val_accuracy": 0.60,
                "global_tile_displacement": 0.90,
                "adjacency_destruction_hardness": 1.00,
                "spatial_permutation_entropy": 0.80,
                "combined_hardness_score": 0.80,
            },
        ]
    )

    correlations = compute_part3_metric_correlations(joined)

    assert np.isfinite(correlations["pearson"]).all()
    assert np.isfinite(correlations["spearman"]).all()
