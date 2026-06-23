"""Tests for experiment result output helpers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from src.evaluation.experiment_results import (
    PERMUTATION_MARKERS,
    add_part2_grid_baseline_deltas,
    aggregate_accuracy,
    build_part3_correlation_input,
    compute_part3_metric_correlations,
    compute_part3_tile_permutation_metrics,
    experiment_intermediate_figure_path,
    load_part1_model_baseline_aggregated,
    load_part1_model_baseline_raw_rows,
    load_part1_model_results,
    part3_output_paths,
    plot_ablation_results,
    plot_accuracy_vs_tiles,
    plot_part3_metric_grid_views,
    refresh_part2_ablation_comparison_figure,
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


def test_plot_accuracy_vs_tiles_archives_existing_figure_before_saving(tmp_path):
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            }
        ]
    )
    output_path = tmp_path / "accuracy_vs_tiles.png"
    output_path.write_text("old figure bytes", encoding="utf-8")

    plot_accuracy_vs_tiles(aggregated, str(output_path))

    archived = list(tmp_path.glob("accuracy_vs_tiles_*.png"))
    assert output_path.exists()
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == "old figure bytes"


def test_plot_accuracy_vs_tiles_uses_mean_lines_without_error_bars(monkeypatch, tmp_path):
    plot_calls = []

    def fail_errorbar(self, *args, **kwargs):
        raise AssertionError("accuracy plots should not use error bars")

    def fake_plot(self, *args, **kwargs):
        plot_calls.append((args, kwargs))
        return []

    monkeypatch.setattr("matplotlib.axes.Axes.errorbar", fail_errorbar)
    monkeypatch.setattr("matplotlib.axes.Axes.plot", fake_plot)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", lambda self, *args, **kwargs: None)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.02,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.65,
                "std_best_epoch_val_accuracy": 0.03,
            },
        ]
    )

    plot_accuracy_vs_tiles(aggregated, str(tmp_path / "accuracy.png"))

    assert plot_calls
    assert list(plot_calls[0][0][1]) == [0.75, 0.65]


def test_plot_accuracy_vs_tiles_title_and_ylim_for_intermediate_model(monkeypatch, tmp_path):
    titles = []
    y_limits = []

    def fake_set_title(self, title, *args, **kwargs):
        titles.append(title)

    def fake_set_ylim(self, *args, **kwargs):
        y_limits.append(args)

    monkeypatch.setattr("matplotlib.axes.Axes.set_title", fake_set_title)
    monkeypatch.setattr("matplotlib.axes.Axes.set_ylim", fake_set_ylim)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            }
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "tile_permutation_name": None,
                "best_val_accuracy": 0.75,
            }
        ]
    )

    plot_accuracy_vs_tiles(
        aggregated,
        str(tmp_path / "accuracy.png"),
        raw_results=raw_results,
        title="Intermediate Model Plot: Validation Accuracy by Tiling Level - resnet18",
    )

    assert titles[-1] == "Intermediate Model Plot: Validation Accuracy by Tiling Level - resnet18"
    assert (0.50, 1.00) in y_limits


def test_plot_accuracy_vs_tiles_overlays_condition_markers_in_intermediate_plot(monkeypatch, tmp_path):
    markers = []
    plot_kwargs = []

    def fake_scatter(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))

    def fake_plot(self, *args, **kwargs):
        plot_kwargs.append(kwargs)
        return []

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.plot", fake_plot)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.74,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
                "std_best_epoch_val_accuracy": 0.01,
            }
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "tiles_per_side": 4,
                "tile_permutation_name": "easy",
                "tile_permutation": "[[0, 1], [2, 3]]",
                "best_val_accuracy": 0.72,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "tiles_per_side": 4,
                "tile_permutation_name": "medium",
                "tile_permutation": "[[1, 0], [2, 3]]",
                "best_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "tiles_per_side": 4,
                "tile_permutation_name": "hard",
                "tile_permutation": "[[3, 2], [1, 0]]",
                "best_val_accuracy": 0.68,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "tiles_per_side": None,
                "tile_permutation_name": "easy",
                "tile_permutation": None,
                "best_val_accuracy": 0.74,
            },
        ]
    )

    plot_accuracy_vs_tiles(
        aggregated,
        str(tmp_path / "accuracy.png"),
        raw_results=raw_results,
        title="Intermediate Model Plot: Validation Accuracy by Tiling Level - resnet18",
    )

    assert PERMUTATION_MARKERS["easy"] in markers
    assert PERMUTATION_MARKERS["medium"] in markers
    assert PERMUTATION_MARKERS["hard"] in markers
    assert PERMUTATION_MARKERS["baseline"] in markers
    assert PERMUTATION_MARKERS["baseline"] == "D"
    assert plot_kwargs[0]["label"] == "Mean across permutations"
    assert plot_kwargs[0]["marker"] is None


def test_plot_accuracy_vs_tiles_intermediate_legends_include_mean_and_conditions(monkeypatch, tmp_path):
    legend_labels = []
    legend_titles = []

    def fake_legend(self, handles=None, labels=None, *args, **kwargs):
        if labels is None and handles is not None:
            labels = [handle.get_label() for handle in handles]
        legend_labels.extend(labels or [])
        legend_titles.append(kwargs.get("title"))
        return None

    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            }
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "tile_permutation_name": None,
                "best_val_accuracy": 0.75,
            }
        ]
    )

    plot_accuracy_vs_tiles(
        aggregated,
        str(tmp_path / "accuracy.png"),
        raw_results=raw_results,
        title="Intermediate Model Plot: Validation Accuracy by Tiling Level - resnet18",
    )

    assert "Mean" in legend_titles
    assert "Condition" in legend_titles
    assert "Baseline: no permutation" in legend_labels


def test_plot_accuracy_vs_tiles_intermediate_experiment_condition_lines(monkeypatch, tmp_path):
    plot_labels = []
    legend_titles = []

    def fake_plot(self, *args, **kwargs):
        plot_labels.append(kwargs.get("label"))
        return []

    def fake_legend(self, handles=None, labels=None, *args, **kwargs):
        legend_titles.append(kwargs.get("title"))
        return None

    monkeypatch.setattr("matplotlib.axes.Axes.plot", fake_plot)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "mobilenetv3_small",
                "experiment_condition": "frozen_pretrained_binary_head",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.75,
            },
            {
                "model_name": "mobilenetv3_small",
                "experiment_condition": "unfrozen_pretrained_binary_head",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.78,
            },
            {
                "model_name": "mobilenetv3_small",
                "experiment_condition": "zero_shot_full_pretrained_head",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.65,
            },
        ]
    )

    plot_accuracy_vs_tiles(
        aggregated,
        str(tmp_path / "intermediate" / "part1_accuracy_vs_tiles_mobilenetv3_small_experiments.png"),
        model_column="experiment_condition",
        title="Intermediate Model Plot: Experiment Comparison by Tiling Level - mobilenetv3_small",
    )

    assert plot_labels == [
        "frozen_pretrained_binary_head",
        "unfrozen_pretrained_binary_head",
        "zero_shot_full_pretrained_head",
    ]
    assert "Experiment Condition" in legend_titles
    assert "Mean" not in legend_titles


def test_plot_accuracy_vs_tiles_final_plot_has_model_lines_and_hardness_condition_points(monkeypatch, tmp_path):
    plot_labels = []
    markers = []
    titles = []
    legend_labels = []
    legend_titles = []

    def fail_errorbar(self, *args, **kwargs):
        raise AssertionError("final accuracy plot should not use error bars")

    def fake_plot(self, *args, **kwargs):
        plot_labels.append(kwargs.get("label"))
        return []

    def fake_scatter(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))

    def fake_set_title(self, title, *args, **kwargs):
        titles.append(title)

    def fake_legend(self, handles=None, labels=None, *args, **kwargs):
        if labels is None and handles is not None:
            labels = [handle.get_label() for handle in handles]
        legend_labels.extend(labels or [])
        legend_titles.append(kwargs.get("title"))
        return None

    monkeypatch.setattr("matplotlib.axes.Axes.errorbar", fail_errorbar)
    monkeypatch.setattr("matplotlib.axes.Axes.plot", fake_plot)
    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.set_title", fake_set_title)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "deit_tiny",
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.65,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "tiles_per_side": 4,
                "tile_permutation_name": "easy",
                "tile_permutation": "[[0, 1], [2, 3]]",
                "best_val_accuracy": 0.72,
            },
            {
                "model_name": "deit_tiny",
                "num_tiles": 16,
                "tiles_per_side": 4,
                "tile_permutation_name": "hard",
                "tile_permutation": "[[3, 2], [1, 0]]",
                "best_val_accuracy": 0.63,
            },
        ]
    )

    plot_accuracy_vs_tiles(aggregated, str(tmp_path / "accuracy.png"), raw_results=raw_results)

    assert plot_labels == ["deit_tiny", "resnet18"]
    assert PERMUTATION_MARKERS["easy"] in markers
    assert PERMUTATION_MARKERS["hard"] in markers
    assert "Model Name" in legend_titles
    assert "Condition" in legend_titles
    assert "easy" in legend_labels
    assert "hard" in legend_labels
    assert titles[-1] == "Final Model Comparison: Mean Validation Accuracy by Tiling Level"


def test_plot_accuracy_vs_tiles_single_model_final_plot_keeps_final_title(monkeypatch, tmp_path):
    titles = []
    legend_titles = []

    def fake_set_title(self, title, *args, **kwargs):
        titles.append(title)

    def fake_legend(self, *args, **kwargs):
        legend_titles.append(kwargs.get("title"))
        return None

    monkeypatch.setattr("matplotlib.axes.Axes.set_title", fake_set_title)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
                "std_best_epoch_val_accuracy": 0.0,
            }
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "tiles_per_side": 4,
                "tile_permutation_name": "easy",
                "tile_permutation": "[[0, 1], [2, 3]]",
                "best_val_accuracy": 0.72,
            }
        ]
    )

    plot_accuracy_vs_tiles(aggregated, str(tmp_path / "accuracy.png"), raw_results=raw_results)

    assert titles[-1] == "Final Model Comparison: Mean Validation Accuracy by Tiling Level"
    assert "Model Name" in legend_titles
    assert "Condition" in legend_titles


def test_plot_accuracy_vs_tiles_uses_equal_categorical_tile_spacing(monkeypatch, tmp_path):
    tick_calls = []

    def fake_set_xticks(self, ticks, labels=None, *args, **kwargs):
        tick_calls.append((list(ticks), list(labels or [])))

    monkeypatch.setattr("matplotlib.axes.Axes.set_xticks", fake_set_xticks)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", lambda self, *args, **kwargs: None)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 49,
                "mean_best_epoch_val_accuracy": 0.68,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "num_tiles": 100,
                "mean_best_epoch_val_accuracy": 0.66,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )

    plot_accuracy_vs_tiles(aggregated, str(tmp_path / "accuracy.png"))

    assert tick_calls[-1] == ([0, 1, 2, 3], ["1", "4x4", "7x7", "10x10"])


def test_plot_ablation_results_uses_aggregate_points_without_error_bars(monkeypatch, tmp_path):
    markers = []

    def fail_errorbar(self, *args, **kwargs):
        raise AssertionError("ablation plots should not use error bars")

    def fake_scatter(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))

    monkeypatch.setattr("matplotlib.axes.Axes.errorbar", fail_errorbar)
    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", lambda self, *args, **kwargs: None)
    aggregated = pd.DataFrame(
        [
            {
                "ablation_name": "regular_part1",
                "tiles_per_side": 1,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "ablation_name": "resnet18_mlp_head",
                "tiles_per_side": 1,
                "mean_best_epoch_val_accuracy": 0.80,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )

    plot_ablation_results(aggregated, str(tmp_path / "ablation.png"))

    assert markers == ["o"]


def test_plot_ablation_results_title_accepts_model_and_ablation_override(monkeypatch, tmp_path):
    titles = []

    def fake_set_title(self, title, *args, **kwargs):
        titles.append(title)

    monkeypatch.setattr("matplotlib.axes.Axes.set_title", fake_set_title)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            }
        ]
    )

    plot_ablation_results(
        aggregated,
        str(tmp_path / "ablation.png"),
        title="resnet18 / patch_shuffle: Ablations vs Matched Grid Baseline",
    )

    assert titles[-1] == "resnet18 / patch_shuffle\nAblations vs Matched Grid Baseline"


def test_plot_ablation_results_default_title_wraps_to_two_lines(monkeypatch, tmp_path):
    titles = []

    def fake_set_title(self, title, *args, **kwargs):
        titles.append(title)

    monkeypatch.setattr("matplotlib.axes.Axes.set_title", fake_set_title)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )

    plot_ablation_results(aggregated, str(tmp_path / "ablation.png"))

    assert titles[-1] == "resnet18\nAblations vs Matched Grid Baseline"


def test_plot_ablation_results_uses_grid_axis_and_filters_baseline_legends(monkeypatch, tmp_path):
    tick_calls = []
    legend_labels = []
    legend_titles = []
    hlines = []

    def fake_set_xticks(self, ticks, labels=None, *args, **kwargs):
        tick_calls.append((list(ticks), list(labels or [])))

    def fake_legend(self, handles=None, labels=None, *args, **kwargs):
        if labels is None and handles is not None:
            labels = [handle.get_label() for handle in handles]
        legend_labels.extend(labels or [])
        legend_titles.append(kwargs.get("title"))
        return None

    def fake_axhline(self, y, *args, **kwargs):
        hlines.append(y)

    monkeypatch.setattr("matplotlib.axes.Axes.set_xticks", fake_set_xticks)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    monkeypatch.setattr("matplotlib.axes.Axes.axhline", fake_axhline)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": None,
                "num_tiles": 1,
                "mean_best_epoch_val_accuracy": 0.72,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 7,
                "num_tiles": 49,
                "mean_best_epoch_val_accuracy": 0.74,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 10,
                "num_tiles": 100,
                "mean_best_epoch_val_accuracy": 0.73,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 0,
                "tile_permutation_name": None,
                "best_val_accuracy": 0.72,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.75,
            },
        ]
    )

    plot_ablation_results(aggregated, str(tmp_path / "ablation.png"), raw_results=raw_results)

    assert tick_calls[-1] == ([0, 1, 2, 3], ["1", "4x4", "7x7", "10x10"])
    assert "Ablation" in legend_titles
    assert "Reference" in legend_titles
    assert "Condition" in legend_titles
    assert "patch_shuffle" in legend_labels
    assert "Matched Part 1 grid baseline reference" in legend_labels
    assert "regular_part1" not in legend_labels
    assert "Baseline: no permutation" not in legend_labels
    assert 0.0 in hlines


def test_plot_ablation_results_intermediate_mode_hides_aggregate_points(monkeypatch, tmp_path):
    markers = []
    legend_labels = []
    legend_titles = []

    def fake_scatter(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))

    def fake_legend(self, handles=None, labels=None, *args, **kwargs):
        if labels is None and handles is not None:
            labels = [handle.get_label() for handle in handles]
        legend_labels.extend(labels or [])
        legend_titles.append(kwargs.get("title"))
        return None

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
            },
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.75,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 2,
                "tile_permutation_name": "medium",
                "best_val_accuracy": 0.74,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 3,
                "tile_permutation_name": "hard",
                "best_val_accuracy": 0.73,
            },
        ]
    )

    plot_ablation_results(
        aggregated,
        str(tmp_path / "ablation.png"),
        raw_results=raw_results,
        show_raw_points=True,
        show_aggregate_points=False,
    )

    assert markers == [
        PERMUTATION_MARKERS["easy"],
        PERMUTATION_MARKERS["medium"],
        PERMUTATION_MARKERS["hard"],
    ]
    assert "Ablation" not in legend_titles
    assert "Condition" in legend_titles
    assert "Baseline: no permutation" not in legend_labels


def test_plot_ablation_results_final_mode_hides_raw_condition_points(monkeypatch, tmp_path):
    markers = []
    legend_labels = []
    legend_titles = []

    def fake_scatter(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))

    def fake_legend(self, handles=None, labels=None, *args, **kwargs):
        if labels is None and handles is not None:
            labels = [handle.get_label() for handle in handles]
        legend_labels.extend(labels or [])
        legend_titles.append(kwargs.get("title"))
        return None

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", fake_legend)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "regular_augmentations",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.76,
            },
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.75,
            }
        ]
    )

    plot_ablation_results(
        aggregated,
        str(tmp_path / "ablation.png"),
        raw_results=raw_results,
        show_raw_points=False,
        show_aggregate_points=True,
    )

    assert markers == ["o", "o"]
    assert "Ablation" in legend_titles
    assert "Condition" not in legend_titles
    assert "patch_shuffle" in legend_labels
    assert "regular_augmentations" in legend_labels
    assert "regular_part1" not in legend_labels


def test_plot_ablation_results_overlays_permutation_markers(monkeypatch, tmp_path):
    markers = []

    def fake_scatter(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter)
    monkeypatch.setattr("matplotlib.axes.Axes.legend", lambda self, *args, **kwargs: None)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
                "std_best_epoch_val_accuracy": 0.0,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.75,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 2,
                "tile_permutation_name": "hard",
                "best_val_accuracy": 0.73,
            },
        ]
    )

    plot_ablation_results(aggregated, str(tmp_path / "ablation.png"), raw_results=raw_results)

    assert PERMUTATION_MARKERS["easy"] in markers
    assert PERMUTATION_MARKERS["hard"] in markers


def test_plot_ablation_results_can_skip_archiving_existing_figure(monkeypatch, tmp_path):
    def fail_archive(_output_path):
        raise AssertionError("archive should not be called")

    monkeypatch.setattr("src.evaluation.experiment_results._archive_existing_figure", fail_archive)
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    )
    output_path = tmp_path / "ablation.png"
    output_path.write_bytes(b"old figure")

    plot_ablation_results(aggregated, str(output_path), archive_existing=False)

    assert output_path.exists()


def test_refresh_part2_ablation_comparison_copies_original_and_passes_raw_results(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    results_dir.mkdir()
    figures_dir.mkdir()
    aggregated_path = results_dir / "part2_aggregated_results.csv"
    raw_path = results_dir / "part2_raw_results.csv"
    figure_path = figures_dir / "part2_ablation_comparison.png"
    original_bytes = b"original figure bytes"
    figure_path.write_bytes(original_bytes)
    pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.75,
                "std_best_epoch_val_accuracy": 0.0,
            },
        ]
    ).to_csv(aggregated_path, index=False)
    pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "tile_permutation_name": "easy",
                "best_val_accuracy": 0.75,
            },
        ]
    ).to_csv(raw_path, index=False)
    captured = {}

    def fake_plot_ablation_results(
        aggregated,
        output_path,
        raw_results=None,
        title=None,
        show_raw_points=True,
        show_aggregate_points=True,
        archive_existing=True,
    ):
        del title
        captured["aggregated"] = aggregated
        captured["raw_results"] = raw_results
        captured["show_raw_points"] = show_raw_points
        captured["show_aggregate_points"] = show_aggregate_points
        captured["archive_existing"] = archive_existing
        Path(output_path).write_bytes(b"refreshed figure bytes")

    monkeypatch.setitem(
        refresh_part2_ablation_comparison_figure.__globals__,
        "plot_ablation_results",
        fake_plot_ablation_results,
    )

    paths = refresh_part2_ablation_comparison_figure(
        results_dir=str(results_dir),
        figures_dir=str(figures_dir),
    )

    original_path = figures_dir / "part2_ablation_comparison_original.png"
    assert original_path.read_bytes() == original_bytes
    assert figure_path.read_bytes() == b"refreshed figure bytes"
    assert paths["figure"] == str(figure_path)
    assert paths["original_figure"] == str(original_path)
    assert captured["raw_results"] is not None
    assert captured["raw_results"]["tile_permutation_name"].tolist() == ["easy"]
    assert captured["show_raw_points"] is True
    assert captured["show_aggregate_points"] is True
    assert captured["archive_existing"] is False


def test_part2_grid_baseline_deltas_match_tile_permutation_before_grid_fallback():
    aggregated = pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.60,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "regular_part1",
                "tiles_per_side": 7,
                "num_tiles": 49,
                "mean_best_epoch_val_accuracy": 0.80,
            },
            {
                "model_name": "resnet18",
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "mean_best_epoch_val_accuracy": 0.70,
            },
        ]
    )
    raw_results = pd.DataFrame(
        [
            {
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.50,
            },
            {
                "ablation_name": "regular_part1",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 2,
                "best_val_accuracy": 0.70,
            },
            {
                "ablation_name": "patch_shuffle",
                "tiles_per_side": 4,
                "num_tiles": 16,
                "tile_permutation_id": 2,
                "best_val_accuracy": 0.75,
            },
        ]
    )

    aggregated_delta, raw_delta = add_part2_grid_baseline_deltas(aggregated, raw_results)

    row = aggregated_delta[aggregated_delta["ablation_name"] == "patch_shuffle"].iloc[0]
    assert row["delta_vs_grid_baseline"] == pytest.approx(0.10)
    raw_row = raw_delta[raw_delta["ablation_name"] == "patch_shuffle"].iloc[0]
    assert raw_row["delta_vs_grid_baseline"] == pytest.approx(0.05)


def test_intermediate_figure_path_uses_part_and_slug(tmp_path):
    path = experiment_intermediate_figure_path(str(tmp_path), "part1", "accuracy vs tiles resnet18")

    assert path == str(tmp_path / "intermediate" / "part1_accuracy_vs_tiles_resnet18.png")


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


def test_load_part1_model_results_defaults_to_regular_frozen_baseline_condition(tmp_path):
    pd.DataFrame(
        [
            {
                "model_name": "mobilenetv3_small",
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.96,
                "ablation_name": None,
                "freeze_backbone": True,
                "classification_head": None,
            },
            {
                "model_name": "mobilenetv3_small",
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.97,
                "ablation_name": "unfrozen_pretrained_binary_head",
                "freeze_backbone": False,
                "classification_head": "binary_linear",
            },
            {
                "model_name": "mobilenetv3_small",
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.65,
                "ablation_name": "zero_shot_full_pretrained_head",
                "freeze_backbone": True,
                "classification_head": "imagenet_full_head",
            },
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)

    model_results = load_part1_model_results(str(tmp_path / "part1_raw_results.csv"), "mobilenetv3_small")

    assert len(model_results) == 1
    assert model_results.iloc[0]["best_val_accuracy"] == 0.96
    assert pd.isna(model_results.iloc[0]["ablation_name"])


def test_load_part1_model_results_raises_for_duplicate_part3_join_keys(tmp_path):
    pd.DataFrame(
        [
            {
                "model_name": "resnet18",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.70,
            },
            {
                "model_name": "resnet18",
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.72,
            },
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)

    with pytest.raises(ValueError, match="not unique"):
        load_part1_model_results(str(tmp_path / "part1_raw_results.csv"), "resnet18")


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


def test_part3_metrics_rebuild_named_deterministic_records_when_csv_is_missing(tmp_path):
    metrics = compute_part3_tile_permutation_metrics(
        tile_permutation_csv=str(tmp_path / "missing_tile_permutations.csv"),
        tiles_per_side_values=[1, 4, 7, 10],
        num_tile_permutations=3,
        seed=42,
        validation_samples=[],
        image_size=224,
    )

    assert len(metrics) == 12
    assert metrics["tile_permutation_name"].tolist() == ["easy", "medium", "hard"] * 4
    assert metrics.loc[:2, "global_tile_displacement"].eq(0.0).all()
    assert set(metrics.loc[3:, "tiles_per_side"]) == {4, 7, 10}


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


def test_part3_hardness_analysis_normalizes_joined_permutation_metadata(tmp_path):
    pd.DataFrame(
        [
            {
                "tiles_per_side": 2,
                "tile_permutation_id": 1,
                "tile_permutation_name": "hard",
                "tile_permutation_seed": 42,
                "tile_permutation": "[[[1, 0], [0, 1]], [[1, 1], [0, 0]]]",
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
                "tile_permutation_name": "easy",
                "tile_permutation_seed": 7,
                "best_val_accuracy": 0.75,
                "val_accuracy": 0.70,
            }
        ]
    ).to_csv(tmp_path / "part1_raw_results.csv", index=False)

    results = run_part3_hardness_analysis(
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

    assert "tile_permutation_name" in results["joined"].columns
    assert "tile_permutation_name_x" not in results["joined"].columns
    assert "tile_permutation_name_y" not in results["joined"].columns
    assert results["joined"].loc[0, "tile_permutation_name"] == "hard"
    assert results["joined"].loc[0, "tile_permutation_seed"] == 42


def test_part3_metric_grid_plots_are_reported_by_output_paths(tmp_path):
    joined = pd.DataFrame(
        [
            {
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 0,
                "tile_permutation_name": None,
                "best_val_accuracy": 0.70,
                "global_tile_displacement": 0.10,
                "adjacency_destruction_hardness": 0.30,
                "spatial_permutation_entropy": 0.20,
                "combined_hardness_score": 0.25,
            },
            {
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 1,
                "tile_permutation_name": "hard",
                "best_val_accuracy": 0.60,
                "global_tile_displacement": 0.80,
                "adjacency_destruction_hardness": 0.90,
                "spatial_permutation_entropy": 0.70,
                "combined_hardness_score": 0.70,
            },
        ]
    )

    output_paths = plot_part3_metric_grid_views(joined, str(tmp_path), "combined_hardness_score")
    paths = part3_output_paths(str(tmp_path), str(tmp_path))

    assert output_paths == {
        "grid_hardness": str(tmp_path / "part3_combined_hardness_grid_vs_hardness.png"),
        "grid_hardness_accuracy_3d": str(tmp_path / "part3_combined_hardness_grid_hardness_accuracy_3d.png"),
    }
    assert paths["plots"] == [
        str(tmp_path / "part3_combined_hardness_grid_vs_hardness.png"),
        str(tmp_path / "part3_combined_hardness_grid_hardness_accuracy_3d.png"),
    ]
    assert paths["metric_plots"]["combined_hardness_score"] == output_paths


def test_part3_metric_grid_plots_use_condition_markers_and_grid_axes(monkeypatch, tmp_path):
    markers = []
    scatter_args = []
    xtick_labels = []

    def fake_scatter_2d(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))
        scatter_args.append(args)

    def fake_scatter_3d(self, *args, **kwargs):
        markers.append(kwargs.get("marker"))
        scatter_args.append(args)

    def fake_set_xticks(self, ticks, labels=None, *args, **kwargs):
        if labels is not None:
            xtick_labels.extend(labels)

    monkeypatch.setattr("matplotlib.axes.Axes.scatter", fake_scatter_2d)
    monkeypatch.setattr("mpl_toolkits.mplot3d.axes3d.Axes3D.scatter", fake_scatter_3d)
    monkeypatch.setattr("matplotlib.axes.Axes.set_xticks", fake_set_xticks)
    joined = pd.DataFrame(
        [
            {
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.70,
                "tile_permutation_name_x": "hard",
                "tile_permutation_name_y": "easy",
                "global_tile_displacement": 0.10,
                "adjacency_destruction_hardness": 0.30,
                "spatial_permutation_entropy": 0.20,
                "combined_hardness_score": 0.25,
            },
            {
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 1,
                "best_val_accuracy": 0.65,
                "tile_permutation_name_x": "hard",
                "tile_permutation_name_y": "easy",
                "global_tile_displacement": 0.20,
                "adjacency_destruction_hardness": 0.35,
                "spatial_permutation_entropy": 0.25,
                "combined_hardness_score": 0.30,
            },
            {
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 2,
                "best_val_accuracy": 0.62,
                "tile_permutation_name_x": "hard",
                "tile_permutation_name_y": "medium",
                "global_tile_displacement": 0.40,
                "adjacency_destruction_hardness": 0.50,
                "spatial_permutation_entropy": 0.45,
                "combined_hardness_score": 0.48,
            },
            {
                "tiles_per_side": 2,
                "num_tiles": 4,
                "tile_permutation_id": 3,
                "best_val_accuracy": 0.60,
                "tile_permutation_name_x": "easy",
                "tile_permutation_name_y": "hard",
                "global_tile_displacement": 0.80,
                "adjacency_destruction_hardness": 0.90,
                "spatial_permutation_entropy": 0.70,
                "combined_hardness_score": 0.70,
            },
        ]
    )

    plot_part3_metric_grid_views(joined, str(tmp_path), "combined_hardness_score")

    assert PERMUTATION_MARKERS["baseline"] in markers
    assert PERMUTATION_MARKERS["easy"] in markers
    assert PERMUTATION_MARKERS["medium"] in markers
    assert PERMUTATION_MARKERS["hard"] in markers
    assert xtick_labels[:2] == ["1", "2x2"]
    assert list(scatter_args[0][0]) == [0]
    assert list(scatter_args[0][1]) == [0.25]
    assert len(scatter_args[0]) == 2
    assert len(scatter_args[-1]) == 3
    assert list(scatter_args[-1][2]) == [0.60]


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


def test_part3_correlation_input_scopes_deduplicate_and_exclude_baseline_rows():
    rows = []
    for tile_permutation_id, accuracy in [(1, 0.91), (2, 0.90), (3, 0.89)]:
        rows.append(
            {
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": tile_permutation_id,
                "tile_permutation_name": ["easy", "medium", "hard"][tile_permutation_id - 1],
                "best_val_accuracy": accuracy,
                "global_tile_displacement": 0.00,
                "adjacency_destruction_hardness": 0.00,
                "spatial_permutation_entropy": 0.00,
                "combined_hardness_score": 0.00,
            }
        )
    for tiles_per_side in [4, 7, 10]:
        for tile_permutation_id in [1, 2, 3]:
            rows.append(
                {
                    "tiles_per_side": tiles_per_side,
                    "num_tiles": tiles_per_side * tiles_per_side,
                    "tile_permutation_id": tile_permutation_id,
                    "tile_permutation_name": ["easy", "medium", "hard"][tile_permutation_id - 1],
                    "best_val_accuracy": 0.90 - 0.01 * tile_permutation_id,
                    "global_tile_displacement": 0.10 * tile_permutation_id,
                    "adjacency_destruction_hardness": 0.20 * tile_permutation_id,
                    "spatial_permutation_entropy": 0.15 * tile_permutation_id,
                    "combined_hardness_score": 0.18 * tile_permutation_id,
                }
            )

    correlation_input = build_part3_correlation_input(pd.DataFrame(rows))
    counts = correlation_input.groupby("analysis_scope").size().to_dict()
    one_baseline = correlation_input[correlation_input["analysis_scope"] == "one_baseline_row"]
    non_baseline = correlation_input[correlation_input["analysis_scope"] == "non_baseline_rows"]

    assert counts == {"all_rows": 12, "non_baseline_rows": 9, "one_baseline_row": 10}
    assert one_baseline["num_tiles"].eq(1).sum() == 1
    assert one_baseline.loc[one_baseline["num_tiles"] == 1, "tile_permutation_name"].tolist() == ["baseline"]
    assert non_baseline["num_tiles"].eq(1).sum() == 0


def test_part3_correlations_include_analysis_scopes_and_scope_sample_sizes():
    rows = []
    for tile_permutation_id, accuracy in [(1, 0.91), (2, 0.90), (3, 0.89)]:
        rows.append(
            {
                "tiles_per_side": None,
                "num_tiles": 1,
                "tile_permutation_id": tile_permutation_id,
                "best_val_accuracy": accuracy,
                "global_tile_displacement": 0.00,
                "adjacency_destruction_hardness": 0.00,
                "spatial_permutation_entropy": 0.00,
                "combined_hardness_score": 0.00,
            }
        )
    for index in range(9):
        rows.append(
            {
                "tiles_per_side": 4 + index,
                "num_tiles": 16 + index,
                "tile_permutation_id": (index % 3) + 1,
                "best_val_accuracy": 0.88 - 0.01 * index,
                "global_tile_displacement": 0.10 + 0.03 * index,
                "adjacency_destruction_hardness": 0.20 + 0.04 * index,
                "spatial_permutation_entropy": 0.15 + 0.02 * index,
                "combined_hardness_score": 0.18 + 0.03 * index,
            }
        )

    correlations = compute_part3_metric_correlations(pd.DataFrame(rows))

    assert set(correlations["analysis_scope"]) == {"all_rows", "one_baseline_row", "non_baseline_rows"}
    assert correlations.groupby("analysis_scope")["n"].first().to_dict() == {
        "all_rows": 12,
        "non_baseline_rows": 9,
        "one_baseline_row": 10,
    }


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
