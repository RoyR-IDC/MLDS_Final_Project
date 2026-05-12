"""Shared evaluation helpers for experiment outputs."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import torch
from torch._C import device as TorchDevice
from tqdm.auto import tqdm

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)
from src.experiments.results import (
    aggregate_accuracy,
    build_result_row,
    experiment_output_paths,
    save_aggregated_accuracy,
    save_rows,
)
from src.models.registry import validate_model_name
from src.preprocessing.samples import Sample, discover_samples, stratified_split
from src.preprocessing.tile_permutations import (
    generate_tile_permutations,
    tile_permutation_from_jsonable,
    tile_permutation_to_jsonable,
)
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir, save_csv


__all__ = [
    "aggregate_accuracy",
    "build_result_row",
    "experiment_output_paths",
    "save_aggregated_accuracy",
    "save_rows",
]


def _read_csv_dataframe(path: str) -> pd.DataFrame:
    """Read a CSV as a DataFrame with a narrowed static type."""

    return cast(pd.DataFrame, pd.read_csv(path))


def _as_axes(axis: object) -> Axes:
    """Narrow matplotlib's broad ``subplots`` return type for single-axis plots."""

    return cast(Axes, axis)


def _sorted_dataframe(frame: pd.DataFrame, by: Sequence[str] | str, *, na_position: str = "last") -> pd.DataFrame:
    """Return a sorted DataFrame with a narrowed static type."""

    return cast(pd.DataFrame, frame.sort_values(by, na_position=na_position))


def _reset_dataframe_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a reset index and a narrowed static type."""

    return cast(pd.DataFrame, frame.reset_index(drop=True))


def get_device(config: CVExperimentConfig) -> TorchDevice:
    """Return a configured device, falling back to CPU when needed.

    Args:
        config: Normalized experiment configuration.

    Returns:
        Torch device selected from the config.
    """

    requested = str(config.device).lower()
    if requested == "auto":
        device = TorchDevice("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Selected device: {device}")
        return device
    device = TorchDevice(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA GPU is available. In Colab, choose "
            "Runtime > Change runtime type > GPU, then reconnect and rerun setup."
        )
    print(f"Selected device: {device}")
    return device


def _build_balanced_sample_subset(samples: list[Sample], limit: int, seed: int) -> list[Sample]:
    cats = [sample for sample in samples if sample[1] == 0]
    dogs = [sample for sample in samples if sample[1] == 1]
    max_per_class = limit // 2
    selected_count = min(len(cats), len(dogs), max_per_class)
    if selected_count == 0:
        raise ValueError("Not enough cat/dog samples to build a balanced subset")

    rng = random.Random(seed)
    cats = cats[:selected_count]
    dogs = dogs[:selected_count]
    balanced = cats + dogs
    rng.shuffle(balanced)
    return balanced


def load_experiment_samples(config: CVExperimentConfig, seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Discover and split configured Dogs vs Cats samples.

    Args:
        config: Normalized experiment configuration with data options.
        seed: Random seed used by the stratified split.

    Returns:
        Tuple of train, validation, and test sample lists.
    """

    samples = discover_samples(config.data_dir)
    if config.sample_data and config.sample_limit is not None:
        samples = _build_balanced_sample_subset(samples, config.sample_limit, seed)

    split_samples = stratified_split(
        samples,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=seed,
    )
    return split_samples


def load_part1_model_baseline_raw_rows(
    config: CVExperimentConfig,
    model_name: str,
    ablation_name: str = "regular_part1",
) -> list[dict[str, Any]]:
    """Load raw Part 1 rows for one model and retag them for Part 2 comparison."""

    raw_path = os.path.join(config.results_dir, "part1_raw_results.csv")
    if not os.path.exists(raw_path):
        return []

    part1_raw = _read_csv_dataframe(raw_path)
    if "model_name" not in part1_raw.columns:
        return []

    part1_raw_any = cast(Any, part1_raw)
    baseline = cast(pd.DataFrame, part1_raw_any[part1_raw_any["model_name"] == model_name].copy())
    if baseline.empty:
        return []

    baseline["part"] = config.part
    baseline["config_name"] = config.config_name
    baseline["ablation_name"] = ablation_name
    if "run_id" not in baseline.columns:
        baseline["run_id"] = "part1_regular_training"
    return cast(list[dict[str, Any]], cast(Any, baseline).to_dict("records"))


def load_part1_model_baseline_aggregated(
    config: CVExperimentConfig,
    model_name: str,
    ablation_name: str = "regular_part1",
) -> pd.DataFrame:
    """Load aggregated Part 1 rows for one model and retag them for Part 2 comparison."""

    aggregated_path = os.path.join(config.results_dir, "part1_aggregated_results.csv")
    raw_path = os.path.join(config.results_dir, "part1_raw_results.csv")
    source_path = aggregated_path if os.path.exists(aggregated_path) else raw_path
    if not os.path.exists(source_path):
        print(f"Part 1 results were not found. Run Part 1 first to include the regular {model_name} baseline.")
        return pd.DataFrame()

    part1_results = _read_csv_dataframe(source_path)
    if "model_name" not in part1_results.columns:
        return pd.DataFrame()

    part1_results_any = cast(Any, part1_results)
    baseline = cast(pd.DataFrame, part1_results_any[part1_results_any["model_name"] == model_name].copy())
    if baseline.empty:
        print(f"Part 1 results exist, but no rows were found for {model_name}.")
        return pd.DataFrame()

    if "mean_best_epoch_val_accuracy" not in baseline.columns:
        if not {"val_accuracy", "best_val_accuracy"}.issubset(baseline.columns):
            raise ValueError(
                "Part 1 aggregated results use an unsupported schema. "
                "Rerun Part 1 to regenerate aggregate columns with explicit final_epoch/best_epoch names."
            )
        baseline = cast(pd.DataFrame, aggregate_accuracy(
            cast(pd.DataFrame, baseline),
            group_columns=["model_name", "tiles_per_side", "num_tiles"],
        ))

    baseline["ablation_name"] = ablation_name
    baseline["config_name"] = config.config_name
    return baseline


def plot_accuracy_vs_tiles(aggregated: pd.DataFrame, output_path: str, model_column: str = "model_name") -> None:
    """Save an accuracy-vs-number-of-tiles plot.

    Args:
        aggregated: Aggregated result table.
        output_path: Destination path for the figure.
        model_column: Column used to split one curve per model.
    """

    ensure_dir(os.path.dirname(output_path) or ".")
    fig, axis = plt.subplots(figsize=(8, 5))
    ax = _as_axes(axis)
    for model_name, group in aggregated.groupby(model_column):
        group = _sorted_dataframe(cast(pd.DataFrame, group), "num_tiles")
        ax.errorbar(
            group["num_tiles"],
            group["mean_best_epoch_val_accuracy"],
            yerr=cast(Any, group)["std_best_epoch_val_accuracy"].fillna(0.0),
            marker="o",
            label=str(model_name),
        )
    ax.set_xlabel("Number of tiles")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Accuracy vs Number of Tiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_ablation_results(aggregated: pd.DataFrame, output_path: str) -> None:
    """Save a baseline-vs-improvement ablation plot.

    Args:
        aggregated: Aggregated Part 2 result table.
        output_path: Destination path for the figure.
    """

    ensure_dir(os.path.dirname(output_path) or ".")
    fig, axis = plt.subplots(figsize=(8, 5))
    ax = _as_axes(axis)
    for tiles_per_side, group in aggregated.groupby("tiles_per_side"):
        sorted_group = _sorted_dataframe(cast(pd.DataFrame, group), "ablation_name")
        ax.plot(
            sorted_group["ablation_name"],
            sorted_group["mean_best_epoch_val_accuracy"],
            marker="o",
            label=f"{tiles_per_side}x{tiles_per_side}",
        )
    ax.set_xlabel("Ablation")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Baseline vs Improved Ablations")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def part3_output_paths(results_dir: str, figures_dir: str) -> Dict[str, object]:
    """Return stable output paths for notebook-owned Part 3 analysis."""

    combined_plot = os.path.join(figures_dir, "part3_metrics_vs_accuracy.png")
    return {
        "metrics": os.path.join(results_dir, "tile_permutation_metrics.csv"),
        "joined": os.path.join(results_dir, "metric_accuracy_joined.csv"),
        "correlations": os.path.join(results_dir, "metric_accuracy_correlations.csv"),
        "plots": [combined_plot] if os.path.exists(combined_plot) else [],
    }


def load_or_build_part1_tile_permutations(
    tile_permutation_csv: str,
    tiles_per_side_values: Sequence[int],
    num_tile_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Load saved Part 1 tile-permutation metadata, or recreate it deterministically."""

    if os.path.exists(tile_permutation_csv):
        return _read_csv_dataframe(tile_permutation_csv)

    rows = [
        {
            "tiles_per_side": None,
            "tile_permutation_id": 0,
            "tile_permutation_seed": seed,
            "tile_permutation": json.dumps(None),
        }
    ]
    for tiles_per_side in [int(value) for value in tiles_per_side_values]:
        if tiles_per_side == 1:
            continue
        for offset, tile_permutation in enumerate(
            generate_tile_permutations(tiles_per_side, int(num_tile_permutations), seed=seed),
            start=1,
        ):
            rows.append(
                {
                    "tiles_per_side": tiles_per_side,
                    "tile_permutation_id": offset,
                    "tile_permutation_seed": seed,
                    "tile_permutation": json.dumps(tile_permutation_to_jsonable(tile_permutation)),
                }
            )
    return pd.DataFrame(rows)


def _executable_part1_tile_permutations(tile_permutations: pd.DataFrame) -> pd.DataFrame:
    """Return tile-permutation rows that correspond to actual Part 1 executions."""

    executable = tile_permutations.copy()
    return _reset_dataframe_index(
        _sorted_dataframe(executable, ["tiles_per_side", "tile_permutation_id"], na_position="first")
    )


PART3_METRIC_COLUMNS = [
    "global_tile_displacement",
    "center_weighted_displacement",
    "adjacency_destruction_hardness",
    "combined_hardness_score",
]


def compute_part3_tile_permutation_metrics(
    *,
    tile_permutation_csv: str,
    tiles_per_side_values: Sequence[int],
    num_tile_permutations: int,
    seed: int,
    alpha_center: float = 1.0,
    weight_center: float = 0.5,
    weight_dist: float = 0.5,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute Part 3 hardness metrics for reusable Part 1 tile permutations."""

    rows: list[dict[str, Any]] = []
    tile_permutations = load_or_build_part1_tile_permutations(
        tile_permutation_csv=tile_permutation_csv,
        tiles_per_side_values=tiles_per_side_values,
        num_tile_permutations=num_tile_permutations,
        seed=seed,
    )
    tile_permutations = _executable_part1_tile_permutations(tile_permutations)
    iterator = tile_permutations.iterrows()
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(tile_permutations),
            desc="Part 3 metrics",
            unit="tile permutation",
        )
    for _, row in iterator:
        raw_tile_permutation = cast(Any, row["tile_permutation"])
        if raw_tile_permutation is None or (
            isinstance(raw_tile_permutation, float) and bool(pd.isna(raw_tile_permutation))
        ):
            serialized_tile_permutation = None
        elif isinstance(raw_tile_permutation, str):
            serialized_tile_permutation = json.loads(raw_tile_permutation)
        else:
            serialized_tile_permutation = raw_tile_permutation
        raw_tiles_per_side = cast(Any, row["tiles_per_side"])
        tiles_per_side = None if bool(pd.isna(raw_tiles_per_side)) else int(raw_tiles_per_side)
        tile_permutation = tile_permutation_from_jsonable(serialized_tile_permutation, tiles_per_side)
        num_tiles = 1 if tiles_per_side is None else tiles_per_side * tiles_per_side
        global_tile_displacement = compute_global_displacement(tile_permutation, tiles_per_side)
        center_weighted_displacement = compute_center_weighted_displacement(
            tile_permutation,
            tiles_per_side,
            alpha_center,
        )
        adjacency_destruction_hardness = compute_adjacency_destruction_hardness(
            tile_permutation,
            tiles_per_side,
        )
        combined_hardness_score = compute_combined_hardness(
            tile_permutation=tile_permutation,
            tiles_per_side=tiles_per_side,
            alpha_center=alpha_center,
            weight_center=weight_center,
            weight_dist=weight_dist,
        )
        rows.append(
            {
                "tiles_per_side": tiles_per_side,
                "num_tiles": num_tiles,
                "tile_permutation_id": int(cast(Any, row["tile_permutation_id"])),
                "tile_permutation_seed": cast(Any, row.get("tile_permutation_seed")),
                "global_tile_displacement": global_tile_displacement,
                "center_weighted_displacement": center_weighted_displacement,
                "adjacency_destruction_hardness": adjacency_destruction_hardness,
                "combined_hardness_score": combined_hardness_score,
            }
        )
    return _reset_dataframe_index(
        _sorted_dataframe(pd.DataFrame(rows), ["tiles_per_side", "tile_permutation_id"], na_position="first")
    )


def validate_part3_non_identity_metrics(metrics: pd.DataFrame) -> None:
    """Raise if a tiled non-baseline permutation has zero hardness for every metric."""

    if metrics.empty:
        return

    metrics_any = cast(Any, metrics)
    invalid = cast(
        pd.DataFrame,
        metrics_any[
            (metrics_any["tiles_per_side"].notna())
            & (metrics_any["tiles_per_side"].astype(float) > 1)
            & (metrics_any["tile_permutation_id"].astype(int) > 0)
            & (metrics_any[PART3_METRIC_COLUMNS].fillna(0.0).eq(0.0).all(axis=1))
        ],
    )
    if invalid.empty:
        return

    row = invalid.iloc[0]
    raise ValueError(
        "Part 3 hardness metrics are all zero for a tiled permutation: "
        f"tiles_per_side={int(row['tiles_per_side'])}, tile_permutation_id={int(row['tile_permutation_id'])}."
    )


def load_part1_model_results(part1_results_csv: str, model_name: str) -> pd.DataFrame:
    """Load Part 1 raw results filtered to one trained model."""

    model_name = validate_model_name(model_name)
    raw_results = _read_csv_dataframe(part1_results_csv)
    if "model_name" not in raw_results.columns:
        raise ValueError("Part 1 results must contain a model_name column")

    raw_results_any = cast(Any, raw_results)
    model_results = cast(pd.DataFrame, raw_results_any[raw_results_any["model_name"] == model_name].copy())
    if model_results.empty:
        raise ValueError(f"No Part 1 rows found for model_name='{model_name}'")
    return model_results


def compute_part3_metric_correlations(joined: pd.DataFrame, group_name: str = "resnet18") -> pd.DataFrame:
    """Compute accuracy correlations for each Part 3 hardness metric."""

    rows: list[dict[str, Any]] = []
    for metric in PART3_METRIC_COLUMNS:
        frame = cast(pd.DataFrame, joined.dropna(subset=[metric, "best_val_accuracy"]))
        frame_any = cast(Any, frame)
        if len(frame) < 2 or frame_any[metric].nunique() < 2 or frame_any["best_val_accuracy"].nunique() < 2:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(frame_any[metric].corr(frame_any["best_val_accuracy"], method="pearson"))
            spearman = float(frame_any[metric].corr(frame_any["best_val_accuracy"], method="spearman"))
        rows.append(
            {
                "group": group_name,
                "metric": metric,
                "pearson": pearson,
                "spearman": spearman,
                "n": len(frame),
            }
        )
    return pd.DataFrame(rows)


def plot_part3_metrics_vs_accuracy(joined: pd.DataFrame, figures_dir: str) -> None:
    """Save one combined Part 3 hardness metric-vs-accuracy scatter plot."""

    ensure_dir(figures_dir)
    fig, axis = plt.subplots(figsize=(8, 5))
    ax = _as_axes(axis)
    for metric in PART3_METRIC_COLUMNS:
        ax.scatter(
            joined[metric],
            joined["best_val_accuracy"],
            label=metric.replace("_", " ").title(),
            alpha=0.8,
        )
    ax.set_xlabel("Hardness metric value")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Part 3 Hardness Metrics vs Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "part3_metrics_vs_accuracy.png"), dpi=160)
    plt.close(fig)


def load_part3_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load saved Part 3 metric, joined, and correlation tables."""

    return {
        "metrics": _read_csv_dataframe(os.path.join(results_dir, "tile_permutation_metrics.csv")),
        "joined": _read_csv_dataframe(os.path.join(results_dir, "metric_accuracy_joined.csv")),
        "correlations": _read_csv_dataframe(os.path.join(results_dir, "metric_accuracy_correlations.csv")),
    }


def run_part3_hardness_analysis(
    *,
    results_dir: str,
    figures_dir: str,
    part1_results_csv: str,
    tile_permutation_csv: str,
    tiles_per_side_values: Sequence[int],
    num_tile_permutations: int,
    seed: int,
    model_name: str = "resnet18",
    alpha_center: float = 1.0,
    weight_center: float = 0.5,
    weight_dist: float = 0.5,
    verbose: bool = True,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run notebook-owned Part 3 hardness analysis and save output tables/plots."""

    def log(message: str) -> None:
        if verbose:
            print(message)

    log("Preparing Part 3 output directories...")
    ensure_dir(results_dir)
    ensure_dir(figures_dir)
    log("Loading or rebuilding Part 1 tile permutations...")
    log("Calculating hardness metrics...")
    metrics = compute_part3_tile_permutation_metrics(
        tile_permutation_csv=tile_permutation_csv,
        tiles_per_side_values=tiles_per_side_values,
        num_tile_permutations=num_tile_permutations,
        seed=seed,
        alpha_center=alpha_center,
        weight_center=weight_center,
        weight_dist=weight_dist,
        show_progress=show_progress,
    )
    validate_part3_non_identity_metrics(metrics)
    log("Saving hardness metric table...")
    save_csv(metrics, os.path.join(results_dir, "tile_permutation_metrics.csv"))

    log(f"Loading Part 1 {model_name} results...")
    raw_results = load_part1_model_results(part1_results_csv, model_name)
    log(f"Joining hardness metrics with {model_name} accuracy...")
    joined = raw_results.merge(metrics, on=["tiles_per_side", "num_tiles", "tile_permutation_id"], how="left")
    joined = _reset_dataframe_index(
        _sorted_dataframe(joined, ["tiles_per_side", "tile_permutation_id"], na_position="first")
    )
    log("Saving joined metric-accuracy table...")
    save_csv(joined, os.path.join(results_dir, "metric_accuracy_joined.csv"))

    log("Computing metric-accuracy correlations...")
    correlations = compute_part3_metric_correlations(joined, group_name=model_name)
    log("Saving correlations and plots...")
    save_csv(correlations, os.path.join(results_dir, "metric_accuracy_correlations.csv"))
    plot_part3_metrics_vs_accuracy(joined, figures_dir)
    log("Part 3 hardness analysis complete.")
    return {"metrics": metrics, "joined": joined, "correlations": correlations}
