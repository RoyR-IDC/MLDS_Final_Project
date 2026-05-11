"""Shared evaluation helpers for experiment outputs."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.evaluation.permutation_difficulty import (
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)
from src.preprocessing.dogs_cats import Sample, discover_samples, stratified_split
from src.preprocessing.permutations import PermutationRecord, generate_permutations, identity_permutation
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir, save_csv


def get_device(config: CVExperimentConfig) -> torch.device:
    """Return a configured device, falling back to CPU when needed.

    Args:
        config: Normalized experiment configuration.

    Returns:
        Torch device selected from the config.
    """

    requested = str(config.device).lower()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Selected device: {device}")
        return device
    device = torch.device(requested)
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


def build_result_row(
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: PermutationRecord,
    seed: int,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one experiment result row for repeated accuracy measurements."""

    row = {
        'part': config.part,
        'run_id': run_id,
        'config_name': config.config_name,
        'model_name': model_name,
        'grid_size': record.grid_size,
        'num_tiles': record.grid_size * record.grid_size,
        'permutation_id': record.permutation_id,
        'permutation_seed': record.permutation_seed,
        'seed': seed,
        **metrics,
    }
    return row


def _csv_safe_value(value: Any) -> Any:
    """Convert array-like experiment values into CSV-friendly scalars/strings."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return str(tensor.tolist())

    if isinstance(value, Mapping):
        return str({str(key): _csv_safe_value(nested_value) for key, nested_value in value.items()})

    if isinstance(value, (list, tuple)):
        return str([_csv_safe_value(item) for item in value])

    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not isinstance(value, (str, bytes)):
        converted = tolist()
        if converted is value:
            return value
        return _csv_safe_value(converted)

    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        try:
            return item()
        except (TypeError, ValueError):
            return value

    return value


def _csv_safe_rows(rows: Sequence[Mapping[Any, Any]]) -> list[dict[str, Any]]:
    """Return rows with string column names and scalar/string values."""

    safe_rows = []
    for row in rows:
        safe_rows.append({str(key): _csv_safe_value(value) for key, value in row.items()})
    return safe_rows


def aggregate_accuracy(raw_results: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate accuracy columns for repeated experiment runs.

    Args:
        raw_results: Per-run result table.
        group_columns: Columns used to group repeated measurements.

    Returns:
        Aggregated result table with mean, standard deviation, and counts.
    """

    aggregated_results = (
        raw_results.groupby(list(group_columns), dropna=False)
        .agg(
            mean_final_epoch_val_accuracy=("val_accuracy", "mean"),
            std_final_epoch_val_accuracy=("val_accuracy", "std"),
            mean_best_epoch_val_accuracy=("best_val_accuracy", "mean"),
            std_best_epoch_val_accuracy=("best_val_accuracy", "std"),
            n_runs=("val_accuracy", "count"),
        )
        .reset_index()
    )
    return aggregated_results


def load_part1_model_baseline_raw_rows(
    config: CVExperimentConfig,
    model_name: str,
    ablation_name: str = "regular_part1",
) -> list[dict[str, Any]]:
    """Load raw Part 1 rows for one model and retag them for Part 2 comparison."""

    raw_path = os.path.join(config.results_dir, "part1_raw_results.csv")
    if not os.path.exists(raw_path):
        return []

    part1_raw = pd.read_csv(raw_path)
    if "model_name" not in part1_raw.columns:
        return []

    baseline = part1_raw[part1_raw["model_name"] == model_name].copy()
    if baseline.empty:
        return []

    baseline["part"] = config.part
    baseline["config_name"] = config.config_name
    baseline["ablation_name"] = ablation_name
    if "run_id" not in baseline.columns:
        baseline["run_id"] = "part1_regular_training"
    return baseline.to_dict("records")


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

    part1_results = pd.read_csv(source_path)
    if "model_name" not in part1_results.columns:
        return pd.DataFrame()

    baseline = part1_results[part1_results["model_name"] == model_name].copy()
    if baseline.empty:
        print(f"Part 1 results exist, but no rows were found for {model_name}.")
        return pd.DataFrame()

    if "mean_best_epoch_val_accuracy" not in baseline.columns:
        if not {"val_accuracy", "best_val_accuracy"}.issubset(baseline.columns):
            raise ValueError(
                "Part 1 aggregated results use an unsupported schema. "
                "Rerun Part 1 to regenerate aggregate columns with explicit final_epoch/best_epoch names."
            )
        baseline = aggregate_accuracy(
            baseline,
            group_columns=["model_name", "grid_size", "num_tiles"],
        )

    baseline["ablation_name"] = ablation_name
    baseline["config_name"] = config.config_name
    return baseline


def experiment_output_paths(results_dir: str, figures_dir: str, part_name: str) -> Dict[str, str]:
    """Return standard output paths for a notebook-owned experiment.

    Args:
        results_dir: Directory for CSV result files.
        figures_dir: Directory for saved figures.
        part_name: Experiment prefix such as ``part1`` or ``part2``.

    Returns:
        Mapping of stable artifact names to filesystem paths.
    """

    figure_name = "accuracy_vs_tiles" if part_name == "part1" else "ablation_comparison"
    output_paths = {
        "raw_results": os.path.join(results_dir, f"{part_name}_raw_results.csv"),
        "aggregated_results": os.path.join(results_dir, f"{part_name}_aggregated_results.csv"),
        "figure": os.path.join(figures_dir, f"{part_name}_{figure_name}.png"),
    }
    if part_name == "part1":
        output_paths["permutations"] = os.path.join(results_dir, "part1_permutations.csv")
    return output_paths


def save_rows(rows: Sequence[Mapping[Any, Any]], output_path: str) -> None:
    """Save experiment rows to CSV.

    Args:
        rows: Result rows to save.
        output_path: Destination CSV path.
    """

    save_csv(_csv_safe_rows(rows), output_path)


def save_aggregated_accuracy(raw_results: pd.DataFrame, group_columns: Sequence[str], output_path: str) -> pd.DataFrame:
    """Aggregate raw results and save the aggregated table.

    Args:
        raw_results: Per-run result table.
        group_columns: Columns used for aggregation.
        output_path: Destination CSV path.

    Returns:
        Aggregated accuracy table.
    """

    aggregated_results = aggregate_accuracy(raw_results, group_columns)
    ensure_dir(os.path.dirname(output_path) or ".")
    aggregated_results.to_csv(output_path, index=False)
    return aggregated_results


def plot_accuracy_vs_tiles(aggregated: pd.DataFrame, output_path: str, model_column: str = "model_name") -> None:
    """Save an accuracy-vs-number-of-tiles plot.

    Args:
        aggregated: Aggregated result table.
        output_path: Destination path for the figure.
        model_column: Column used to split one curve per model.
    """

    ensure_dir(os.path.dirname(output_path) or ".")
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, group in aggregated.groupby(model_column):
        group = group.sort_values("num_tiles")
        ax.errorbar(
            group["num_tiles"],
            group["mean_best_epoch_val_accuracy"],
            yerr=group["std_best_epoch_val_accuracy"].fillna(0.0),
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
    fig, ax = plt.subplots(figsize=(8, 5))
    for grid_size, group in aggregated.groupby("grid_size"):
        sorted_group = group.sort_values("ablation_name")
        ax.plot(
            sorted_group["ablation_name"],
            sorted_group["mean_best_epoch_val_accuracy"],
            marker="o",
            label=f"{grid_size}x{grid_size}",
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
        "metrics": os.path.join(results_dir, "permutation_metrics.csv"),
        "joined": os.path.join(results_dir, "metric_accuracy_joined.csv"),
        "correlations": os.path.join(results_dir, "metric_accuracy_correlations.csv"),
        "plots": [combined_plot] if os.path.exists(combined_plot) else [],
    }


def load_or_build_part1_permutations(
    permutation_csv: str,
    grid_sizes: Sequence[int],
    num_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Load saved Part 1 permutation metadata, or recreate it deterministically."""

    if os.path.exists(permutation_csv):
        return pd.read_csv(permutation_csv)

    rows = []
    for grid_size in [int(value) for value in grid_sizes]:
        rows.append(
            {
                "grid_size": grid_size,
                "permutation_id": 0,
                "permutation_seed": seed,
                "permutation": json.dumps(identity_permutation(grid_size)),
            }
        )
        if grid_size == 1:
            continue
        for offset, permutation in enumerate(
            generate_permutations(grid_size, int(num_permutations), seed=seed),
            start=1,
        ):
            rows.append(
                {
                    "grid_size": grid_size,
                    "permutation_id": offset,
                    "permutation_seed": seed,
                    "permutation": json.dumps(permutation),
                }
            )
    return pd.DataFrame(rows)


def _executable_part1_permutations(permutations: pd.DataFrame) -> pd.DataFrame:
    """Return permutation rows that correspond to actual Part 1 executions."""

    duplicate_untiled_permutation = (permutations["grid_size"].astype(int) == 1) & (
        permutations["permutation_id"].astype(int) > 0
    )
    executable = permutations[~duplicate_untiled_permutation]
    return executable.sort_values(["grid_size", "permutation_id"]).reset_index(drop=True)


PART3_METRIC_COLUMNS = [
    "global_tile_displacement",
    "center_weighted_displacement",
    "combined_hardness_score",
]


def compute_part3_permutation_metrics(
    *,
    permutation_csv: str,
    grid_sizes: Sequence[int],
    num_permutations: int,
    seed: int,
    alpha_center: float = 1.0,
    weight_center: float = 0.5,
    weight_dist: float = 0.5,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Compute renamed Part 3 hardness metrics for reusable Part 1 permutations."""

    rows: list[dict[str, Any]] = []
    permutations = load_or_build_part1_permutations(
        permutation_csv=permutation_csv,
        grid_sizes=grid_sizes,
        num_permutations=num_permutations,
        seed=seed,
    )
    permutations = _executable_part1_permutations(permutations)
    iterator = permutations.iterrows()
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(permutations),
            desc="Part 3 metrics",
            unit="permutation",
        )
    for _, row in iterator:
        permutation = json.loads(row["permutation"]) if isinstance(row["permutation"], str) else row["permutation"]
        grid_size = int(row["grid_size"])
        global_tile_displacement = compute_global_displacement(permutation, grid_size)
        center_weighted_displacement = compute_center_weighted_displacement(permutation, grid_size, alpha_center)
        combined_hardness_score = compute_combined_hardness(
            permutation=permutation,
            N=grid_size,
            alpha_center=alpha_center,
            weight_center=weight_center,
            weight_dist=weight_dist,
        )
        rows.append(
            {
                "grid_size": grid_size,
                "num_tiles": grid_size * grid_size,
                "permutation_id": int(row["permutation_id"]),
                "permutation_seed": row.get("permutation_seed"),
                "global_tile_displacement": global_tile_displacement,
                "center_weighted_displacement": center_weighted_displacement,
                "combined_hardness_score": combined_hardness_score,
            }
        )
    return pd.DataFrame(rows).sort_values(["grid_size", "permutation_id"]).reset_index(drop=True)


def validate_part3_non_identity_metrics(metrics: pd.DataFrame) -> None:
    """Raise if a tiled non-identity permutation has zero hardness for every metric."""

    if metrics.empty:
        return

    invalid = metrics[
        (metrics["grid_size"].astype(int) > 1)
        & (metrics["permutation_id"].astype(int) > 0)
        & (metrics[PART3_METRIC_COLUMNS].fillna(0.0).eq(0.0).all(axis=1))
    ]
    if invalid.empty:
        return

    row = invalid.iloc[0]
    raise ValueError(
        "Part 3 hardness metrics are all zero for a non-identity tiled permutation: "
        f"grid_size={int(row['grid_size'])}, permutation_id={int(row['permutation_id'])}."
    )


def load_part1_model_results(part1_results_csv: str, model_name: str) -> pd.DataFrame:
    """Load Part 1 raw results filtered to one trained model."""

    raw_results = pd.read_csv(part1_results_csv)
    if "model_name" not in raw_results.columns:
        raise ValueError("Part 1 results must contain a model_name column")

    model_results = raw_results[raw_results["model_name"] == model_name].copy()
    if model_results.empty:
        raise ValueError(f"No Part 1 rows found for model_name='{model_name}'")
    return model_results


def compute_part3_metric_correlations(joined: pd.DataFrame, group_name: str = "resnet18") -> pd.DataFrame:
    """Compute accuracy correlations for each Part 3 hardness metric."""

    rows: list[dict[str, Any]] = []
    for metric in PART3_METRIC_COLUMNS:
        frame = joined.dropna(subset=[metric, "best_val_accuracy"])
        if len(frame) < 2 or frame[metric].nunique() < 2 or frame["best_val_accuracy"].nunique() < 2:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = float(frame[metric].corr(frame["best_val_accuracy"], method="pearson"))
            spearman = float(frame[metric].corr(frame["best_val_accuracy"], method="spearman"))
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
    fig, ax = plt.subplots(figsize=(8, 5))
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
        "metrics": pd.read_csv(os.path.join(results_dir, "permutation_metrics.csv")),
        "joined": pd.read_csv(os.path.join(results_dir, "metric_accuracy_joined.csv")),
        "correlations": pd.read_csv(os.path.join(results_dir, "metric_accuracy_correlations.csv")),
    }


def run_part3_hardness_analysis(
    *,
    results_dir: str,
    figures_dir: str,
    part1_results_csv: str,
    permutation_csv: str,
    grid_sizes: Sequence[int],
    num_permutations: int,
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
    log("Loading or rebuilding Part 1 permutations...")
    log("Calculating hardness metrics...")
    metrics = compute_part3_permutation_metrics(
        permutation_csv=permutation_csv,
        grid_sizes=grid_sizes,
        num_permutations=num_permutations,
        seed=seed,
        alpha_center=alpha_center,
        weight_center=weight_center,
        weight_dist=weight_dist,
        show_progress=show_progress,
    )
    validate_part3_non_identity_metrics(metrics)
    log("Saving hardness metric table...")
    save_csv(metrics, os.path.join(results_dir, "permutation_metrics.csv"))

    log(f"Loading Part 1 {model_name} results...")
    raw_results = load_part1_model_results(part1_results_csv, model_name)
    log(f"Joining hardness metrics with {model_name} accuracy...")
    joined = raw_results.merge(metrics, on=["grid_size", "num_tiles", "permutation_id"], how="left")
    joined = joined.sort_values(["grid_size", "permutation_id"]).reset_index(drop=True)
    log("Saving joined metric-accuracy table...")
    save_csv(joined, os.path.join(results_dir, "metric_accuracy_joined.csv"))

    log("Computing metric-accuracy correlations...")
    correlations = compute_part3_metric_correlations(joined, group_name=model_name)
    log("Saving correlations and plots...")
    save_csv(correlations, os.path.join(results_dir, "metric_accuracy_correlations.csv"))
    plot_part3_metrics_vs_accuracy(joined, figures_dir)
    log("Part 3 hardness analysis complete.")
    return {"metrics": metrics, "joined": joined, "correlations": correlations}
