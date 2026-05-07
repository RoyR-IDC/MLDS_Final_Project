"""Shared evaluation helpers for experiment outputs."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.preprocessing.dogs_cats import Sample, discover_samples, stratified_split
from src.preprocessing.permutations import PermutationRecord
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir, save_csv


def get_device(config: CVExperimentConfig) -> torch.device:
    """Return a configured device, falling back to CPU when needed.

    Args:
        config: Normalized experiment configuration.

    Returns:
        Torch device selected from the config.
    """

    requested = str(config.device)
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    device = torch.device(requested)
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
            mean_val_accuracy=("val_accuracy", "mean"),
            std_val_accuracy=("val_accuracy", "std"),
            mean_best_val_accuracy=("best_val_accuracy", "mean"),
            std_best_val_accuracy=("best_val_accuracy", "std"),
            n_runs=("val_accuracy", "count"),
        )
        .reset_index()
    )
    return aggregated_results


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
            group["mean_best_val_accuracy"],
            yerr=group["std_best_val_accuracy"].fillna(0.0),
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
            sorted_group["mean_best_val_accuracy"],
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
