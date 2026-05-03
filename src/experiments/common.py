"""Shared experiment helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data.dogs_cats import discover_samples, stratified_split
from src.utils.io import ensure_dir


def get_device(config: Dict) -> torch.device:
    """Return a configured device, falling back to CPU when needed."""

    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_experiment_samples(config: Dict, seed: int):
    """Discover and split configured Dogs vs Cats samples."""

    samples = discover_samples(config["data_dir"], limit=config.get("sample_limit"))
    return stratified_split(
        samples,
        val_fraction=float(config.get("val_fraction", 0.2)),
        test_fraction=float(config.get("test_fraction", 0.0)),
        seed=seed,
    )


def aggregate_accuracy(raw_results: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate accuracy columns for repeated experiment runs."""

    return (
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


def plot_accuracy_vs_tiles(aggregated: pd.DataFrame, output_path: str, model_column: str = "model_name") -> None:
    """Save an accuracy-vs-number-of-tiles plot."""

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

