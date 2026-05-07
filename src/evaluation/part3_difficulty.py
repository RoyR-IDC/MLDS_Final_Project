"""Part 3 permutation difficulty analysis."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.permutation_difficulty import permutation_metric_row
from src.preprocessing.permutations import generate_permutations, identity_permutation
from src.utils.config import load_experiment_config, normalize_config
from src.utils.io import ensure_dir, save_csv


def _load_part1_permutations(config: Dict) -> pd.DataFrame:
    """Load or recreate the permutation metadata used by Part 1.

    Args:
        config: Normalized Part 3 configuration.

    Returns:
        DataFrame containing one row per permutation.
    """

    permutation_csv = config.get("permutation_csv")
    if permutation_csv and os.path.exists(permutation_csv):
        permutation_table = pd.read_csv(permutation_csv)
        return permutation_table
    rows = []
    seed = int(config.get("seed", 42))
    for grid_size in [int(value) for value in config.get("grid_sizes", [1, 2, 3, 4])]:
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
            generate_permutations(grid_size, int(config.get("num_permutations", 2)), seed=seed),
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
    permutation_table = pd.DataFrame(rows)
    return permutation_table


def compute_permutation_metrics(config: Dict) -> pd.DataFrame:
    """Compute model-agnostic metrics for Part 1 permutations.

    Args:
        config: Grouped or flat Part 3 analysis configuration.

    Returns:
        DataFrame containing one metric row per permutation.
    """

    config = normalize_config(config)
    rows: List[dict] = []
    permutations = _load_part1_permutations(config)
    for _, row in permutations.iterrows():
        permutation = json.loads(row["permutation"]) if isinstance(row["permutation"], str) else row["permutation"]
        grid_size = int(row["grid_size"])
        rows.append(
            {
                "grid_size": grid_size,
                "num_tiles": grid_size * grid_size,
                "permutation_id": int(row["permutation_id"]),
                "permutation_seed": row.get("permutation_seed"),
                **permutation_metric_row(permutation, grid_size),
            }
        )
    metric_table = pd.DataFrame(rows)
    return metric_table


def _correlations(joined: pd.DataFrame) -> pd.DataFrame:
    """Compute metric-to-accuracy correlations overall and by model.

    Args:
        joined: Part 1 accuracy rows joined with permutation metrics.

    Returns:
        DataFrame of Pearson and Spearman correlations.
    """

    metric_columns = [
        "average_displacement",
        "normalized_average_displacement",
        "adjacency_preservation",
        "locality_disruption",
        "displacement_entropy",
        "combined_difficulty",
    ]
    rows: List[dict] = []
    groups = [("overall", joined)] + [(str(model), frame) for model, frame in joined.groupby("model_name")]
    for group_name, frame in groups:
        for metric in metric_columns:
            if len(frame) < 2 or frame[metric].nunique() < 2 or frame["best_val_accuracy"].nunique() < 2:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(frame[metric].corr(frame["best_val_accuracy"], method="pearson"))
                spearman = float(frame[metric].corr(frame["best_val_accuracy"], method="spearman"))
            rows.append({"group": group_name, "metric": metric, "pearson": pearson, "spearman": spearman, "n": len(frame)})
    correlation_table = pd.DataFrame(rows)
    return correlation_table


def _plot_metric_vs_accuracy(joined: pd.DataFrame, figures_dir: str) -> None:
    """Save scatter plots comparing selected difficulty metrics with accuracy.

    Args:
        joined: Part 1 accuracy rows joined with permutation metrics.
        figures_dir: Directory where plot files are written.
    """

    for metric in ["normalized_average_displacement", "adjacency_preservation", "combined_difficulty"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for model_name, group in joined.groupby("model_name"):
            ax.scatter(group[metric], group["best_val_accuracy"], label=model_name, alpha=0.8)
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("Best validation accuracy")
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, f"part3_{metric}_vs_accuracy.png"), dpi=160)
        plt.close(fig)


class Part3DifficultyAnalysis:
    """Notebook-friendly runner for Part 3 permutation difficulty analysis.

    Args:
        config: Grouped or flat Part 3 analysis configuration.
    """

    def __init__(self, config: Dict) -> None:
        self.config = normalize_config(config)
        self.results_dir = ensure_dir(self.config.get("results_dir", "outputs/results"))
        self.figures_dir = ensure_dir(self.config.get("figures_dir", "outputs/figures"))

    def load_data(self) -> pd.DataFrame:
        """Load the Part 1 raw result table used for accuracy comparison."""

        part1_results_csv = self.config.get("part1_results_csv", os.path.join(self.results_dir, "part1_raw_results.csv"))
        raw_results = pd.read_csv(part1_results_csv)
        return raw_results

    def load_results(self) -> Dict[str, pd.DataFrame]:
        """Load saved Part 3 metric, joined, and correlation tables."""

        result_tables = {
            "metrics": pd.read_csv(os.path.join(self.results_dir, "permutation_metrics.csv")),
            "joined": pd.read_csv(os.path.join(self.results_dir, "metric_accuracy_joined.csv")),
            "correlations": pd.read_csv(os.path.join(self.results_dir, "metric_accuracy_correlations.csv")),
        }
        return result_tables

    def display_outputs(self) -> Dict[str, object]:
        """Return saved output paths for notebook display cells."""

        output_paths = {
            "metrics": os.path.join(self.results_dir, "permutation_metrics.csv"),
            "joined": os.path.join(self.results_dir, "metric_accuracy_joined.csv"),
            "correlations": os.path.join(self.results_dir, "metric_accuracy_correlations.csv"),
            "plots": sorted(
                os.path.join(self.figures_dir, filename)
                for filename in os.listdir(self.figures_dir)
                if filename.startswith("part3_") and filename.endswith("_vs_accuracy.png")
            )
            if os.path.isdir(self.figures_dir)
            else [],
        }
        return output_paths

    def run(self) -> Dict[str, pd.DataFrame]:
        """Run metric computation, accuracy join, correlations, and plots."""

        # Metrics are computed only from permutations and tile coordinates.
        metrics = compute_permutation_metrics(self.config)
        save_csv(metrics, os.path.join(self.results_dir, "permutation_metrics.csv"))

        # Accuracy is joined only after metric computation for analysis.
        raw_results = self.load_data()
        joined = raw_results.merge(metrics, on=["grid_size", "num_tiles", "permutation_id"], how="left")
        save_csv(joined, os.path.join(self.results_dir, "metric_accuracy_joined.csv"))

        correlations = _correlations(joined)
        save_csv(correlations, os.path.join(self.results_dir, "metric_accuracy_correlations.csv"))
        _plot_metric_vs_accuracy(joined, self.figures_dir)
        result_tables = {"metrics": metrics, "joined": joined, "correlations": correlations}
        return result_tables


def run_part3(config: Dict) -> Dict[str, pd.DataFrame]:
    """Run Part 3 metric computation, join, correlations, and plots."""

    analysis = Part3DifficultyAnalysis(config)
    result_tables = analysis.run()
    return result_tables


def main(config_path: str) -> Dict[str, pd.DataFrame]:
    """Run Part 3 from a YAML config path."""

    config = load_experiment_config(config_path)
    result_tables = run_part3(config)
    return result_tables


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
