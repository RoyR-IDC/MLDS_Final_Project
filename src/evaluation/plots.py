"""Plotting and aggregation utilities for tile-permutation experiments.

This module provides helpers to aggregate the runner's `summary.csv`,
produce the Accuracy vs Number of Tiles plot, compute tile-permutation difficulty
metrics for each run, and produce scatter plots of metric vs accuracy with
correlation statistics.

Usage:
    python -m src.evaluation.plots --summary results/tiles_experiment/summary.csv --out results/tiles_experiment/plots --num_tile_permutations 5
"""
from __future__ import annotations

import os
import json
from typing import Any, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_center_weighted_displacement,
    compute_combined_hardness,
    compute_global_displacement,
)
from src.preprocessing.tile_permutations import (
    generate_tile_permutations,
    tile_permutation_from_jsonable,
)


def _read_csv_dataframe(path: str) -> pd.DataFrame:
    """Read a CSV as a DataFrame with a narrowed static type."""

    return cast(pd.DataFrame, pd.read_csv(path))


def _as_axes(axis: object) -> Axes:
    """Narrow matplotlib's broad ``subplots`` return type for single-axis plots."""

    return cast(Axes, axis)


def _sorted_dataframe(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Return a sorted DataFrame with a narrowed static type."""

    return cast(pd.DataFrame, frame.sort_values(by))


def aggregate_summary(summary_csv: str) -> pd.DataFrame:
    """Read summary CSV and aggregate mean/std accuracy per model and tile count.

    Args:
        summary_csv: Path to runner summary CSV.

    Returns:
        DataFrame with columns ``model_name``, ``tiles_per_side``, accuracy stats, and ``n_runs``.
    """
    summary = _read_csv_dataframe(summary_csv)
    aggregated = cast(
        pd.DataFrame,
        cast(Any, summary).groupby(['model_name', 'num_tiles'], dropna=False)['val_accuracy'].agg(
            ['mean', 'std', 'count']
        ).reset_index(),
    )
    return cast(
        pd.DataFrame,
        aggregated.rename(columns={'mean': 'mean_val_accuracy', 'std': 'std_val_accuracy', 'count': 'n_runs'}),
    )


def plot_accuracy_vs_tiles(summary_csv: str, out_dir: str):
    """Create Accuracy vs Number of Tiles plot.

    Args:
        summary_csv: Path to runner summary CSV.
        out_dir: Directory where ``accuracy_vs_tiles.png`` is saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    aggregated = aggregate_summary(summary_csv)
    aggregated_any = cast(Any, aggregated)
    model_names = aggregated_any['model_name'].unique()
    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        model_results = _sorted_dataframe(
            cast(pd.DataFrame, aggregated_any[aggregated_any['model_name'] == model_name]),
            'num_tiles',
        )
        plt.errorbar(
            model_results['num_tiles'],
            model_results['mean_val_accuracy'],
            yerr=model_results['std_val_accuracy'],
            label=model_name,
            marker='o',
        )
    plt.xlabel('Number of tiles')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Number of Tiles')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(out_dir, 'accuracy_vs_tiles.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print('Saved', out_path)


def _get_tile_permutation_for_row(
    tiles_per_side: int | None,
    permutation_idx: int,
    n_tile_permutations: int = 5,
    gen_seed: int = 42,
):
    """Reconstruct the generated tile permutation for one summary row.

    Args:
        tiles_per_side: Number of tiles along each image side.
        permutation_idx: Tile-permutation ID from the summary.
        n_tile_permutations: Number of tile permutations used by the runner.
        gen_seed: Seed used by the runner.

    Returns:
        Output tile permutation for the summary row.
    """
    if tiles_per_side is None or permutation_idx == 0:
        return None
    tile_permutations = generate_tile_permutations(tiles_per_side, n_tile_permutations - 1, seed=gen_seed)
    return tile_permutations[permutation_idx - 1]


def compute_metrics_for_summary(summary_csv: str, out_dir: str, num_tile_permutations: int = 5):
    """Attach tile-permutation metrics to each row in summary CSV and save augmented CSV.

    Args:
        summary_csv: Path to runner summary CSV.
        out_dir: Directory to save augmented CSV and JSON stats.
        num_tile_permutations: Number of tile permutations used by the runner.

    Returns:
        DataFrame with added metric columns.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = _read_csv_dataframe(summary_csv)
    metric_rows = []
    for _, row in summary.iterrows():
        raw_tiles_per_side = cast(Any, row.get('tiles_per_side'))
        tiles_per_side = None if bool(pd.isna(raw_tiles_per_side)) else int(raw_tiles_per_side)
        raw_tile_permutation = cast(Any, row.get('tile_permutation'))
        if 'tile_permutation' in row and not bool(pd.isna(raw_tile_permutation)):
            serialized_tile_permutation = json.loads(cast(str, raw_tile_permutation))
            tile_permutation = tile_permutation_from_jsonable(serialized_tile_permutation, tiles_per_side)
        else:
            tile_permutation = _get_tile_permutation_for_row(
                tiles_per_side,
                int(cast(Any, row['tile_permutation_id'])),
                n_tile_permutations=num_tile_permutations,
            )
        metric_rows.append(
            {
                'global_tile_displacement': compute_global_displacement(tile_permutation, tiles_per_side),
                'center_weighted_displacement': compute_center_weighted_displacement(
                    tile_permutation,
                    tiles_per_side,
                    alpha_center=1.0,
                ),
                'adjacency_destruction_hardness': compute_adjacency_destruction_hardness(
                    tile_permutation,
                    tiles_per_side,
                ),
                'combined_hardness_score': compute_combined_hardness(
                    tile_permutation,
                    tiles_per_side,
                    alpha_center=1.0,
                ),
            }
        )
    metric_table = pd.DataFrame(metric_rows)
    output_table = cast(pd.DataFrame, pd.concat([summary.reset_index(drop=True), metric_table], axis=1))
    aug_path = os.path.join(out_dir, 'summary_with_metrics.csv')
    output_table.to_csv(aug_path, index=False)
    print('Saved', aug_path)
    return output_table


def plot_metric_vs_accuracy(summary_with_metrics_csv: str, out_dir: str):
    """Create scatter plots of metric vs accuracy and save correlation stats.

    Args:
        summary_with_metrics_csv: Path to summary CSV with metric columns.
        out_dir: Directory for figures and correlation JSON.

    Returns:
        Dictionary with Pearson and Spearman correlations.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = _read_csv_dataframe(summary_with_metrics_csv)
    metrics = [
        'global_tile_displacement',
        'center_weighted_displacement',
        'adjacency_destruction_hardness',
        'combined_hardness_score',
    ]
    stats = {}
    for metric in metrics:
        fig, axis = plt.subplots(figsize=(6, 5))
        ax = _as_axes(axis)
        summary_any = cast(Any, summary)
        ax.scatter(summary_any[metric], summary_any['val_accuracy'], alpha=0.6)
        # Fit linear regression
        coefficient = np.polyfit(summary_any[metric], summary_any['val_accuracy'], 1)
        x_values = np.linspace(float(summary_any[metric].min()), float(summary_any[metric].max()), 100)
        y_values = np.polyval(coefficient, x_values)
        ax.plot(x_values, y_values, color='red', linewidth=1)
        ax.set_xlabel(metric)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title(f'{metric} vs Accuracy')
        path = os.path.join(out_dir, f'{metric}_vs_accuracy.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        # Correlations overall
        pearson = float(np.corrcoef(summary_any[metric], summary_any['val_accuracy'])[0, 1])
        # Spearman via ranks
        ranks_x = summary_any[metric].rank().values
        ranks_y = summary_any['val_accuracy'].rank().values
        spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        stats[metric] = {'pearson': pearson, 'spearman': spearman}
        # Per-model stats
        per_model = {}
        for model_name in summary_any['model_name'].unique():
            model_summary = cast(pd.DataFrame, summary_any[summary_any['model_name'] == model_name])
            if len(model_summary) < 2:
                continue
            model_summary_any = cast(Any, model_summary)
            model_pearson = float(np.corrcoef(model_summary_any[metric], model_summary_any['val_accuracy'])[0, 1])
            model_ranks_x = model_summary_any[metric].rank().values
            model_ranks_y = model_summary_any['val_accuracy'].rank().values
            model_spearman = float(np.corrcoef(model_ranks_x, model_ranks_y)[0, 1])
            per_model[model_name] = {'pearson': model_pearson, 'spearman': model_spearman}
        stats[f'{metric}_per_model'] = per_model

    stats_path = os.path.join(out_dir, 'metric_correlation_stats.json')
    with open(stats_path, 'w') as handle:
        json.dump(stats, handle, indent=2)
    print('Saved', stats_path)
    return stats


def main(summary_csv: str, out_dir: str, num_tile_permutations: int = 5):
    """Run all summary plotting steps.

    Args:
        summary_csv: Path to runner summary CSV.
        out_dir: Base output directory.
        num_tile_permutations: Number of tile permutations used by the runner.

    Returns:
        Dictionary with metric correlation statistics.
    """

    agg_out = os.path.join(out_dir, 'plots')
    os.makedirs(agg_out, exist_ok=True)
    plot_accuracy_vs_tiles(summary_csv, agg_out)
    augmented_summary = compute_metrics_for_summary(summary_csv, agg_out, num_tile_permutations=num_tile_permutations)
    stats = plot_metric_vs_accuracy(os.path.join(agg_out, 'summary_with_metrics.csv'), agg_out)
    del augmented_summary
    correlation_stats = stats
    return correlation_stats


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--num_tile_permutations', type=int, default=5)
    args = p.parse_args()
    main(args.summary, args.out, args.num_tile_permutations)
