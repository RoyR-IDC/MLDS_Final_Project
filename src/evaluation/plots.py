"""Plotting and aggregation utilities for tile-permutation experiments.

This module provides helpers to aggregate the runner's `summary.csv`,
produce the Accuracy vs Number of Tiles plot, compute permutation difficulty
metrics for each run, and produce scatter plots of metric vs accuracy with
correlation statistics.

Usage:
    python -m src.evaluation.plots --summary results/tiles_experiment/summary.csv --out results/tiles_experiment/plots --n_permutations 5
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.permutation_difficulty import average_displacement, adjacency_preservation, displacement_entropy
from src.preprocessing.permutations import generate_permutations, identity_permutation


def aggregate_summary(summary_csv: str) -> pd.DataFrame:
    """Read summary CSV and aggregate mean/std accuracy per (model, grid).

    Args:
        summary_csv: Path to runner summary CSV.

    Returns:
        DataFrame with columns ``model``, ``grid``, ``mean_acc``, ``std_acc``, and ``n_runs``.
    """
    summary = pd.read_csv(summary_csv)
    aggregated = summary.groupby(['model', 'grid'])['val_acc'].agg(['mean', 'std', 'count']).reset_index()
    aggregated = aggregated.rename(columns={'mean': 'mean_acc', 'std': 'std_acc', 'count': 'n_runs'})
    return aggregated


def plot_accuracy_vs_tiles(summary_csv: str, out_dir: str):
    """Create Accuracy vs Number of Tiles plot.

    Args:
        summary_csv: Path to runner summary CSV.
        out_dir: Directory where ``accuracy_vs_tiles.png`` is saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    aggregated = aggregate_summary(summary_csv)
    model_names = aggregated['model'].unique()
    plt.figure(figsize=(8, 6))
    for model_name in model_names:
        model_results = aggregated[aggregated['model'] == model_name].sort_values('grid')
        plt.errorbar(model_results['grid'], model_results['mean_acc'], yerr=model_results['std_acc'], label=model_name, marker='o')
    plt.xlabel('Number of tiles (GxG)')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Number of Tiles')
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(out_dir, 'accuracy_vs_tiles.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print('Saved', out_path)


def _get_permutation_for_row(grid: int, perm_idx: int, n_permutations: int = 5, gen_seed: int = 42):
    """Reconstruct permutation used by the runner for a given grid and perm_idx.

    Args:
        grid: Number of tiles along each image side.
        perm_idx: Permutation index from the legacy summary.
        n_permutations: Number of permutations used by the legacy runner.
        gen_seed: Seed used by the legacy runner.

    Returns:
        Tile permutation for the summary row.
    """
    if perm_idx == 0:
        permutation = identity_permutation(grid)
        return permutation
    permutations = generate_permutations(grid, n_permutations - 1, seed=gen_seed)
    # Permutation index 1 corresponds to the first generated permutation.
    permutation = permutations[perm_idx - 1]
    return permutation


def compute_metrics_for_summary(summary_csv: str, out_dir: str, n_permutations: int = 5):
    """Attach permutation metrics to each row in summary CSV and save augmented CSV.

    Args:
        summary_csv: Path to runner summary CSV.
        out_dir: Directory to save augmented CSV and JSON stats.
        n_permutations: Number of permutations used by the runner.

    Returns:
        DataFrame with added metric columns.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = pd.read_csv(summary_csv)
    metric_rows = []
    for _, row in summary.iterrows():
        grid_size = int(row['grid'])
        perm_idx = int(row['perm_idx'])
        permutation = _get_permutation_for_row(grid_size, perm_idx, n_permutations=n_permutations)
        average_tile_displacement = average_displacement(permutation, grid_size)
        adjacency_score = adjacency_preservation(permutation, grid_size)
        entropy = displacement_entropy(permutation, grid_size)
        metric_rows.append({'avg_disp': average_tile_displacement, 'adj_pres': adjacency_score, 'disp_ent': entropy})
    metric_table = pd.DataFrame(metric_rows)
    output_table = pd.concat([summary.reset_index(drop=True), metric_table], axis=1)
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
    summary = pd.read_csv(summary_with_metrics_csv)
    metrics = ['avg_disp', 'adj_pres', 'disp_ent']
    stats = {}
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(summary[metric], summary['val_acc'], alpha=0.6)
        # Fit linear regression
        coefficient = np.polyfit(summary[metric], summary['val_acc'], 1)
        x_values = np.linspace(summary[metric].min(), summary[metric].max(), 100)
        y_values = np.polyval(coefficient, x_values)
        ax.plot(x_values, y_values, color='red', linewidth=1)
        ax.set_xlabel(metric)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title(f'{metric} vs Accuracy')
        path = os.path.join(out_dir, f'{metric}_vs_accuracy.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        # Correlations overall
        pearson = float(np.corrcoef(summary[metric], summary['val_acc'])[0, 1])
        # Spearman via ranks
        ranks_x = summary[metric].rank().values
        ranks_y = summary['val_acc'].rank().values
        spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        stats[metric] = {'pearson': pearson, 'spearman': spearman}
        # Per-model stats
        per_model = {}
        for model_name in summary['model'].unique():
            model_summary = summary[summary['model'] == model_name]
            if len(model_summary) < 2:
                continue
            model_pearson = float(np.corrcoef(model_summary[metric], model_summary['val_acc'])[0, 1])
            model_ranks_x = model_summary[metric].rank().values
            model_ranks_y = model_summary['val_acc'].rank().values
            model_spearman = float(np.corrcoef(model_ranks_x, model_ranks_y)[0, 1])
            per_model[model_name] = {'pearson': model_pearson, 'spearman': model_spearman}
        stats[f'{metric}_per_model'] = per_model

    stats_path = os.path.join(out_dir, 'metric_correlation_stats.json')
    with open(stats_path, 'w') as handle:
        json.dump(stats, handle, indent=2)
    print('Saved', stats_path)
    return stats


def main(summary_csv: str, out_dir: str, n_permutations: int = 5):
    """Run all legacy summary plotting steps.

    Args:
        summary_csv: Path to runner summary CSV.
        out_dir: Base output directory.
        n_permutations: Number of permutations used by the runner.

    Returns:
        Dictionary with metric correlation statistics.
    """

    agg_out = os.path.join(out_dir, 'plots')
    os.makedirs(agg_out, exist_ok=True)
    plot_accuracy_vs_tiles(summary_csv, agg_out)
    augmented_summary = compute_metrics_for_summary(summary_csv, agg_out, n_permutations=n_permutations)
    stats = plot_metric_vs_accuracy(os.path.join(agg_out, 'summary_with_metrics.csv'), agg_out)
    del augmented_summary
    correlation_stats = stats
    return correlation_stats


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--n_permutations', type=int, default=5)
    args = p.parse_args()
    main(args.summary, args.out, args.n_permutations)
