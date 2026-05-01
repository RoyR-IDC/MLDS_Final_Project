"""Plotting and aggregation utilities for tile-permutation experiments.

This module provides helpers to aggregate the runner's `summary.csv`,
produce the Accuracy vs Number of Tiles plot, compute permutation difficulty
metrics for each run, and produce scatter plots of metric vs accuracy with
correlation statistics.

Usage:
    python -m src.utils.plots --summary results/tiles_experiment/summary.csv --out results/tiles_experiment/plots --n_permutations 5
"""
from typing import Optional
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.permutations import generate_permutations, identity_permutation
from src.utils.metrics import average_displacement, adjacency_preservation, displacement_entropy


def aggregate_summary(summary_csv: str) -> pd.DataFrame:
    """Read summary CSV and aggregate mean/std accuracy per (model, grid).

    Args:
        summary_csv: Path to runner summary.csv

    Returns:
        DataFrame with columns ['model','grid','mean_acc','std_acc','n_runs']
    """
    df = pd.read_csv(summary_csv)
    agg = df.groupby(['model', 'grid'])['val_acc'].agg(['mean', 'std', 'count']).reset_index()
    agg = agg.rename(columns={'mean': 'mean_acc', 'std': 'std_acc', 'count': 'n_runs'})
    return agg


def plot_accuracy_vs_tiles(summary_csv: str, out_dir: str):
    """Create Accuracy vs Number of Tiles plot (one curve per model).

    Saves `accuracy_vs_tiles.png` in `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)
    agg = aggregate_summary(summary_csv)
    models = agg['model'].unique()
    plt.figure(figsize=(8, 6))
    for m in models:
        sub = agg[agg['model'] == m].sort_values('grid')
        plt.errorbar(sub['grid'], sub['mean_acc'], yerr=sub['std_acc'], label=m, marker='o')
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

    Assumes the runner used identity permutation at index 0 and `generate_permutations`
    with `seed=gen_seed` for the remaining perms.
    """
    if perm_idx == 0:
        return identity_permutation(grid)
    perms = generate_permutations(grid, n_permutations - 1, seed=gen_seed)
    # perm_idx 1 corresponds to perms[0]
    return perms[perm_idx - 1]


def compute_metrics_for_summary(summary_csv: str, out_dir: str, n_permutations: int = 5):
    """Attach permutation metrics to each row in summary CSV and save augmented CSV.

    Args:
        summary_csv: Path to runner summary.csv
        out_dir: Directory to save augmented CSV and JSON stats
        n_permutations: Number of permutations used in runner (default 5)
    Returns:
        DataFrame with added metric columns.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_csv)
    metrics = []
    for _, row in df.iterrows():
        G = int(row['grid'])
        perm_idx = int(row['perm_idx'])
        perm = _get_permutation_for_row(G, perm_idx, n_permutations=n_permutations)
        avg_disp = average_displacement(perm, G)
        adj_pres = adjacency_preservation(perm, G)
        ent = displacement_entropy(perm, G)
        metrics.append({'avg_disp': avg_disp, 'adj_pres': adj_pres, 'disp_ent': ent})
    met_df = pd.DataFrame(metrics)
    out = pd.concat([df.reset_index(drop=True), met_df], axis=1)
    aug_path = os.path.join(out_dir, 'summary_with_metrics.csv')
    out.to_csv(aug_path, index=False)
    print('Saved', aug_path)
    return out


def plot_metric_vs_accuracy(summary_with_metrics_csv: str, out_dir: str):
    """Create scatter plots of metric vs accuracy and save correlation stats.

    Produces one scatter per metric (avg_disp, adj_pres, disp_ent) and saves a JSON
    with Pearson/Spearman correlations per model and overall.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary_with_metrics_csv)
    metrics = ['avg_disp', 'adj_pres', 'disp_ent']
    stats = {}
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(df[metric], df['val_acc'], alpha=0.6)
        # Fit linear regression
        coef = np.polyfit(df[metric], df['val_acc'], 1)
        xs = np.linspace(df[metric].min(), df[metric].max(), 100)
        ys = np.polyval(coef, xs)
        ax.plot(xs, ys, color='red', linewidth=1)
        ax.set_xlabel(metric)
        ax.set_ylabel('Validation Accuracy')
        ax.set_title(f'{metric} vs Accuracy')
        path = os.path.join(out_dir, f'{metric}_vs_accuracy.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        # Correlations overall
        pearson = float(np.corrcoef(df[metric], df['val_acc'])[0, 1])
        # Spearman via ranks
        ranks_x = df[metric].rank().values
        ranks_y = df['val_acc'].rank().values
        spearman = float(np.corrcoef(ranks_x, ranks_y)[0, 1])
        stats[metric] = {'pearson': pearson, 'spearman': spearman}
        # Per-model stats
        per_model = {}
        for m in df['model'].unique():
            sub = df[df['model'] == m]
            if len(sub) < 2:
                continue
            p_ = float(np.corrcoef(sub[metric], sub['val_acc'])[0, 1])
            rx = sub[metric].rank().values
            ry = sub['val_acc'].rank().values
            s_ = float(np.corrcoef(rx, ry)[0, 1])
            per_model[m] = {'pearson': p_, 'spearman': s_}
        stats[f'{metric}_per_model'] = per_model

    stats_path = os.path.join(out_dir, 'metric_correlation_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print('Saved', stats_path)
    return stats


def main(summary_csv: str, out_dir: str, n_permutations: int = 5):
    agg_out = os.path.join(out_dir, 'plots')
    os.makedirs(agg_out, exist_ok=True)
    plot_accuracy_vs_tiles(summary_csv, agg_out)
    df_aug = compute_metrics_for_summary(summary_csv, agg_out, n_permutations=n_permutations)
    stats = plot_metric_vs_accuracy(os.path.join(agg_out, 'summary_with_metrics.csv'), agg_out)
    return stats


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--summary', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--n_permutations', type=int, default=5)
    args = p.parse_args()
    main(args.summary, args.out, args.n_permutations)
