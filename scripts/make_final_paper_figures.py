"""Regenerate final-paper figures from saved CSV artifacts.

This script is presentation-only: it reads existing saved results under
``outputs/results`` and writes refreshed figures under ``outputs/figures``.
It does not train models or alter any numerical result files.
"""

from __future__ import annotations

from pathlib import Path
import ast
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"
INTERMEDIATE_DIR = FIGURES_DIR / "intermediate"
DATA_DIR = ROOT / "data" / "dogs-vs-cats" / "train"

MODEL_LABELS = {
    "mobilenetv3_small": "MobileNetV3-Small",
    "deit_tiny": "DeiT-Tiny",
    "gmlp_s16": "gMLP-S16",
}
CONDITION_LABELS = {
    "frozen_pretrained_binary_head": "Frozen pretrained binary head",
    "unfrozen_pretrained_binary_head": "Unfrozen pretrained binary head",
    "zero_shot_full_pretrained_head": "Zero-shot native ImageNet-1k head",
}
ABLATION_LABELS = {
    "regular_part1": "Regular Part 1 reference",
    "augmentation_patch_shuffle": "Patch shuffle",
    "regular_augmentations": "Regular augmentations",
    "mixed_original_permuted": "Mixed original/permuted",
    "cnn_mlp_head": "CNN MLP head",
    "curriculum_corruption_probability": "Corruption-probability curriculum",
    "curriculum_permutation_difficulty": "Permutation-difficulty curriculum",
}
GRID_ORDER = [1, 16, 49, 100]
GRID_LABELS = {
    1: "No permutation\n/ 1 tile",
    16: "4 × 4\n/ 16 tiles",
    49: "7 × 7\n/ 49 tiles",
    100: "10 × 10\n/ 100 tiles",
}
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DIFFICULTY_LABELS = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
DIFFICULTY_COLORS = {"easy": "#0072B2", "medium": "#E69F00", "hard": "#009E73", "baseline": "#666666"}
DIFFICULTY_MARKERS = {"easy": "o", "medium": "s", "hard": "^", "baseline": "D"}
MODEL_COLORS = {"mobilenetv3_small": "#0072B2", "deit_tiny": "#D55E00", "gmlp_s16": "#009E73"}
MODEL_MARKERS = {"mobilenetv3_small": "o", "deit_tiny": "D", "gmlp_s16": "P"}
CONDITION_LINESTYLES = {
    "frozen_pretrained_binary_head": "-",
    "unfrozen_pretrained_binary_head": "--",
    "zero_shot_full_pretrained_head": ":",
}
GRID_COLORS = {1: "#555555", 16: "#0072B2", 49: "#CC79A7", 100: "#009E73"}
STRATEGY_COLORS = {
    "augmentation_patch_shuffle": "#0072B2",
    "regular_augmentations": "#E69F00",
    "mixed_original_permuted": "#009E73",
    "cnn_mlp_head": "#D55E00",
    "curriculum_corruption_probability": "#CC79A7",
    "curriculum_permutation_difficulty": "#56B4E9",
}
STRATEGY_MARKERS = {
    "augmentation_patch_shuffle": "o",
    "regular_augmentations": "D",
    "mixed_original_permuted": "P",
    "cnn_mlp_head": "X",
    "curriculum_corruption_probability": "v",
    "curriculum_permutation_difficulty": "*",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 17,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _tile_position(num_tiles: int) -> int:
    return GRID_ORDER.index(int(num_tiles))


def _format_percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0f}")


def _set_xticks(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(GRID_ORDER)))
    ax.set_xticklabels([GRID_LABELS[v] for v in GRID_ORDER])


def _parse_permutation(value: object) -> list[list[tuple[int, int]]] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    parsed = ast.literal_eval(str(value))
    order = []
    for row in parsed:
        parsed_row = []
        for item in row:
            if isinstance(item, str):
                parsed_row.append(tuple(ast.literal_eval(item)))
            else:
                parsed_row.append(tuple(item))
        order.append(parsed_row)
    return order


def _resize_square(image: Image.Image, side: int) -> Image.Image:
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    return image.convert("RGB").resize((side, side), resample)


def _apply_tile_permutation(image: Image.Image, order: list[list[tuple[int, int]]] | None) -> Image.Image:
    if order is None:
        return image
    side = image.width
    tiles_per_side = len(order)
    tile_size = side // tiles_per_side
    out = Image.new("RGB", (side, side))
    for new_row, row in enumerate(order):
        for new_col, (old_row, old_col) in enumerate(row):
            tile = image.crop(
                (
                    old_col * tile_size,
                    old_row * tile_size,
                    (old_col + 1) * tile_size,
                    (old_row + 1) * tile_size,
                )
            )
            out.paste(tile, (new_col * tile_size, new_row * tile_size))
    return out


def make_figure_1() -> None:
    metrics = pd.read_csv(RESULTS_DIR / "tile_permutation_metrics.csv")
    permutations = pd.read_csv(RESULTS_DIR / "part1_tile_permutations.csv")
    permutations["num_tiles"] = permutations["tiles_per_side"].apply(
        lambda value: 1 if pd.isna(value) else int(value) * int(value)
    )
    candidate_paths = [
        DATA_DIR / "dog.2489 2.jpg",
        DATA_DIR / "cat.991.jpg",
        *sorted(DATA_DIR.glob("dog.*.jpg")),
    ]
    image_path = next(path for path in candidate_paths if path.exists())
    base = _resize_square(Image.open(image_path), 310)

    fig, axes = plt.subplots(3, 4, figsize=(14.2, 9.6), constrained_layout=True)
    for row_index, difficulty in enumerate(DIFFICULTY_ORDER):
        for col_index, num_tiles in enumerate(GRID_ORDER):
            ax = axes[row_index, col_index]
            if num_tiles == 1 and row_index != 1:
                ax.axis("off")
                continue
            record = permutations[
                (permutations["num_tiles"].astype(int) == num_tiles)
                & (permutations["tile_permutation_name"] == difficulty)
            ].iloc[0]
            order = _parse_permutation(record["tile_permutation"])
            shown = _apply_tile_permutation(base, order)
            ax.imshow(shown)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0 or num_tiles == 1:
                ax.set_title(GRID_LABELS[num_tiles].replace("\n", " "), pad=10)
            if col_index == 1:
                ax.set_ylabel(DIFFICULTY_LABELS[difficulty], rotation=90, labelpad=18, weight="bold")
            if num_tiles == 1:
                ax.text(
                    0.5,
                    -0.055,
                    "Shared unpermuted baseline",
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=11,
                )
                continue
            metric = metrics[
                (metrics["num_tiles"].astype(int) == num_tiles)
                & (metrics["tile_permutation_name"] == difficulty)
            ].iloc[0]
            metric_text = (
                f"A={metric['adjacency_destruction_hardness']:.2f}, "
                f"D={metric['global_tile_displacement']:.2f}, "
                f"E={metric['spatial_permutation_entropy']:.2f}, "
                f"C={metric['combined_hardness_score']:.2f}"
            )
            ax.text(0.5, -0.055, metric_text, ha="center", va="top", transform=ax.transAxes, fontsize=10.5)
    fig.suptitle("Illustrative Deterministic Tile Permutations", y=1.02)
    _save(fig, "part3_hardness_examples_readable")


def make_figure_2() -> None:
    df = pd.read_csv(RESULTS_DIR / "part1_aggregated_results.csv")
    df["accuracy_percent"] = 100.0 * df["mean_best_epoch_val_accuracy"]
    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    for condition in CONDITION_LABELS:
        for model_name in MODEL_LABELS:
            group = df[
                (df["experiment_condition"] == condition)
                & (df["model_name"] == model_name)
            ].sort_values("num_tiles")
            if group.empty:
                continue
            x = [_tile_position(v) for v in group["num_tiles"]]
            ax.plot(
                x,
                group["accuracy_percent"],
                color=MODEL_COLORS[model_name],
                marker=MODEL_MARKERS[model_name],
                linestyle=CONDITION_LINESTYLES[condition],
                linewidth=2.5,
                markersize=7.8,
                alpha=0.92,
            )
    _set_xticks(ax)
    ax.set_ylabel("Mean best validation accuracy (%)")
    ax.set_xlabel("Grid size")
    ax.set_ylim(40, 101)
    ax.grid(True, axis="y", alpha=0.28)
    _format_percent_axis(ax)
    model_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[name],
            marker=MODEL_MARKERS[name],
            linestyle="-",
            linewidth=2.5,
            markersize=7.8,
            label=label,
        )
        for name, label in MODEL_LABELS.items()
    ]
    condition_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=CONDITION_LINESTYLES[name],
            linewidth=2.5,
            label=label,
        )
        for name, label in CONDITION_LABELS.items()
    ]
    first_legend = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.33, -0.22),
        ncol=3,
        frameon=False,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=condition_handles,
        title="Experiment condition",
        loc="upper center",
        bbox_to_anchor=(0.78, -0.22),
        ncol=1,
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, "part1_combined_comparison_readable")


def _plot_control(condition: str, stem: str, title: str, y_min: float) -> None:
    df = pd.read_csv(RESULTS_DIR / "part1_aggregated_results.csv")
    df = df[df["experiment_condition"] == condition].copy()
    df["accuracy_percent"] = 100.0 * df["mean_best_epoch_val_accuracy"]
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    for model_name in MODEL_LABELS:
        group = df[df["model_name"] == model_name].sort_values("num_tiles")
        if group.empty:
            continue
        x = [_tile_position(v) for v in group["num_tiles"]]
        ax.plot(
            x,
            group["accuracy_percent"],
            label=MODEL_LABELS[model_name],
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linestyle=CONDITION_LINESTYLES[condition],
            linewidth=2.5 if len(group) > 1 else 0,
            markersize=8,
        )
    ax.set_title(title)
    _set_xticks(ax)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Mean best validation accuracy (%)")
    ax.set_xlim(-0.15, len(GRID_ORDER) - 0.85)
    ax.set_ylim(y_min, 100)
    ax.grid(True, axis="y", alpha=0.28)
    _format_percent_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    _save(fig, stem)


def make_figure_3() -> None:
    _plot_control(
        "frozen_pretrained_binary_head",
        "part1_accuracy_vs_tiles_readable",
        "Main experiment: frozen pretrained binary head",
        74,
    )
    _plot_control(
        "unfrozen_pretrained_binary_head",
        "part1_control_unfrozen_pretrained_binary_head_readable",
        "Control experiment: unfrozen pretrained binary head",
        88,
    )
    _plot_control(
        "zero_shot_full_pretrained_head",
        "part1_control_zero_shot_full_pretrained_head_readable",
        "Control experiment: zero-shot native ImageNet-1k head",
        42,
    )


def make_figure_4() -> None:
    raw = pd.read_csv(RESULTS_DIR / "part2_raw_results.csv")
    raw = raw[raw["model_name"] == "mobilenetv3_small"].copy()
    baseline = raw[raw["ablation_name"] == "regular_part1"][
        ["num_tiles", "tile_permutation_name", "best_val_accuracy"]
    ].rename(columns={"best_val_accuracy": "baseline_best_val_accuracy"})
    ablations = raw[raw["ablation_name"] != "regular_part1"].copy()
    joined = ablations.merge(baseline, on=["num_tiles", "tile_permutation_name"], how="left")
    joined["delta_pp"] = 100.0 * (joined["best_val_accuracy"] - joined["baseline_best_val_accuracy"])

    ablation_order = [name for name in ABLATION_LABELS if name != "regular_part1"]
    grid_offsets = {1: -0.27, 16: -0.09, 49: 0.09, 100: 0.27}
    fig, ax = plt.subplots(figsize=(13.8, 6.2))
    for num_tiles in GRID_ORDER:
        for difficulty in DIFFICULTY_ORDER:
            group = joined[
                (joined["num_tiles"].astype(int) == num_tiles)
                & (joined["tile_permutation_name"] == difficulty)
            ]
            x = [ablation_order.index(name) + grid_offsets[num_tiles] for name in group["ablation_name"]]
            ax.scatter(
                x,
                group["delta_pp"],
                color=GRID_COLORS[num_tiles],
                marker=DIFFICULTY_MARKERS[difficulty],
                s=74,
                alpha=0.88,
                edgecolor="#222222",
                linewidth=0.55,
            )
    ax.axhline(0, color="#333333", linewidth=1.2)
    ax.set_xticks(range(len(ablation_order)))
    ax.set_xticklabels([ABLATION_LABELS[name] for name in ablation_order], rotation=22, ha="right")
    ax.set_ylabel("Best validation accuracy delta (percentage points)")
    ax.set_xlabel("Part 2 ablation")
    ax.set_title("Delta from regular Part 1 reference baseline")
    ax.grid(True, axis="y", alpha=0.28)
    difficulty_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=DIFFICULTY_MARKERS[d],
            linestyle="None",
            markersize=8,
            label=DIFFICULTY_LABELS[d],
        )
        for d in DIFFICULTY_ORDER
    ]
    grid_handles = [
        Line2D(
            [0],
            [0],
            color=GRID_COLORS[num_tiles],
            marker="o",
            linestyle="None",
            markersize=8,
            label=GRID_LABELS[num_tiles].replace("\n", " "),
        )
        for num_tiles in GRID_ORDER
    ]
    first_legend = ax.legend(
        handles=difficulty_handles,
        title="Difficulty",
        loc="center left",
        bbox_to_anchor=(1.01, 0.68),
        frameon=False,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=grid_handles,
        title="Grid size",
        loc="center left",
        bbox_to_anchor=(1.01, 0.27),
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, "part2_ablation_delta_by_strategy_readable")

    fig, ax = plt.subplots(figsize=(13.4, 6.0))
    averaged = (
        joined.groupby(["ablation_name", "num_tiles"], as_index=False)["delta_pp"]
        .mean()
        .sort_values(["ablation_name", "num_tiles"])
    )
    for ablation_name in ablation_order:
        group = averaged[averaged["ablation_name"] == ablation_name].sort_values("num_tiles")
        ax.plot(
            [_tile_position(v) for v in group["num_tiles"]],
            group["delta_pp"],
            color=STRATEGY_COLORS[ablation_name],
            marker=STRATEGY_MARKERS[ablation_name],
            linewidth=2.3,
            markersize=7.7,
            label=ABLATION_LABELS[ablation_name],
        )
    ax.axhline(0, color="#333333", linewidth=1.2)
    _set_xticks(ax)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Mean best validation accuracy delta (percentage points)")
    ax.set_title("Average delta by grid size")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, "part2_ablation_delta_by_grid_readable")


def make_figure_5() -> None:
    raw = pd.read_csv(RESULTS_DIR / "part2_raw_results.csv")
    raw = raw[raw["model_name"] == "mobilenetv3_small"].copy()
    y_min = 100.0 * raw["best_val_accuracy"].min() - 2.0
    y_max = 100.0 * raw["best_val_accuracy"].max() + 1.0
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    for ablation_name, label in ABLATION_LABELS.items():
        if ablation_name == "regular_part1":
            continue
        fig, ax = plt.subplots(figsize=(8.2, 5.1))
        frame = raw[raw["ablation_name"].isin(["regular_part1", ablation_name])].copy()
        for difficulty in DIFFICULTY_ORDER:
            for setting_name, linestyle in [("regular_part1", "--"), (ablation_name, "-")]:
                group = frame[
                    (frame["ablation_name"] == setting_name)
                    & (frame["tile_permutation_name"] == difficulty)
                ].sort_values("num_tiles")
                ax.plot(
                    [_tile_position(v) for v in group["num_tiles"]],
                    100.0 * group["best_val_accuracy"],
                    color=DIFFICULTY_COLORS[difficulty],
                    marker=DIFFICULTY_MARKERS[difficulty],
                    linestyle=linestyle,
                    linewidth=2.1,
                    markersize=7.2,
                    alpha=0.9,
                )
        ax.set_title(label)
        _set_xticks(ax)
        ax.set_xlabel("Grid size")
        ax.set_ylabel("Best validation accuracy (%)")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", alpha=0.28)
        _format_percent_axis(ax)
        handles = [
            Line2D([0], [0], color="#444444", linestyle="-", linewidth=2.2, label=label),
            Line2D([0], [0], color="#444444", linestyle="--", linewidth=2.2, label="Regular Part 1 reference"),
            *[
                Line2D(
                    [0],
                    [0],
                    color=DIFFICULTY_COLORS[d],
                    marker=DIFFICULTY_MARKERS[d],
                    linestyle="None",
                    markersize=7.5,
                    label=DIFFICULTY_LABELS[d],
                )
                for d in DIFFICULTY_ORDER
            ],
        ]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        fig.tight_layout()
        stem = f"part2_ablation_{ablation_name}_readable"
        fig.savefig(INTERMEDIATE_DIR / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(INTERMEDIATE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_figure_6() -> None:
    metrics = pd.read_csv(RESULTS_DIR / "tile_permutation_metrics.csv")
    metric_specs = [
        ("global_tile_displacement", "Global tile displacement"),
        ("adjacency_destruction_hardness", "Adjacency destruction"),
        ("spatial_permutation_entropy", "Spatial permutation entropy"),
        ("combined_hardness_score", "Combined hardness score"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, constrained_layout=True)
    for index, (ax, (metric, label)) in enumerate(zip(axes.flat, metric_specs)):
        for difficulty in DIFFICULTY_ORDER:
            group = metrics[metrics["tile_permutation_name"] == difficulty].sort_values("num_tiles")
            ax.plot(
                [_tile_position(v) for v in group["num_tiles"]],
                group[metric],
                color=DIFFICULTY_COLORS[difficulty],
                marker=DIFFICULTY_MARKERS[difficulty],
                linewidth=2.2,
                markersize=7.6,
                label=DIFFICULTY_LABELS[difficulty],
            )
        ax.set_title(label)
        _set_xticks(ax)
        ax.set_xlabel("" if index < 2 else "Grid size")
        ax.set_ylabel("")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, axis="y", alpha=0.28)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.supylabel("Metric value")
    _save(fig, "part3_hardness_metrics_grid_readable")


def main() -> None:
    _set_style()
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_4()
    make_figure_5()
    make_figure_6()


if __name__ == "__main__":
    main()
