"""Shared evaluation helpers for experiment outputs."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import torch
from torch._C import device as TorchDevice
from tqdm.auto import tqdm

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_combined_hardness,
    compute_global_displacement,
    compute_spatial_permutation_entropy,
)
from src.experiments.results import (
    aggregate_accuracy,
    build_result_row,
    experiment_intermediate_figure_path,
    experiment_output_paths,
    save_aggregated_accuracy,
    save_rows,
)
from src.models.registry import SUPPORTED_MODEL_NAMES, validate_model_name
from src.preprocessing.samples import Sample, discover_samples, stratified_split
from src.preprocessing.tile_permutations import (
    build_tile_permutation_records,
    tile_permutation_from_jsonable,
    tile_permutation_to_jsonable,
)
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir, save_csv


__all__ = [
    "aggregate_accuracy",
    "build_result_row",
    "experiment_intermediate_figure_path",
    "experiment_output_paths",
    "save_aggregated_accuracy",
    "save_rows",
]


PERMUTATION_MARKERS = {
    "easy": "o",
    "medium": "x",
    "hard": "^",
    "baseline": "D",
    "unknown": "s",
}
PERMUTATION_X_OFFSETS = {
    "easy": -1.5,
    "medium": -0.5,
    "hard": 0.5,
    "baseline": 1.5,
    "unknown": 0.0,
}
ACCURACY_Y_LIMITS = (0.50, 1.00)
CONDITION_LABELS = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    "baseline": "Baseline: no permutation",
    "unknown": "unknown",
}


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


def _archive_existing_figure(output_path: str) -> str | None:
    """Rename an existing figure with a timestamp before writing a fresh one."""

    path = Path(output_path)
    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    suffix_index = 1
    while archive_path.exists():
        archive_path = path.with_name(f"{path.stem}_{timestamp}_{suffix_index}{path.suffix}")
        suffix_index += 1
    path.rename(archive_path)
    return str(archive_path)


def _permutation_marker_name(value: object) -> str:
    """Return a normalized marker key for a tile permutation difficulty label."""

    if value is None or (isinstance(value, float) and bool(pd.isna(value))):
        return "baseline"
    name = str(value).strip().lower()
    return name if name in PERMUTATION_MARKERS else "unknown"


def _is_missing_scalar(value: object) -> bool:
    return value is None or (isinstance(value, float) and bool(pd.isna(value)))


def _accuracy_plot_marker_name(row: pd.Series) -> str:
    """Return the marker key for accuracy-vs-tiles raw condition points."""

    num_tiles = row.get("num_tiles")
    tiles_per_side = row.get("tiles_per_side")
    tile_permutation = row.get("tile_permutation")
    if (
        (num_tiles is not None and not pd.isna(num_tiles) and int(num_tiles) == 1)
        or _is_missing_scalar(tiles_per_side)
        or _is_missing_scalar(tile_permutation)
    ):
        return "baseline"
    return _permutation_marker_name(row.get("tile_permutation_name"))


def _tile_grid_label(value: object) -> str:
    """Return a readable tile-grid label."""

    if value is None or (isinstance(value, float) and bool(pd.isna(value))):
        return "1x1"
    tiles_per_side = int(value)
    return f"{tiles_per_side}x{tiles_per_side}"


def _single_value_label(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = [str(value) for value in frame[column].dropna().unique()]
    return values[0] if len(values) == 1 else None


def _plot_title(base_title: str, *, frame: pd.DataFrame, model_column: str = "model_name") -> str:
    model_name = _single_value_label(frame, model_column)
    ablation_name = _single_value_label(frame, "ablation_name")
    context = " / ".join(value for value in [model_name, ablation_name] if value)
    return f"{context}: {base_title}" if context else base_title


def _model_values(frame: pd.DataFrame, model_column: str) -> list[str]:
    if model_column not in frame.columns:
        return []
    return [str(value) for value in frame[model_column].dropna().unique()]


def _color_by_model(model_values: Sequence[str]) -> dict[str, str]:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        return {}
    known_colors = {
        model_name: color_cycle[index % len(color_cycle)]
        for index, model_name in enumerate(SUPPORTED_MODEL_NAMES)
    }
    extra_names = sorted(model_name for model_name in model_values if model_name not in known_colors)
    extra_colors = {
        model_name: color_cycle[(len(SUPPORTED_MODEL_NAMES) + index) % len(color_cycle)]
        for index, model_name in enumerate(extra_names)
    }
    return {model_name: known_colors.get(model_name, extra_colors.get(model_name, "")) for model_name in model_values}


def _tile_axis_label(value: object) -> str:
    num_tiles = int(value)
    if num_tiles == 1:
        return "1"
    tiles_per_side = int(num_tiles ** 0.5)
    if tiles_per_side * tiles_per_side == num_tiles:
        return f"{tiles_per_side}x{tiles_per_side}"
    return str(num_tiles)


def _tile_axis_positions(values: Sequence[object]) -> dict[int, int]:
    tick_values = sorted({int(value) for value in values if not pd.isna(value)})
    return {value: index for index, value in enumerate(tick_values)}


def _is_intermediate_accuracy_plot(output_path: str, title: str | None) -> bool:
    if title is not None:
        return title.startswith("Intermediate Model Plot:")
    return f"{os.sep}intermediate{os.sep}" in output_path


def _set_tile_axis_ticks(ax: Axes, values: Sequence[object]) -> dict[int, int]:
    tile_positions = _tile_axis_positions(values)
    tick_values = list(tile_positions)
    ax.set_xticks(
        [tile_positions[value] for value in tick_values],
        [_tile_axis_label(value) for value in tick_values],
    )
    return tile_positions


def _condition_marker_kwargs(marker_name: str, *, color: str | None, alpha: float = 0.78) -> dict[str, Any]:
    marker = PERMUTATION_MARKERS[marker_name]
    kwargs: dict[str, Any] = {
        "marker": marker,
        "s": 72 if marker_name == "baseline" else 58,
        "alpha": alpha,
        "color": color,
        "linewidths": 1.1,
    }
    if marker != "x":
        kwargs["edgecolors"] = "black"
    return kwargs


def _raw_accuracy_column(raw_results: pd.DataFrame) -> str:
    if "best_val_accuracy" in raw_results.columns:
        return "best_val_accuracy"
    if "val_accuracy" in raw_results.columns:
        return "val_accuracy"
    raise ValueError("raw_results must contain best_val_accuracy or val_accuracy")


def _with_num_tiles(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with ``num_tiles`` populated from ``tiles_per_side`` if needed."""

    frame = frame.copy()
    if "num_tiles" in frame.columns or "tiles_per_side" not in frame.columns:
        return frame
    frame["num_tiles"] = [
        1 if pd.isna(tiles_per_side) else int(tiles_per_side) * int(tiles_per_side)
        for tiles_per_side in frame["tiles_per_side"]
    ]
    return frame


def _scatter_raw_accuracy_vs_tiles(
    ax: Axes,
    raw_results: pd.DataFrame | None,
    *,
    model_column: str,
    color_by_model: dict[str, str],
    tile_positions: dict[int, int],
    alpha: float = 0.78,
) -> None:
    """Overlay raw permutation results on an accuracy-vs-tiles axis."""

    if raw_results is None or raw_results.empty or "num_tiles" not in raw_results.columns:
        return

    raw = raw_results.copy()
    accuracy_column = _raw_accuracy_column(raw)
    if "tile_permutation_name" not in raw.columns:
        raw["tile_permutation_name"] = None

    for _, row in raw.iterrows():
        num_tiles = int(row["num_tiles"])
        if num_tiles not in tile_positions:
            continue
        marker_name = _accuracy_plot_marker_name(row)
        offset = PERMUTATION_X_OFFSETS[marker_name] * 0.045
        model_name = str(row.get(model_column, "raw"))
        ax.scatter(
            tile_positions[num_tiles] + offset,
            row[accuracy_column],
            **_condition_marker_kwargs(
                marker_name,
                color=color_by_model.get(model_name),
                alpha=alpha,
            ),
            label="_nolegend_",
        )


def _condition_legend_handles() -> list[plt.Line2D]:
    handles = []
    for name in ("easy", "medium", "hard", "baseline"):
        marker = PERMUTATION_MARKERS[name]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="0.25",
                markeredgecolor="black" if marker != "x" else "0.25",
                linestyle="None",
                markersize=7,
                label=CONDITION_LABELS[name],
            )
        )
    return handles


def _add_permutation_marker_legend(
    ax: Axes,
    *,
    loc: str = "best",
    bbox_to_anchor: tuple[float, float] | None = None,
) -> Any | None:
    """Add a compact legend for raw permutation marker shapes."""

    marker_legend = ax.legend(
        handles=_condition_legend_handles(),
        title="Condition",
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
    )
    if marker_legend is not None:
        ax.add_artist(marker_legend)
    return marker_legend


def add_part2_grid_baseline_deltas(
    aggregated: pd.DataFrame,
    raw_results: pd.DataFrame | None = None,
    *,
    baseline_name: str = "regular_part1",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add grid-matched Part 1 baseline deltas to Part 2 result frames."""

    if aggregated.empty or "ablation_name" not in aggregated.columns:
        return aggregated.copy(), None if raw_results is None else raw_results.copy()

    key_columns = ["tiles_per_side", "num_tiles"]
    aggregated_with_delta = _with_num_tiles(aggregated)
    baseline_rows = aggregated_with_delta[aggregated_with_delta["ablation_name"] == baseline_name]
    if baseline_rows.empty:
        aggregated_with_delta["delta_vs_grid_baseline"] = pd.NA
    else:
        baseline_by_grid = (
            baseline_rows.groupby(key_columns, dropna=False)["mean_best_epoch_val_accuracy"].mean().reset_index()
        )
        baseline_by_grid = baseline_by_grid.rename(
            columns={"mean_best_epoch_val_accuracy": "_grid_baseline_best_val_accuracy"}
        )
        aggregated_with_delta = aggregated_with_delta.merge(baseline_by_grid, on=key_columns, how="left")
        aggregated_with_delta["delta_vs_grid_baseline"] = (
            aggregated_with_delta["mean_best_epoch_val_accuracy"]
            - aggregated_with_delta["_grid_baseline_best_val_accuracy"]
        )

    if raw_results is None:
        return aggregated_with_delta, None

    raw_with_delta = _with_num_tiles(raw_results)
    if raw_with_delta.empty or "ablation_name" not in raw_with_delta.columns:
        return aggregated_with_delta, raw_with_delta

    accuracy_column = _raw_accuracy_column(raw_with_delta)
    raw_baseline = raw_with_delta[raw_with_delta["ablation_name"] == baseline_name]
    if raw_baseline.empty:
        raw_with_delta["delta_vs_grid_baseline"] = pd.NA
        return aggregated_with_delta, raw_with_delta

    exact_key_columns = [*key_columns, "tile_permutation_id"]
    exact_baseline = raw_baseline.groupby(exact_key_columns, dropna=False)[accuracy_column].mean().reset_index()
    exact_baseline = exact_baseline.rename(columns={accuracy_column: "_exact_baseline_best_val_accuracy"})
    grid_baseline = raw_baseline.groupby(key_columns, dropna=False)[accuracy_column].mean().reset_index()
    grid_baseline = grid_baseline.rename(columns={accuracy_column: "_grid_baseline_best_val_accuracy"})

    raw_with_delta = raw_with_delta.merge(exact_baseline, on=exact_key_columns, how="left")
    raw_with_delta = raw_with_delta.merge(grid_baseline, on=key_columns, how="left")
    raw_with_delta["_matched_baseline_best_val_accuracy"] = raw_with_delta[
        "_exact_baseline_best_val_accuracy"
    ].fillna(raw_with_delta["_grid_baseline_best_val_accuracy"])
    raw_with_delta["delta_vs_grid_baseline"] = (
        raw_with_delta[accuracy_column] - raw_with_delta["_matched_baseline_best_val_accuracy"]
    )
    return aggregated_with_delta, raw_with_delta


def _part3_metric_slug(metric: str) -> str:
    return "combined_hardness" if metric == "combined_hardness_score" else metric


def _part3_metric_grid_plot_paths(figures_dir: str, metric: str) -> dict[str, str]:
    metric_slug = _part3_metric_slug(metric)
    return {
        "grid_hardness": os.path.join(figures_dir, f"part3_{metric_slug}_grid_vs_hardness.png"),
        "grid_hardness_accuracy_3d": os.path.join(
            figures_dir,
            f"part3_{metric_slug}_grid_hardness_accuracy_3d.png",
        ),
    }


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


def _maybe_stage_colab_data_dir(config: CVExperimentConfig) -> str:
    """Stage Drive-backed Colab data only when the active runtime is Colab."""

    data_dir = str(config.data_dir)
    if not bool(getattr(config, "using_google_colab", False)):
        return data_dir

    from src.utils.colab import stage_colab_data_to_local_disk

    return stage_colab_data_to_local_disk(
        data_dir,
        local_data_dir=getattr(
            config,
            "colab_local_data_dir",
            "/content/MLDS_Final_Project/data/dogs-vs-cats/train",
        ),
        enabled=bool(getattr(config, "stage_colab_data_to_local_disk", True)),
        using_google_colab=True,
    )


def stage_configured_colab_data_dir(config: CVExperimentConfig) -> str:
    """Apply the configured Colab data-staging policy and update ``config.data_dir``."""

    config.data_dir = _maybe_stage_colab_data_dir(config)
    return str(config.data_dir)


def load_experiment_samples(config: CVExperimentConfig, seed: int) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Discover and split configured Dogs vs Cats samples.

    Args:
        config: Normalized experiment configuration with data options.
        seed: Random seed used by the stratified split.

    Returns:
        Tuple of train, validation, and test sample lists.
    """

    stage_configured_colab_data_dir(config)
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


def plot_accuracy_vs_tiles(
    aggregated: pd.DataFrame,
    output_path: str,
    model_column: str = "model_name",
    raw_results: pd.DataFrame | None = None,
    title: str | None = None,
) -> None:
    """Save an accuracy-vs-number-of-tiles plot.

    Args:
        aggregated: Aggregated result table.
        output_path: Destination path for the figure.
        model_column: Column used to split one curve per model.
        raw_results: Optional raw result table to overlay individual permutations.
        title: Optional explicit plot title.
    """

    ensure_dir(os.path.dirname(output_path) or ".")
    aggregated = _with_num_tiles(aggregated)
    raw_results = None if raw_results is None else _with_num_tiles(raw_results)
    model_values = _model_values(aggregated, model_column)
    color_by_model = _color_by_model(model_values)
    is_intermediate_plot = _is_intermediate_accuracy_plot(output_path, title)

    fig, axis = plt.subplots(figsize=(9, 5.5))
    ax = _as_axes(axis)
    x_values = list(aggregated["num_tiles"])
    if raw_results is not None and not raw_results.empty and "num_tiles" in raw_results.columns:
        x_values.extend(list(raw_results["num_tiles"]))
    tile_positions = _set_tile_axis_ticks(ax, x_values)
    for model_name, group in aggregated.groupby(model_column):
        group = _sorted_dataframe(cast(pd.DataFrame, group), "num_tiles")
        line_label = "Mean across permutations" if is_intermediate_plot else str(model_name)
        line_x_values = [tile_positions[int(num_tiles)] for num_tiles in group["num_tiles"]]
        ax.plot(
            line_x_values,
            group["mean_best_epoch_val_accuracy"],
            marker=None if is_intermediate_plot else "o",
            linewidth=2.0,
            color=color_by_model.get(str(model_name)),
            label=line_label,
        )

    if raw_results is not None and not raw_results.empty:
        _scatter_raw_accuracy_vs_tiles(
            ax,
            raw_results,
            model_column=model_column,
            color_by_model=color_by_model,
            tile_positions=tile_positions,
            alpha=0.82 if is_intermediate_plot else 0.56,
        )

    ax.set_ylim(*ACCURACY_Y_LIMITS)
    ax.set_xlabel("Number of image tiles after splitting")
    ax.set_ylabel("Best validation accuracy")
    default_title = (
        f"Intermediate Model Plot: Validation Accuracy by Tiling Level - {model_values[0]}"
        if is_intermediate_plot
        else "Final Model Comparison: Mean Validation Accuracy by Tiling Level"
    )
    ax.set_title(title or default_title)
    ax.grid(True, alpha=0.3)
    model_legend_title = "Mean" if is_intermediate_plot else model_column.replace("_", " ").title()
    model_legend = ax.legend(
        title=model_legend_title,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0) if raw_results is not None and not raw_results.empty else None,
    )
    if model_legend is not None:
        ax.add_artist(model_legend)
    extra_artists = []
    if model_legend is not None:
        extra_artists.append(model_legend)
    if raw_results is not None and not raw_results.empty:
        marker_legend = _add_permutation_marker_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 0.66))
        if marker_legend is not None:
            extra_artists.append(marker_legend)
    if raw_results is not None and not raw_results.empty:
        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    else:
        fig.tight_layout()
    _archive_existing_figure(output_path)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(fig)


def plot_ablation_results(
    aggregated: pd.DataFrame,
    output_path: str,
    raw_results: pd.DataFrame | None = None,
    title: str | None = None,
) -> None:
    """Save a baseline-vs-improvement ablation plot.

    Args:
        aggregated: Aggregated Part 2 result table.
        output_path: Destination path for the figure.
        raw_results: Optional raw result table to overlay individual permutations.
        title: Optional explicit plot title.
    """

    ensure_dir(os.path.dirname(output_path) or ".")
    aggregated, raw_results = add_part2_grid_baseline_deltas(aggregated, raw_results)
    y_column = (
        "delta_vs_grid_baseline"
        if "delta_vs_grid_baseline" in aggregated.columns and aggregated["delta_vs_grid_baseline"].notna().any()
        else "mean_best_epoch_val_accuracy"
    )
    ablation_names = sorted(str(name) for name in aggregated["ablation_name"].dropna().unique())
    x_positions = {ablation_name: index for index, ablation_name in enumerate(ablation_names)}
    aggregated["_tile_label"] = aggregated["tiles_per_side"].map(_tile_grid_label)
    tile_values = list(aggregated["_tile_label"].drop_duplicates())
    tile_offsets = {
        tile_value: (index - (len(tile_values) - 1) / 2.0) * 0.12
        for index, tile_value in enumerate(tile_values)
    }

    fig, axis = plt.subplots(figsize=(8, 5))
    ax = _as_axes(axis)
    grid_handles = []
    for tile_label, group in aggregated.groupby("_tile_label", sort=False):
        sorted_group = _sorted_dataframe(cast(pd.DataFrame, group), "ablation_name")
        x_values = [
            x_positions[str(ablation_name)] + tile_offsets.get(str(tile_label), 0.0)
            for ablation_name in sorted_group["ablation_name"]
        ]
        scatter = ax.scatter(
            x_values,
            sorted_group[y_column],
            marker="o",
            s=58,
            alpha=0.85,
            label=str(tile_label),
        )
        grid_handles.append((scatter, str(tile_label)))

    if raw_results is not None and not raw_results.empty:
        raw = raw_results.copy()
        raw_y_column = "delta_vs_grid_baseline" if y_column == "delta_vs_grid_baseline" else _raw_accuracy_column(raw)
        if "tile_permutation_name" not in raw.columns:
            raw["tile_permutation_name"] = None
        for _, row in raw.iterrows():
            ablation_name = str(row.get("ablation_name"))
            if ablation_name not in x_positions or raw_y_column not in row or pd.isna(row[raw_y_column]):
                continue
            marker_name = _permutation_marker_name(row.get("tile_permutation_name"))
            tile_label = _tile_grid_label(row.get("tiles_per_side"))
            x_value = (
                x_positions[ablation_name]
                + tile_offsets.get(tile_label, 0.0)
                + PERMUTATION_X_OFFSETS[marker_name] * 0.035
            )
            ax.scatter(
                x_value,
                row[raw_y_column],
                **_condition_marker_kwargs(marker_name, color="0.25", alpha=0.65),
                label="_nolegend_",
            )

    if y_column == "delta_vs_grid_baseline":
        ax.axhline(0.0, color="0.25", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.set_xlabel("Ablation")
    ax.set_ylabel(
        "Best validation accuracy - matched Part 1 grid baseline"
        if y_column == "delta_vs_grid_baseline"
        else "Best validation accuracy"
    )
    ax.set_title(title or _plot_title("Ablations vs Matched Grid Baseline", frame=aggregated))
    ax.set_xticks(list(x_positions.values()), ablation_names)
    ax.grid(True, axis="y", alpha=0.3)
    grid_legend = ax.legend(
        [handle for handle, _ in grid_handles],
        [label for _, label in grid_handles],
        title="Grid",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0) if raw_results is not None and not raw_results.empty else None,
    )
    if grid_legend is not None:
        ax.add_artist(grid_legend)
    if raw_results is not None and not raw_results.empty:
        _add_permutation_marker_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 0.66))
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    _archive_existing_figure(output_path)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def part3_output_paths(results_dir: str, figures_dir: str) -> Dict[str, object]:
    """Return stable output paths for notebook-owned Part 3 analysis."""

    metric_plots = {
        metric: _part3_metric_grid_plot_paths(figures_dir, metric)
        for metric in PART3_METRIC_COLUMNS
    }
    ordered_plot_paths = [
        path
        for metric in PART3_METRIC_COLUMNS
        for path in [
            metric_plots[metric]["grid_hardness"],
            metric_plots[metric]["grid_hardness_accuracy_3d"],
        ]
    ]
    existing_plots = [
        path
        for path in ordered_plot_paths
        if os.path.exists(path)
    ]
    return {
        "metrics": os.path.join(results_dir, "tile_permutation_metrics.csv"),
        "joined": os.path.join(results_dir, "metric_accuracy_joined.csv"),
        "correlations": os.path.join(results_dir, "metric_accuracy_correlations.csv"),
        "metric_plots": metric_plots,
        "plots": existing_plots,
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

    rows = []
    for record in build_tile_permutation_records(
        tiles_per_side_values=tiles_per_side_values,
        num_tile_permutations=int(num_tile_permutations),
        seed=seed,
        include_baseline=True,
    ):
        rows.append(
            {
                "tiles_per_side": record.tiles_per_side,
                "tile_permutation_id": record.tile_permutation_id,
                "tile_permutation_name": record.tile_permutation_name,
                "tile_permutation_seed": record.tile_permutation_seed,
                "tile_permutation": json.dumps(tile_permutation_to_jsonable(record.tile_permutation)),
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
    "adjacency_destruction_hardness",
    "spatial_permutation_entropy",
    "combined_hardness_score",
]


def _add_combined_hardness_scores(
    metrics: pd.DataFrame,
    *,
    weight_adj: float,
    weight_entropy: float,
    weight_dist: float,
) -> pd.DataFrame:
    """Add final combined hardness from normalized component scores."""

    metrics = metrics.copy()
    if metrics.empty:
        metrics["combined_hardness_score"] = []
        return metrics

    metrics["combined_hardness_score"] = [
        compute_combined_hardness(
            adjacency_destruction_hardness=float(row["adjacency_destruction_hardness"]),
            spatial_permutation_entropy=float(row["spatial_permutation_entropy"]),
            global_tile_displacement=float(row["global_tile_displacement"]),
            weight_adj=weight_adj,
            weight_entropy=weight_entropy,
            weight_dist=weight_dist,
        )
        for _, row in metrics.iterrows()
    ]
    return metrics


def compute_part3_tile_permutation_metrics(
    *,
    tile_permutation_csv: str,
    tiles_per_side_values: Sequence[int],
    num_tile_permutations: int,
    seed: int,
    validation_samples: Sequence[Sample],
    image_size: int,
    weight_adj: float = 0.5,
    weight_entropy: float = 0.3,
    weight_dist: float = 0.2,
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
        adjacency_destruction_hardness = compute_adjacency_destruction_hardness(
            tile_permutation,
            tiles_per_side,
        )
        spatial_permutation_entropy = compute_spatial_permutation_entropy(
            tile_permutation,
            tiles_per_side,
        )
        rows.append(
            {
                "tiles_per_side": tiles_per_side,
                "num_tiles": num_tiles,
                "tile_permutation_id": int(cast(Any, row["tile_permutation_id"])),
                "tile_permutation_name": cast(Any, row.get("tile_permutation_name")),
                "tile_permutation_seed": cast(Any, row.get("tile_permutation_seed")),
                "global_tile_displacement": global_tile_displacement,
                "adjacency_destruction_hardness": adjacency_destruction_hardness,
                "spatial_permutation_entropy": spatial_permutation_entropy,
            }
        )
    metrics = _add_combined_hardness_scores(
        pd.DataFrame(rows),
        weight_adj=weight_adj,
        weight_entropy=weight_entropy,
        weight_dist=weight_dist,
    )
    return _reset_dataframe_index(
        _sorted_dataframe(metrics, ["tiles_per_side", "tile_permutation_id"], na_position="first")
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


def _normalize_part3_joined_columns(joined: pd.DataFrame) -> pd.DataFrame:
    """Collapse merge-suffixed Part 3 metadata columns into stable names."""

    joined = joined.copy()
    for column in ("tile_permutation_name", "tile_permutation_seed"):
        metric_column = f"{column}_y"
        raw_column = f"{column}_x"
        if metric_column in joined.columns and raw_column in joined.columns:
            joined[column] = joined[metric_column].combine_first(joined[raw_column])
            joined = joined.drop(columns=[raw_column, metric_column])
        elif metric_column in joined.columns:
            joined[column] = joined[metric_column]
            joined = joined.drop(columns=[metric_column])
        elif raw_column in joined.columns:
            joined[column] = joined[raw_column]
            joined = joined.drop(columns=[raw_column])
    return joined


def _part3_plot_marker_name(row: pd.Series) -> str:
    """Return condition marker key for a Part 3 joined row."""

    num_tiles = row.get("num_tiles")
    tiles_per_side = row.get("tiles_per_side")
    tile_permutation_id = row.get("tile_permutation_id")
    if (
        (num_tiles is not None and not pd.isna(num_tiles) and int(num_tiles) == 1)
        or _is_missing_scalar(tiles_per_side)
        or (tile_permutation_id is not None and not pd.isna(tile_permutation_id) and int(tile_permutation_id) == 0)
    ):
        return "baseline"
    return _permutation_marker_name(row.get("tile_permutation_name"))


def _part3_grid_positions(frame: pd.DataFrame) -> dict[int, int]:
    frame = _with_num_tiles(frame)
    return _tile_axis_positions(list(frame["num_tiles"]))


def _set_part3_grid_axis_ticks(ax: Axes, grid_positions: dict[int, int]) -> None:
    tick_values = list(grid_positions)
    ax.set_xticks(
        [grid_positions[value] for value in tick_values],
        [_tile_axis_label(value) for value in tick_values],
    )


def _part3_metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def plot_part3_metric_grid_views(
    joined: pd.DataFrame,
    figures_dir: str,
    metric: str,
    title: str | None = None,
) -> dict[str, str]:
    """Save Part 3 grid-vs-hardness and grid/hardness/accuracy scatter plots."""

    if metric not in PART3_METRIC_COLUMNS:
        raise ValueError(f"Unsupported Part 3 metric: {metric}")

    ensure_dir(figures_dir)
    output_paths = _part3_metric_grid_plot_paths(figures_dir, metric)
    frame = _with_num_tiles(_normalize_part3_joined_columns(joined))
    if "tile_permutation_name" not in frame.columns:
        frame["tile_permutation_name"] = None
    frame["_condition_marker"] = [_part3_plot_marker_name(row) for _, row in frame.iterrows()]
    grid_positions = _part3_grid_positions(frame)
    metric_label = _part3_metric_label(metric)

    fig, axis = plt.subplots(figsize=(7, 5))
    ax = _as_axes(axis)
    for marker_name, group in frame.groupby("_condition_marker", sort=False):
        x_values = [grid_positions[int(num_tiles)] for num_tiles in group["num_tiles"]]
        ax.scatter(
            x_values,
            group[metric],
            **_condition_marker_kwargs(str(marker_name), color=None, alpha=0.78),
            label="_nolegend_",
        )
    _set_part3_grid_axis_ticks(ax, grid_positions)
    ax.set_xlabel("Grid size")
    ax.set_ylabel(metric_label)
    base_title = f"{metric_label} by Grid Size"
    ax.set_title(title or _plot_title(base_title, frame=frame))
    ax.grid(True, alpha=0.3)
    _add_permutation_marker_legend(ax)
    fig.tight_layout()
    _archive_existing_figure(output_paths["grid_hardness"])
    fig.savefig(output_paths["grid_hardness"], dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    axis_3d = fig.add_subplot(111, projection="3d")
    for marker_name, group in frame.groupby("_condition_marker", sort=False):
        x_values = [grid_positions[int(num_tiles)] for num_tiles in group["num_tiles"]]
        axis_3d.scatter(
            x_values,
            group[metric],
            group["best_val_accuracy"],
            **_condition_marker_kwargs(str(marker_name), color=None, alpha=0.78),
            label="_nolegend_",
        )
    tick_values = list(grid_positions)
    axis_3d.set_xticks(
        [grid_positions[value] for value in tick_values],
        [_tile_axis_label(value) for value in tick_values],
    )
    axis_3d.set_xlabel("Grid size")
    axis_3d.set_ylabel(metric_label)
    axis_3d.set_zlabel("Best validation accuracy")
    base_title = f"{metric_label} by Grid Size and Accuracy"
    axis_3d.set_title(title or _plot_title(base_title, frame=frame))
    axis_3d.grid(True, alpha=0.3)
    _add_permutation_marker_legend(axis_3d)
    fig.tight_layout()
    _archive_existing_figure(output_paths["grid_hardness_accuracy_3d"])
    fig.savefig(output_paths["grid_hardness_accuracy_3d"], dpi=160)
    plt.close(fig)
    return output_paths


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
    validation_samples: Sequence[Sample],
    image_size: int,
    model_name: str = "resnet18",
    weight_adj: float = 0.5,
    weight_entropy: float = 0.3,
    weight_dist: float = 0.2,
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
        validation_samples=validation_samples,
        image_size=image_size,
        weight_adj=weight_adj,
        weight_entropy=weight_entropy,
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
    joined = _normalize_part3_joined_columns(joined)
    joined = _reset_dataframe_index(
        _sorted_dataframe(joined, ["tiles_per_side", "tile_permutation_id"], na_position="first")
    )
    log("Saving joined metric-accuracy table...")
    save_csv(joined, os.path.join(results_dir, "metric_accuracy_joined.csv"))

    log("Computing metric-accuracy correlations...")
    correlations = compute_part3_metric_correlations(joined, group_name=model_name)
    log("Saving correlations and plots...")
    save_csv(correlations, os.path.join(results_dir, "metric_accuracy_correlations.csv"))
    for metric in PART3_METRIC_COLUMNS:
        plot_part3_metric_grid_views(joined, figures_dir, metric)
    log("Part 3 hardness analysis complete.")
    return {"metrics": metrics, "joined": joined, "correlations": correlations}
