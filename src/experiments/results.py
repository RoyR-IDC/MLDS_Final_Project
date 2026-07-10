"""Shared experiment result row and CSV persistence helpers."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from src.preprocessing.tile_permutations import TilePermutationRecord, tile_permutation_to_jsonable
from src.training.run import TrainingResult
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir, save_csv


def build_result_row(
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    seed: int,
    metrics: Mapping[str, Any],
    ablation_name: str | None = None,
) -> dict[str, Any]:
    """Build one experiment result row."""

    num_tiles = 1 if record.tiles_per_side is None else record.tiles_per_side * record.tiles_per_side
    row = {
        "part": config.part,
        "run_id": run_id,
        "config_name": config.config_name,
        "model_name": model_name,
        "tiles_per_side": record.tiles_per_side,
        "num_tiles": num_tiles,
        "tile_permutation_id": record.tile_permutation_id,
        "tile_permutation_name": record.tile_permutation_name,
        "tile_permutation_seed": record.tile_permutation_seed,
        "tile_permutation": tile_permutation_to_jsonable(record.tile_permutation),
        "seed": seed,
        **metrics,
    }
    if ablation_name is not None:
        row["ablation_name"] = ablation_name
    return row


def build_training_result_row(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    seed: int,
    result: TrainingResult,
    ablation_name: str | None = None,
) -> dict[str, Any]:
    """Convert a structured training result to a CSV row."""

    row = build_result_row(
        config=config,
        run_id=run_id,
        model_name=model_name,
        record=record,
        seed=seed,
        metrics=result.latest_metrics(),
        ablation_name=ablation_name,
    )
    for key in (
        "pretrained",
        "freeze_backbone",
        "epochs",
        "augmentation_name",
        "batch_augmentation_name",
        "curriculum_name",
        "curriculum_stages",
        "classification_head",
        "p_original",
        "tile_permutation_probability",
        "tile_permutation_name",
        "global_tile_displacement",
        "adjacency_destruction_hardness",
        "spatial_permutation_entropy",
        "combined_hardness_score",
        "hardness_level",
    ):
        if key in result.metadata:
            row[key] = result.metadata[key]
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

    return [{str(key): _csv_safe_value(value) for key, value in row.items()} for row in rows]


def _is_missing_result_value(value: Any) -> bool:
    """Return whether a CSV cell value should compare as missing."""

    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, bool) else False


def _normalized_result_key_value(value: Any) -> Any:
    """Return a stable, hashable key value for CSV row matching."""

    if _is_missing_result_value(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)
    return str(value)


def _result_row_identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the narrow identity used to replace only incoming result rows."""

    key_columns = (
        "run_id",
        "config_name",
        "model_name",
        "ablation_name",
        "tiles_per_side",
        "tile_permutation_id",
        "seed",
    )
    return tuple(_normalized_result_key_value(row.get(column)) for column in key_columns)


def save_rows(rows: Sequence[Mapping[Any, Any]], output_path: str) -> None:
    """Save experiment rows to CSV."""

    save_csv(_csv_safe_rows(rows), output_path)


def save_run_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_path: str,
    run_id: str,
    model_name: str | None = None,
) -> None:
    """Save rows for one run while preserving unrelated existing results."""

    existing_rows: list[dict[str, Any]] = []
    if os.path.exists(output_path):
        existing_results = pd.read_csv(output_path)
        if rows:
            incoming_keys = {_result_row_identity_key(row) for row in rows}
            if incoming_keys:
                existing_mask = existing_results.apply(
                    lambda existing_row: _result_row_identity_key(existing_row.to_dict()) in incoming_keys,
                    axis=1,
                )
                existing_results = existing_results[~existing_mask]
        existing_rows = existing_results.to_dict("records")

    save_rows([*existing_rows, *rows], output_path)


def aggregate_accuracy(raw_results: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Aggregate accuracy columns for repeated experiment runs."""

    return (
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


def save_aggregated_accuracy(raw_results: pd.DataFrame, group_columns: Sequence[str], output_path: str) -> pd.DataFrame:
    """Aggregate raw results and save the aggregated table."""

    aggregated_results = aggregate_accuracy(raw_results, group_columns)
    ensure_dir(os.path.dirname(output_path) or ".")
    aggregated_results.to_csv(output_path, index=False)
    return aggregated_results


def experiment_output_paths(results_dir: str, figures_dir: str, part_name: str) -> dict[str, str]:
    """Return standard output paths for a notebook-owned experiment."""

    figure_name = "accuracy_vs_tiles" if part_name == "part1" else "ablation_comparison"
    output_paths = {
        "raw_results": os.path.join(results_dir, f"{part_name}_raw_results.csv"),
        "aggregated_results": os.path.join(results_dir, f"{part_name}_aggregated_results.csv"),
        "figure": os.path.join(figures_dir, f"{part_name}_{figure_name}.png"),
    }
    output_paths["intermediate_figures_dir"] = os.path.join(figures_dir, "intermediate")
    if part_name == "part1":
        output_paths["tile_permutations"] = os.path.join(results_dir, "part1_tile_permutations.csv")
    return output_paths


def experiment_intermediate_figure_path(figures_dir: str, part_name: str, figure_slug: str) -> str:
    """Return a stable path for a notebook intermediate figure."""

    safe_slug = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in figure_slug)
    return os.path.join(figures_dir, "intermediate", f"{part_name}_{safe_slug}.png")
