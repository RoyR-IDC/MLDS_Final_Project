"""Reviewer-requested enhanced confidence experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import json
import os
from time import perf_counter
from typing import Any, Iterator, Optional, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from torch._C import device as TorchDevice

from src.evaluation.experiment_results import get_device, load_experiment_samples
from src.experiments.part1 import (
    initialize_tile_permutation_result_rows,
    train_single_tile_permutation_run,
)
from src.experiments.part2 import (
    initialize_ablation_result_rows,
    train_single_ablation_tile_run,
)
from src.experiments.results import aggregate_accuracy, save_rows, save_run_rows
from src.models.registry import CNN_MODEL_NAMES
from src.preprocessing.tile_permutations import (
    ENHANCED_BASELINE_TILE_PERMUTATION_ID,
    ENHANCED_TILE_PERMUTATION_IDS,
    ENHANCED_TILE_PERMUTATION_NAMES,
    TilePermutationRecord,
    base_tile_permutation_name,
    build_enhanced_tile_permutation_records,
    tile_permutation_to_jsonable,
)
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir
from src.utils.reproducibility import seed_everything


ENHANCED_CONFIDENCE_SEEDS = (42, 43)
ENHANCED_CONFIDENCE_TILES_PER_SIDE_VALUES = (1, 4, 7, 10, 14, 17)
ENHANCED_CONFIDENCE_MODEL_NAME = "mobilenetv3_small"
PART1_ENHANCED_CONFIG_NAME = "part1_enhanced_confidence"
PART2_ENHANCED_CONFIG_NAME = "part2_enhanced_confidence"
PART1_ENHANCED_ABLATION_NAME: str | None = None
PART2_BASELINE_ABLATION_NAME = "regular_part1"
PART2_CURRICULUM_ABLATION_NAME = "curriculum_permutation_difficulty"
PART2_CURRICULUM_ABLATION = {
    "name": PART2_CURRICULUM_ABLATION_NAME,
    "use_pretrained": True,
    "freeze_backbone": True,
    "augmentation": "none",
    "curriculum": "permutation_difficulty",
    "classification_head": "linear",
}
ENHANCED_MARKERS = {
    "easy": "o",
    "medium": "x",
    "hard": "^",
    "baseline": "D",
}


def enhanced_confidence_output_paths(results_dir: str, figures_dir: str, part_name: str) -> dict[str, str]:
    """Return stable enhanced-confidence output paths for a notebook part."""

    prefix = f"{part_name}_enhanced"
    return {
        "raw_results": os.path.join(results_dir, f"{prefix}_raw_results.csv"),
        "aggregated_results": os.path.join(results_dir, f"{prefix}_aggregated_results.csv"),
        "all_points_figure": os.path.join(figures_dir, f"{prefix}_confidence_all_points.png"),
        "mean_by_seed_figure": os.path.join(figures_dir, f"{prefix}_confidence_mean_by_seed.png"),
    }


def enhanced_confidence_run_id(part_name: str, seed: int) -> str:
    """Return the stable run id for one enhanced seed."""

    return f"{part_name}_enhanced_confidence_seed_{int(seed)}"


def build_enhanced_confidence_records_for_seed(seed: int) -> list[TilePermutationRecord]:
    """Return the enhanced records scheduled for one training seed."""

    return build_enhanced_tile_permutation_records(
        ENHANCED_CONFIDENCE_TILES_PER_SIDE_VALUES,
        seed=int(seed),
        include_baseline=int(seed) == ENHANCED_CONFIDENCE_SEEDS[0],
    )


def build_enhanced_confidence_run_matrix() -> list[dict[str, Any]]:
    """Return the full enhanced matrix as lightweight dictionaries."""

    rows = []
    for seed in ENHANCED_CONFIDENCE_SEEDS:
        for record in build_enhanced_confidence_records_for_seed(seed):
            rows.append(
                {
                    "seed": int(seed),
                    "tiles_per_side": record.tiles_per_side,
                    "num_tiles": 1 if record.tiles_per_side is None else record.tiles_per_side**2,
                    "tile_permutation_id": record.tile_permutation_id,
                    "tile_permutation_name": record.tile_permutation_name,
                }
            )
    return rows


def _as_axes(axis: object) -> Axes:
    return cast(Axes, axis)


def _is_missing_scalar(value: object) -> bool:
    return value is None or (isinstance(value, float) and bool(pd.isna(value)))


def _tiles_key(value: object) -> int:
    return 0 if _is_missing_scalar(value) else int(value)


def _record_key(record: TilePermutationRecord) -> tuple[int, int]:
    return _tiles_key(record.tiles_per_side), int(record.tile_permutation_id)


def _row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return _tiles_key(row.get("tiles_per_side")), int(row.get("tile_permutation_id", 0))


def _enhanced_result_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    ablation = row.get("ablation_name")
    ablation_key = "" if _is_missing_scalar(ablation) else str(ablation)
    return int(row.get("seed", ENHANCED_CONFIDENCE_SEEDS[0])), *_row_key(row), ablation_key


def _record_lookup(seed: int) -> dict[tuple[int, int], TilePermutationRecord]:
    return {_record_key(record): record for record in build_enhanced_confidence_records_for_seed(seed)}


def _is_completed_row(row: Mapping[str, Any]) -> bool:
    status = row.get("run_status", "completed")
    return str(status) == "completed"


def _is_blank_ablation(value: object) -> bool:
    return _is_missing_scalar(value) or str(value).strip() in {"", "nan", "None"}


def _legacy_enhanced_permutation_id(row: Mapping[str, Any]) -> int | None:
    if _is_missing_scalar(row.get("tiles_per_side")):
        return ENHANCED_BASELINE_TILE_PERMUTATION_ID
    name = str(row.get("tile_permutation_name", "")).strip().lower()
    if name in ENHANCED_TILE_PERMUTATION_IDS:
        return ENHANCED_TILE_PERMUTATION_IDS[name]
    return None


def _canonicalize_row(
    row: Mapping[str, Any],
    *,
    config: CVExperimentConfig,
    part_name: str,
    config_name: str,
    run_id: str,
    record: TilePermutationRecord,
    seed: int,
    model_name: str,
    ablation_name: str | None,
) -> dict[str, Any]:
    canonical = dict(row)
    canonical.update(
        {
            "part": part_name,
            "run_id": run_id,
            "config_name": config_name,
            "model_name": model_name,
            "tiles_per_side": record.tiles_per_side,
            "num_tiles": 1 if record.tiles_per_side is None else record.tiles_per_side * record.tiles_per_side,
            "tile_permutation_id": record.tile_permutation_id,
            "tile_permutation_name": record.tile_permutation_name,
            "tile_permutation_seed": record.tile_permutation_seed,
            "tile_permutation": json.dumps(tile_permutation_to_jsonable(record.tile_permutation)),
            "seed": int(seed),
            "pretrained": True,
            "freeze_backbone": True,
            "epochs": int(config.epochs),
            "classification_head": "linear",
        }
    )
    if ablation_name is None:
        canonical.pop("ablation_name", None)
    else:
        canonical["ablation_name"] = ablation_name
    return canonical


def _completed_enhanced_rows(
    *,
    output_path: str,
    model_name: str,
    ablation_name: str | None,
) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    if not os.path.exists(output_path):
        return {}
    raw = pd.read_csv(output_path)
    if raw.empty or "model_name" not in raw.columns:
        return {}
    subset = raw[raw["model_name"].astype(str) == model_name].copy()
    if "ablation_name" in subset.columns:
        if ablation_name is None:
            subset = subset[subset["ablation_name"].apply(_is_blank_ablation)]
        else:
            subset = subset[subset["ablation_name"].astype(str) == ablation_name]
    elif ablation_name is not None:
        return {}
    return {
        _enhanced_result_key(row): dict(row)
        for row in subset.to_dict("records")
        if _is_completed_row(row)
    }


def _legacy_completed_rows(
    *,
    raw_path: str,
    config: CVExperimentConfig,
    part_name: str,
    config_name: str,
    model_name: str,
    source_ablation_name: str | None,
    target_ablation_name: str | None,
) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    if not os.path.exists(raw_path):
        return {}
    raw = pd.read_csv(raw_path)
    if raw.empty or "model_name" not in raw.columns:
        return {}

    seed = ENHANCED_CONFIDENCE_SEEDS[0]
    records_by_key = _record_lookup(seed)
    subset = raw[raw["model_name"].astype(str) == model_name].copy()
    if "seed" in subset.columns:
        subset = subset[subset["seed"].fillna(seed).astype(int) == seed]
    if "ablation_name" in subset.columns:
        if source_ablation_name is None:
            subset = subset[subset["ablation_name"].apply(_is_blank_ablation)]
        else:
            subset = subset[subset["ablation_name"].astype(str) == source_ablation_name]
    elif source_ablation_name is not None:
        return {}
    if "run_status" in subset.columns:
        subset = subset[subset["run_status"].astype(str) == "completed"]

    canonical_rows: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for row in subset.to_dict("records"):
        permutation_id = _legacy_enhanced_permutation_id(row)
        if permutation_id is None:
            continue
        record = records_by_key.get((_tiles_key(row.get("tiles_per_side")), permutation_id))
        if record is None:
            continue
        canonical = _canonicalize_row(
            row,
            config=config,
            part_name=part_name,
            config_name=config_name,
            run_id=enhanced_confidence_run_id(part_name, seed),
            record=record,
            seed=seed,
            model_name=model_name,
            ablation_name=target_ablation_name,
        )
        canonical_rows.setdefault(_enhanced_result_key(canonical), canonical)
    return canonical_rows


def _write_completed_seed_rows(
    *,
    rows_by_key: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    output_path: str,
) -> None:
    if rows_by_key:
        save_rows(list(rows_by_key.values()), output_path)


@contextmanager
def _enhanced_config_scope(
    config: CVExperimentConfig,
    *,
    part_name: str,
    config_name: str,
    seed: int,
) -> Iterator[None]:
    original_values = {
        "part": config.part,
        "config_name": config.config_name,
        "seed": config.seed,
        "model_names": list(config.model_names),
        "tiles_per_side_values": list(config.tiles_per_side_values),
        "num_tile_permutations": config.num_tile_permutations,
    }
    config.part = part_name
    config.config_name = config_name
    config.seed = int(seed)
    config.model_names = [ENHANCED_CONFIDENCE_MODEL_NAME]
    config.tiles_per_side_values = list(ENHANCED_CONFIDENCE_TILES_PER_SIDE_VALUES)
    config.num_tile_permutations = len(ENHANCED_TILE_PERMUTATION_NAMES)
    try:
        yield
    finally:
        for key, value in original_values.items():
            setattr(config, key, value)


def _completed_rows_for_seed(
    rows_by_key: Mapping[tuple[int, int, int, str], Mapping[str, Any]],
    *,
    seed: int,
    ablation_name: str | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    ablation_key = "" if ablation_name is None else ablation_name
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    for key, row in rows_by_key.items():
        row_seed, tiles_per_side, tile_permutation_id, row_ablation = key
        if row_seed == seed and row_ablation == ablation_key:
            completed[(tiles_per_side, tile_permutation_id)] = dict(row)
    return completed


def _expected_enhanced_keys(ablation_name: str | None) -> set[tuple[int, int, int, str]]:
    ablation_key = "" if ablation_name is None else ablation_name
    return {
        (seed, *_record_key(record), ablation_key)
        for seed in ENHANCED_CONFIDENCE_SEEDS
        for record in build_enhanced_confidence_records_for_seed(seed)
    }


def _filter_expected_rows(raw_results: pd.DataFrame, *, ablation_name: str | None = None) -> pd.DataFrame:
    expected_keys = _expected_enhanced_keys(ablation_name)
    return raw_results[
        raw_results.apply(lambda row: _enhanced_result_key(row.to_dict()) in expected_keys, axis=1)
    ].copy()


def _part1_completed_rows(config: CVExperimentConfig, output_path: str) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    completed = _completed_enhanced_rows(
        output_path=output_path,
        model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
        ablation_name=PART1_ENHANCED_ABLATION_NAME,
    )
    legacy = _legacy_completed_rows(
        raw_path=os.path.join(config.results_dir, "part1_raw_results.csv"),
        config=config,
        part_name="part1",
        config_name=PART1_ENHANCED_CONFIG_NAME,
        model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
        source_ablation_name=None,
        target_ablation_name=PART1_ENHANCED_ABLATION_NAME,
    )
    return {**legacy, **completed}


def _part2_curriculum_completed_rows(config: CVExperimentConfig, output_path: str) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    completed = _completed_enhanced_rows(
        output_path=output_path,
        model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
        ablation_name=PART2_CURRICULUM_ABLATION_NAME,
    )
    legacy = _legacy_completed_rows(
        raw_path=os.path.join(config.results_dir, "part2_raw_results.csv"),
        config=config,
        part_name="part2",
        config_name=PART2_ENHANCED_CONFIG_NAME,
        model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
        source_ablation_name=PART2_CURRICULUM_ABLATION_NAME,
        target_ablation_name=PART2_CURRICULUM_ABLATION_NAME,
    )
    return {**legacy, **completed}


def load_part1_enhanced_baseline_rows(config: CVExperimentConfig) -> list[dict[str, Any]]:
    """Load completed Part 1 enhanced baseline rows and retag them for Part 2."""

    paths = enhanced_confidence_output_paths(config.results_dir, config.figures_dir, "part1")
    rows_by_key = _part1_completed_rows(config, paths["raw_results"])
    rows = []
    for row in rows_by_key.values():
        retagged = dict(row)
        retagged["part"] = "part2"
        retagged["config_name"] = PART2_ENHANCED_CONFIG_NAME
        retagged["ablation_name"] = PART2_BASELINE_ABLATION_NAME
        rows.append(retagged)
    return rows


def run_part1_enhanced_confidence_experiment(
    config: CVExperimentConfig,
    device: Optional[TorchDevice] = None,
) -> pd.DataFrame:
    """Run the enhanced Part 1 MobileNet frozen-backbone experiment."""

    if ENHANCED_CONFIDENCE_MODEL_NAME not in CNN_MODEL_NAMES:
        raise ValueError(f"{ENHANCED_CONFIDENCE_MODEL_NAME} must be a supported CNN model")
    output_paths = enhanced_confidence_output_paths(config.results_dir, config.figures_dir, "part1")
    resolved_device = device or get_device(config)
    rows_by_key = _part1_completed_rows(config, output_paths["raw_results"])
    _write_completed_seed_rows(rows_by_key=rows_by_key, output_path=output_paths["raw_results"])

    session_start_time = perf_counter()
    for seed in ENHANCED_CONFIDENCE_SEEDS:
        with _enhanced_config_scope(config, part_name="part1", config_name=PART1_ENHANCED_CONFIG_NAME, seed=seed):
            seed_everything(seed, deterministic=config.deterministic)
            train_samples, validation_samples, _ = load_experiment_samples(config, seed=seed)
            records = build_enhanced_confidence_records_for_seed(seed)
            run_id = enhanced_confidence_run_id("part1", seed)
            rows, row_indices = initialize_tile_permutation_result_rows(
                config=config,
                run_id=run_id,
                model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
                records=records,
                seed=seed,
                ablation_name=PART1_ENHANCED_ABLATION_NAME,
            )
            completed_for_seed = _completed_rows_for_seed(
                rows_by_key,
                seed=seed,
                ablation_name=PART1_ENHANCED_ABLATION_NAME,
            )
            for index, row in enumerate(rows):
                completed = completed_for_seed.get(_row_key(row))
                if completed is not None:
                    rows[index] = completed
            save_run_rows(
                rows=rows,
                output_path=output_paths["raw_results"],
                run_id=run_id,
                model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
            )

            missing_records = [record for record in records if _record_key(record) not in completed_for_seed]
            print(f"Part 1 enhanced seed {seed}: {len(missing_records)} missing run(s).")
            for record in missing_records:
                train_single_tile_permutation_run(
                    config=config,
                    model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
                    run_id=run_id,
                    train_samples=train_samples,
                    validation_samples=validation_samples,
                    record=record,
                    seed=seed,
                    device=resolved_device,
                    rows=rows,
                    row_indices=row_indices,
                    session_start_time=session_start_time,
                    raw_results_output_path=output_paths["raw_results"],
                    ablation_name=PART1_ENHANCED_ABLATION_NAME,
                    training_overrides={
                        "pretrained": True,
                        "freeze_backbone": True,
                        "classification_head": "linear",
                    },
                    metadata_overrides={"epochs": int(config.epochs), "classification_head": "linear"},
                )

    raw_results = pd.read_csv(output_paths["raw_results"])
    filtered = raw_results[
        (raw_results["model_name"].astype(str) == ENHANCED_CONFIDENCE_MODEL_NAME)
        & (
            raw_results["ablation_name"].apply(_is_blank_ablation)
            if "ablation_name" in raw_results.columns
            else True
        )
    ].copy()
    filtered = _filter_expected_rows(filtered, ablation_name=PART1_ENHANCED_ABLATION_NAME)
    aggregated = aggregate_accuracy(
        filtered[filtered["run_status"].astype(str) == "completed"],
        group_columns=["model_name", "tiles_per_side", "num_tiles"],
    )
    ensure_dir(os.path.dirname(output_paths["aggregated_results"]) or ".")
    aggregated.to_csv(output_paths["aggregated_results"], index=False)
    plot_enhanced_confidence_results(filtered, output_paths["all_points_figure"], aggregate_over_seeds=False)
    plot_enhanced_confidence_results(filtered, output_paths["mean_by_seed_figure"], aggregate_over_seeds=True)
    return aggregated


def run_part2_enhanced_confidence_experiment(
    config: CVExperimentConfig,
    device: Optional[TorchDevice] = None,
) -> pd.DataFrame:
    """Run the enhanced Part 2 curriculum permutation-difficulty experiment."""

    output_paths = enhanced_confidence_output_paths(config.results_dir, config.figures_dir, "part2")
    resolved_device = device or get_device(config)
    baseline_rows = load_part1_enhanced_baseline_rows(config)
    curriculum_rows_by_key = _part2_curriculum_completed_rows(config, output_paths["raw_results"])
    save_rows([*baseline_rows, *curriculum_rows_by_key.values()], output_paths["raw_results"])

    session_start_time = perf_counter()
    for seed in ENHANCED_CONFIDENCE_SEEDS:
        with _enhanced_config_scope(config, part_name="part2", config_name=PART2_ENHANCED_CONFIG_NAME, seed=seed):
            seed_everything(seed, deterministic=config.deterministic)
            train_samples, validation_samples, _ = load_experiment_samples(config, seed=seed)
            records = build_enhanced_confidence_records_for_seed(seed)
            run_id = enhanced_confidence_run_id("part2", seed)
            rows, row_indices = initialize_ablation_result_rows(
                config=config,
                model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
                run_id=run_id,
                ablations=[PART2_CURRICULUM_ABLATION],
                records=records,
            )
            completed_for_seed = _completed_rows_for_seed(
                curriculum_rows_by_key,
                seed=seed,
                ablation_name=PART2_CURRICULUM_ABLATION_NAME,
            )
            for index, row in enumerate(rows):
                completed = completed_for_seed.get(_row_key(row))
                if completed is not None:
                    rows[index] = completed
            save_run_rows(
                rows=rows,
                output_path=output_paths["raw_results"],
                run_id=run_id,
                model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
            )

            missing_records = [record for record in records if _record_key(record) not in completed_for_seed]
            print(f"Part 2 enhanced seed {seed}: {len(missing_records)} missing curriculum run(s).")
            for record in missing_records:
                train_single_ablation_tile_run(
                    config=config,
                    model_name=ENHANCED_CONFIDENCE_MODEL_NAME,
                    run_id=run_id,
                    ablation=PART2_CURRICULUM_ABLATION,
                    record=record,
                    train_samples=train_samples,
                    validation_samples=validation_samples,
                    device=resolved_device,
                    rows=rows,
                    row_indices=row_indices,
                    session_start_time=session_start_time,
                    raw_results_output_path=output_paths["raw_results"],
                )

    raw_results = pd.read_csv(output_paths["raw_results"])
    expected_part2 = pd.concat(
        [
            _filter_expected_rows(
                raw_results[raw_results["ablation_name"].astype(str) == PART2_BASELINE_ABLATION_NAME],
                ablation_name=PART2_BASELINE_ABLATION_NAME,
            ),
            _filter_expected_rows(
                raw_results[raw_results["ablation_name"].astype(str) == PART2_CURRICULUM_ABLATION_NAME],
                ablation_name=PART2_CURRICULUM_ABLATION_NAME,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    completed = expected_part2[expected_part2["run_status"].astype(str) == "completed"].copy()
    aggregated = aggregate_accuracy(
        completed,
        group_columns=["model_name", "ablation_name", "tiles_per_side", "num_tiles"],
    )
    ensure_dir(os.path.dirname(output_paths["aggregated_results"]) or ".")
    aggregated.to_csv(output_paths["aggregated_results"], index=False)
    plot_enhanced_confidence_results(
        expected_part2,
        output_paths["all_points_figure"],
        aggregate_over_seeds=False,
        facet_column="ablation_name",
    )
    plot_enhanced_confidence_results(
        expected_part2,
        output_paths["mean_by_seed_figure"],
        aggregate_over_seeds=True,
        facet_column="ablation_name",
    )
    return aggregated


def _accuracy_column(frame: pd.DataFrame) -> str:
    if "best_val_accuracy" in frame.columns:
        return "best_val_accuracy"
    if "val_accuracy" in frame.columns:
        return "val_accuracy"
    raise ValueError("results must contain best_val_accuracy or val_accuracy")


def _prepare_plot_frame(raw_results: pd.DataFrame) -> pd.DataFrame:
    frame = raw_results.copy()
    if "run_status" in frame.columns:
        frame = frame[frame["run_status"].astype(str) == "completed"].copy()
    if "num_tiles" not in frame.columns:
        frame["num_tiles"] = [
            1 if pd.isna(tiles_per_side) else int(tiles_per_side) * int(tiles_per_side)
            for tiles_per_side in frame["tiles_per_side"]
        ]
    if "tile_permutation_name" not in frame.columns:
        frame["tile_permutation_name"] = "baseline"
    frame["tile_permutation_name"] = frame["tile_permutation_name"].fillna("baseline").astype(str)
    frame["base_permutation_name"] = frame["tile_permutation_name"].map(base_tile_permutation_name)
    return frame


def _tile_axis_positions(num_tiles_values: Sequence[object]) -> dict[int, int]:
    values = sorted({int(value) for value in num_tiles_values if not pd.isna(value)})
    return {value: index for index, value in enumerate(values)}


def _tile_axis_label(num_tiles: int) -> str:
    if num_tiles == 1:
        return "1x1"
    tiles_per_side = int(num_tiles**0.5)
    return f"{tiles_per_side}x{tiles_per_side}" if tiles_per_side * tiles_per_side == num_tiles else str(num_tiles)


def _variant_offsets(labels: Sequence[str]) -> dict[str, float]:
    grouped: dict[str, list[str]] = {}
    for label in labels:
        grouped.setdefault(base_tile_permutation_name(label), []).append(label)
    offsets: dict[str, float] = {}
    base_offsets = {"baseline": 0.0, "easy": -0.22, "medium": 0.0, "hard": 0.22}
    for base_name, names in grouped.items():
        names = sorted(set(names), key=lambda value: ENHANCED_TILE_PERMUTATION_IDS.get(value, 0))
        spread = [-0.045, 0.0, 0.045] if len(names) > 1 else [0.0]
        for index, name in enumerate(names):
            offsets[name] = base_offsets.get(base_name, 0.0) + spread[min(index, len(spread) - 1)]
    return offsets


def _color_by_label(labels: Sequence[str]) -> dict[str, str]:
    colors = plt.get_cmap("tab10")
    ordered = ["baseline", *ENHANCED_TILE_PERMUTATION_NAMES]
    return {
        label: colors(index % 10)
        for index, label in enumerate(label for label in ordered if label in set(labels))
    }


def _plot_condition_panel(
    ax: Axes,
    frame: pd.DataFrame,
    *,
    title: str,
    aggregate_over_seeds: bool,
) -> None:
    accuracy_column = _accuracy_column(frame)
    tile_positions = _tile_axis_positions(frame["num_tiles"])
    labels = sorted(
        set(frame["tile_permutation_name"]),
        key=lambda value: ENHANCED_TILE_PERMUTATION_IDS.get(value, 0),
    )
    offsets = _variant_offsets(labels)
    colors = _color_by_label(labels)

    if aggregate_over_seeds:
        grouped = (
            frame.groupby(["tile_permutation_name", "base_permutation_name", "num_tiles"], dropna=False)[
                accuracy_column
            ]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        labeled_permutations: set[str] = set()
        for _, row in grouped.iterrows():
            label = str(row["tile_permutation_name"])
            base_name = str(row["base_permutation_name"])
            num_tiles = int(row["num_tiles"])
            yerr = 0.0 if pd.isna(row["std"]) else float(row["std"])
            legend_label = label if label not in labeled_permutations else "_nolegend_"
            labeled_permutations.add(label)
            ax.errorbar(
                tile_positions[num_tiles] + offsets.get(label, 0.0),
                float(row["mean"]),
                yerr=yerr,
                marker=ENHANCED_MARKERS.get(base_name, "s"),
                color=colors.get(label),
                linestyle="None",
                capsize=3,
                markersize=6,
                label=legend_label,
            )
    else:
        labeled_permutations: set[str] = set()
        for _, row in frame.iterrows():
            label = str(row["tile_permutation_name"])
            base_name = str(row["base_permutation_name"])
            num_tiles = int(row["num_tiles"])
            legend_label = label if label not in labeled_permutations else "_nolegend_"
            labeled_permutations.add(label)
            ax.scatter(
                tile_positions[num_tiles] + offsets.get(label, 0.0),
                float(row[accuracy_column]),
                marker=ENHANCED_MARKERS.get(base_name, "s"),
                color=colors.get(label),
                alpha=0.58,
                s=44 if base_name != "baseline" else 58,
                label=legend_label,
            )
        overall = frame.groupby("num_tiles", dropna=False)[accuracy_column].agg(["mean", "std", "count"]).reset_index()
        ax.errorbar(
            [tile_positions[int(num_tiles)] for num_tiles in overall["num_tiles"]],
            overall["mean"],
            yerr=overall["std"].fillna(0.0),
            marker="s",
            color="black",
            linewidth=1.8,
            capsize=4,
            label="grid mean",
        )

    ax.set_title(title)
    ax.set_xticks(list(tile_positions.values()), [_tile_axis_label(value) for value in tile_positions])
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Best validation accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.25)


def plot_enhanced_confidence_results(
    raw_results: pd.DataFrame,
    output_path: str,
    *,
    aggregate_over_seeds: bool,
    facet_column: str | None = None,
) -> None:
    """Plot enhanced confidence raw points or seed-aggregated means."""

    frame = _prepare_plot_frame(raw_results)
    if frame.empty:
        print(f"Skipping enhanced plot, no completed rows: {output_path}")
        return

    ensure_dir(os.path.dirname(output_path) or ".")
    if facet_column and facet_column in frame.columns:
        facet_values = [
            value
            for value in [PART2_BASELINE_ABLATION_NAME, PART2_CURRICULUM_ABLATION_NAME]
            if value in set(frame[facet_column].astype(str))
        ]
        if not facet_values:
            facet_values = sorted(frame[facet_column].dropna().astype(str).unique())
    else:
        facet_values = ["Enhanced confidence"]
        facet_column = None

    fig, axes = plt.subplots(1, len(facet_values), figsize=(9 * len(facet_values), 5.5), squeeze=False)
    for axis, facet_value in zip(axes[0], facet_values):
        ax = _as_axes(axis)
        panel = frame if facet_column is None else frame[frame[facet_column].astype(str) == facet_value]
        _plot_condition_panel(
            ax,
            panel,
            title=str(facet_value),
            aggregate_over_seeds=aggregate_over_seeds,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)))
    title = "Mean by permutation across seeds" if aggregate_over_seeds else "All enhanced runs with grid means"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
