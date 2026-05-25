"""Part 1 model and tile-permutation experiments."""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any, Optional, Sequence

import pandas as pd
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader

from src.evaluation.experiment_results import get_device, load_experiment_samples, plot_accuracy_vs_tiles
from src.experiments.results import (
    aggregate_accuracy,
    experiment_intermediate_figure_path,
    experiment_output_paths,
    save_aggregated_accuracy,
    save_rows,
    save_run_rows,
)
import src.experiments.training_runs as _training_runs

# Notebook kernels often keep imported dependencies alive across saved source edits.
# Refresh the shared training helpers before binding their functions below.
_training_runs = importlib.reload(_training_runs)

from src.experiments.training_runs import (
    build_pending_training_result_row,
    build_experiment_run_id,
    build_training_run_spec,
    checkpoint_dir_path,
    checkpoints_enabled,
    format_dataloader_summary,
    format_elapsed_time,
    format_stage_summary,
    train_model_and_save_progress,
    training_result_row_key,
)
from src.models.registry import TIMM_MODEL_IDS
from src.preprocessing.dataloaders import build_dataloaders
from src.preprocessing.image_transforms import make_tile_compatible_image_size
from src.preprocessing.samples import Sample
from src.preprocessing.tile_permutations import TilePermutationRecord, build_tile_permutation_records
from src.utils.config import CVExperimentConfig


def get_executable_tile_permutation_records(records: Sequence[TilePermutationRecord]) -> list[TilePermutationRecord]:
    """Return tile-permutation records that correspond to actual runs."""

    return list(records)


def initialize_tile_permutation_result_rows(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    records: Sequence[TilePermutationRecord],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], int]]:
    """Create pending result rows and their update indexes for a model."""

    rows = [
        build_pending_training_result_row(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            seed=seed,
        )
        for record in records
    ]
    row_indices = {training_result_row_key(row): index for index, row in enumerate(rows)}
    return rows, row_indices


def build_tile_permutation_dataloaders(
    *,
    config: CVExperimentConfig,
    model_name: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    record: TilePermutationRecord,
) -> tuple[DataLoader, DataLoader]:
    """Build dataloaders for one Part 1 tile-permutation run."""

    fixed_input_size = config.image_size if model_name in TIMM_MODEL_IDS else None
    return build_dataloaders(
        train_samples=train_samples,
        val_samples=validation_samples,
        image_size=config.image_size,
        tiles_per_side=record.tiles_per_side or 1,
        tile_permutation=record.tile_permutation,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        standard_augmentation=False,
        output_image_size=fixed_input_size,
    )


def _print_tile_permutation_run_header(
    *,
    record_index: int,
    num_records: int,
    model_name: str,
    record: TilePermutationRecord,
    stage_summary: str,
) -> None:
    print()
    print("=" * 80)
    print(
        f"[{record_index}/{num_records}]\nmodel={model_name}, "
        f"tiles_per_side={record.tiles_per_side}, tile_permutation_id={record.tile_permutation_id}, "
        f"tile_permutation_name={record.tile_permutation_name}, seed={record.tile_permutation_seed}\n{stage_summary}"
    )


def train_single_tile_permutation_run(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    record: TilePermutationRecord,
    seed: int,
    device: TorchDevice,
    rows: list[dict[str, Any]],
    row_indices: dict[tuple[Any, ...], int],
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
) -> None:
    """Train one model on one tile-permutation record and update result rows."""

    run_start_time = perf_counter()
    resolved_session_start_time = session_start_time if session_start_time is not None else run_start_time
    print("Building dataloaders...")
    train_loader, validation_loader = build_tile_permutation_dataloaders(
        config=config,
        model_name=model_name,
        train_samples=train_samples,
        validation_samples=validation_samples,
        record=record,
    )
    print(f"Built dataloaders: {format_dataloader_summary(train_loader, validation_loader)}.")

    tiles_label = record.tiles_per_side or 1
    expected_input_size = (
        config.image_size
        if model_name in TIMM_MODEL_IDS
        else make_tile_compatible_image_size(config.image_size, record.tiles_per_side or 1)
    )
    permutation_label = (
        f"{record.tile_permutation_name} permutation"
        if record.tile_permutation_name
        else f"permutation {record.tile_permutation_id}"
    )
    spec = build_training_run_spec(
        config=config,
        model_name=model_name,
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        run_id=run_id,
        record=record,
        seed=seed,
        overrides={"pretrained": config.pretrained},
        progress_desc=f"{model_name} {tiles_label}x{tiles_label} {permutation_label}",
        expected_input_size=expected_input_size,
    )
    train_model_and_save_progress(
        spec=spec,
        config=config,
        run_id=run_id,
        record=record,
        seed=seed,
        rows=rows,
        row_index=row_indices[(record.tiles_per_side or 0, record.tile_permutation_id)],
        raw_results_output_path=raw_results_output_path,
    )
    print(f"Current run runtime: {format_elapsed_time(perf_counter() - run_start_time)}")
    print(f"Total training runtime: {format_elapsed_time(perf_counter() - resolved_session_start_time)}")


def train_model_on_tile_permutation_records(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[Sample],
    validation_samples: Sequence[Sample],
    tile_permutation_records: Sequence[TilePermutationRecord],
    seed: int,
    device: TorchDevice,
    session_start_time: Optional[float] = None,
    raw_results_output_path: Optional[str] = None,
    intermediate_figure_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train one model across tile-permutation records and collect result rows."""

    resolved_session_start_time = session_start_time if session_start_time is not None else perf_counter()
    executable_records = get_executable_tile_permutation_records(tile_permutation_records)
    rows, row_indices = initialize_tile_permutation_result_rows(
        config=config,
        run_id=run_id,
        model_name=model_name,
        records=executable_records,
        seed=seed,
    )

    if raw_results_output_path:
        save_run_rows(rows=rows, output_path=raw_results_output_path, run_id=run_id, model_name=model_name)
        print(f"Saved {len(rows)} pending row(s) for model '{model_name}'.")

    for record_index, record in enumerate(executable_records, start=1):
        stage_summary = format_stage_summary([("standard", int(config.epochs))])
        _print_tile_permutation_run_header(
            record_index=record_index,
            num_records=len(executable_records),
            model_name=model_name,
            record=record,
            stage_summary=stage_summary,
        )
        train_single_tile_permutation_run(
            config=config,
            model_name=model_name,
            run_id=run_id,
            train_samples=train_samples,
            validation_samples=validation_samples,
            record=record,
            seed=seed,
            device=device,
            rows=rows,
            row_indices=row_indices,
            raw_results_output_path=raw_results_output_path,
            session_start_time=resolved_session_start_time,
        )
    if intermediate_figure_output_path:
        raw_model_results = pd.DataFrame(rows)
        aggregated_model_results = aggregate_accuracy(
            raw_model_results,
            group_columns=["model_name", "tiles_per_side", "num_tiles"],
        )
        plot_accuracy_vs_tiles(
            aggregated_model_results,
            intermediate_figure_output_path,
            raw_results=raw_model_results,
        )
        print(f"Saved intermediate model plot: {intermediate_figure_output_path}")
    return rows


def run_part1_experiments(config: CVExperimentConfig, device: Optional[TorchDevice] = None) -> pd.DataFrame:
    """Run Part 1 model comparison experiments and save raw/aggregated outputs."""

    session_start_time = perf_counter()
    resolved_device = device or get_device(config)
    train_samples, validation_samples, _ = load_experiment_samples(config, seed=config.seed)
    tile_permutation_records = build_tile_permutation_records(
        tiles_per_side_values=config.tiles_per_side_values,
        num_tile_permutations=config.num_tile_permutations,
        seed=config.seed,
        include_baseline=True,
    )
    run_id = build_experiment_run_id(config)
    output_paths = experiment_output_paths(config.results_dir, config.figures_dir, config.part)

    print(f"Raw results path: {output_paths['raw_results']}")
    if checkpoints_enabled(config):
        print(f"Checkpoint directory: {checkpoint_dir_path(config=config, run_id=run_id)}")
    else:
        print("Checkpointing disabled outside Google Colab.")

    rows: list[dict[str, Any]] = []
    for model_name in config.model_names:
        rows.extend(
            train_model_on_tile_permutation_records(
                config=config,
                model_name=model_name,
                run_id=run_id,
                train_samples=train_samples,
                validation_samples=validation_samples,
                tile_permutation_records=tile_permutation_records,
                seed=config.seed,
                device=resolved_device,
                raw_results_output_path=output_paths["raw_results"],
                intermediate_figure_output_path=experiment_intermediate_figure_path(
                    config.figures_dir,
                    config.part,
                    f"accuracy_vs_tiles_{model_name}",
                ),
                session_start_time=session_start_time,
            )
        )

    save_rows(rows, output_paths["raw_results"])
    raw_results = pd.DataFrame(rows)
    aggregated_results = save_aggregated_accuracy(
        raw_results,
        group_columns=["model_name", "tiles_per_side", "num_tiles"],
        output_path=output_paths["aggregated_results"],
    )
    plot_accuracy_vs_tiles(aggregated_results, output_paths["figure"], raw_results=raw_results)
    return aggregated_results
