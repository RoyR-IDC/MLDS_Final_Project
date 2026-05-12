"""Generic training steps shared by notebook-owned experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.experiment_results import (
    build_result_row,
    experiment_output_paths,
    get_device,
    load_experiment_samples,
    load_part1_model_baseline_aggregated,
    load_part1_model_baseline_raw_rows,
    plot_ablation_results,
    save_aggregated_accuracy,
    save_rows,
)
from src.models.factory import get_model
from src.preprocessing.dataloaders import build_dataloaders
from src.preprocessing.tile_permutations import TilePermutationRecord, build_tile_permutation_records
from src.training.engine import TrainingRunComponents, build_optimizer, train_and_validate
from src.utils.config import CVExperimentConfig


@dataclass
class ModelTrainingComponents:
    """Model-owned components built from one experiment configuration."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    criterion: nn.Module


def build_training_components(
    config: CVExperimentConfig,
    model_name: str,
    device: torch.device,
    overrides: Optional[Mapping[str, Any]] = None,
) -> ModelTrainingComponents:
    """Build model, optimizer, and criterion objects for one training run.

    Args:
        config: Normalized experiment configuration.
        model_name: Name of the model architecture to train.
        device: Device where the model should run.
        overrides: Optional per-run settings such as pretrained or freeze flags.

    Returns:
        Model, optimizer, and criterion for this training configuration.
    """

    run_options = dict(overrides or {})
    model = get_model(
        model_name,
        num_classes=config.num_classes,
        pretrained=bool(run_options.get("pretrained", config.pretrained)),
        device=device,
        freeze_backbone=bool(run_options.get("freeze_backbone", getattr(config, "freeze_backbone", False))),
    )
    optimizer = build_optimizer(
        model,
        name=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    components = ModelTrainingComponents(model=model, optimizer=optimizer, criterion=criterion)
    return components


def train_and_evaluate_model_configuration(
    config: CVExperimentConfig,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    overrides: Optional[Mapping[str, Any]] = None,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    """Train and validate one model configuration and return generated metrics.

    Args:
        config: Normalized experiment configuration.
        model_name: Name of the model architecture to train.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        device: Device where training should run.
        overrides: Optional per-run model settings.

    Returns:
        Training and validation metrics from the shared engine.
    """

    print(f"Building training components for model '{model_name}'...")
    components = build_training_components(config, model_name, device, overrides=overrides)
    print(f"Finished building training components for model '{model_name}'.")
    print(f"Training model '{model_name}' for {config.epochs} epoch(s)...")
    training_run = TrainingRunComponents(
        model=components.model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=components.optimizer,
        criterion=components.criterion,
        device=device,
        epochs=config.epochs,
        use_amp=config.use_amp,
        progress_desc=progress_desc or f"{model_name} epochs",
    )
    metrics = train_and_validate(training_run)
    print(f"Finished training model '{model_name}'. Metrics: {metrics}")
    return metrics


def save_model_tile_permutation_progress(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_path: str,
    run_id: str,
    model_name: str,
) -> None:
    """Save completed rows for one model while preserving other model results."""

    if not rows:
        return

    existing_rows: list[dict[str, Any]] = []
    if os.path.exists(output_path):
        existing_results = pd.read_csv(output_path)
        if {"run_id", "model_name"}.issubset(existing_results.columns):
            existing_results = existing_results[
                ~(
                    (existing_results["run_id"].astype(str) == str(run_id))
                    & (existing_results["model_name"].astype(str) == str(model_name))
                )
            ]
        existing_rows = existing_results.to_dict("records")

    save_rows([*existing_rows, *rows], output_path)


def build_pending_result_row(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    seed: int,
) -> dict[str, Any]:
    """Build an empty result row before a tile-permutation training run starts."""

    return build_result_row(
        config=config,
        run_id=run_id,
        model_name=model_name,
        record=record,
        seed=seed,
        metrics={
            "run_status": "pending",
            "train_loss": None,
            "train_accuracy": None,
            "val_loss": None,
            "val_accuracy": None,
            "best_val_accuracy": None,
            "training_duration_seconds": None,
        },
    )


def _result_row_key(row: Mapping[str, Any]) -> tuple[int, int]:
    """Return the per-model tile-permutation key for a raw result row."""

    return int(row["tiles_per_side"]), int(row["tile_permutation_id"])


def collect_model_tile_permutation_results(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[tuple[str, int]],
    validation_samples: Sequence[tuple[str, int]],
    tile_permutation_records: Sequence[TilePermutationRecord],
    seed: int,
    device: torch.device,
    raw_results_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train one model across tile-permutation records and collect result rows."""

    executable_records = get_executable_tile_permutation_records(tile_permutation_records)
    rows = [
        build_pending_result_row(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            seed=seed,
        )
        for record in executable_records
    ]
    row_indices = {_result_row_key(row): index for index, row in enumerate(rows)}

    if raw_results_output_path:
        save_model_tile_permutation_progress(
            rows=rows,
            output_path=raw_results_output_path,
            run_id=run_id,
            model_name=model_name,
        )
        print(
            f"Saved {len(rows)} pending placeholder row(s) for model '{model_name}' "
            f"to {raw_results_output_path}."
        )

    for record_index, record in enumerate(executable_records, start=1):
        print()
        print("=" * 80)
        print(
            f"[{record_index}/{len(executable_records)}] Running tile permutation "
            f"tiles_per_side={record.tiles_per_side}, "
            f"tile_permutation_id={record.tile_permutation_id}, seed={record.tile_permutation_seed}"
        )
        print("Building dataloaders...")
        train_loader, validation_loader = build_dataloaders(
            train_samples=train_samples,
            val_samples=validation_samples,
            image_size=config.image_size,
            tiles_per_side=record.tiles_per_side or 1,
            tile_permutation=record.tile_permutation,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            standard_augmentation=False,
        )
        print(
            "Finished building dataloaders: "
            f"{len(train_loader)} train batches, {len(validation_loader)} validation batches."
        )
        print(f"Training model '{model_name}' on the current tile permutation...")
        progress_desc = (
            f"{model_name} "
            f"{record.tiles_per_side}x{record.tiles_per_side} "
            f"permutation {record.tile_permutation_id}"
        )
        training_start = perf_counter()
        metrics = train_and_evaluate_model_configuration(
            config=config,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            overrides={"pretrained": config.pretrained},
            progress_desc=progress_desc,
        )
        training_duration_seconds = perf_counter() - training_start
        metrics["training_duration_seconds"] = training_duration_seconds
        metrics["run_status"] = "completed"
        print(
            f"Done training model '{model_name}' on tile_permutation_id={record.tile_permutation_id} "
            f"in {training_duration_seconds:.2f} seconds."
        )
        row = build_result_row(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            seed=seed,
            metrics=metrics,
        )
        rows[row_indices[(record.tiles_per_side, record.tile_permutation_id)]] = row
        if raw_results_output_path:
            save_model_tile_permutation_progress(
                rows=rows,
                output_path=raw_results_output_path,
                run_id=run_id,
                model_name=model_name,
            )
            print(f"Saved {len(rows)} completed row(s) for model '{model_name}' to {raw_results_output_path}.")
    return rows


def get_executable_tile_permutation_records(records: Sequence[TilePermutationRecord]) -> list[TilePermutationRecord]:
    """Return tile-permutation records that correspond to actual runs."""

    return list(records)


def collect_part2_ablation_results(
    *,
    config: CVExperimentConfig,
    ablations: Sequence[Mapping[str, Any]],
    train_samples: Sequence[tuple[str, int]],
    validation_samples: Sequence[tuple[str, int]],
    tile_permutation_records: Sequence[TilePermutationRecord],
    device: torch.device,
    run_id: str,
    raw_results_output_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Train all Part 2 improvement ablations across tile-permutation records."""

    model_name = getattr(config, "model_name", config.model_names[0])
    executable_records = get_executable_tile_permutation_records(tile_permutation_records)
    rows: list[dict[str, Any]] = []
    row_indices: dict[tuple[str, int, int], int] = {}

    for ablation in ablations:
        for record in executable_records:
            row = build_pending_result_row(
                config=config,
                run_id=run_id,
                model_name=model_name,
                record=record,
                seed=config.seed,
            )
            row["ablation_name"] = ablation["name"]
            row_indices[(str(ablation["name"]), record.tiles_per_side, record.tile_permutation_id)] = len(rows)
            rows.append(row)

    if raw_results_output_path:
        save_model_tile_permutation_progress(
            rows=rows,
            output_path=raw_results_output_path,
            run_id=run_id,
            model_name=model_name,
        )
        print(
            f"Saved {len(rows)} pending placeholder row(s) for model '{model_name}' "
            f"to {raw_results_output_path}."
        )

    for ablation in ablations:
        print()
        print("=" * 80)
        print(f"Running ablation: {ablation['name']}")

        for record_index, record in enumerate(executable_records, start=1):
            print()
            print(
                f"[{record_index}/{len(executable_records)}] "
                f"tiles_per_side={record.tiles_per_side}, "
                f"tile_permutation_id={record.tile_permutation_id}, "
                f"seed={record.tile_permutation_seed}"
            )
            train_loader, validation_loader = build_dataloaders(
                train_samples=train_samples,
                val_samples=validation_samples,
                image_size=config.image_size,
                tiles_per_side=record.tiles_per_side or 1,
                tile_permutation=record.tile_permutation,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                standard_augmentation=bool(ablation["use_standard_augmentation"]),
            )
            progress_desc = (
                f"{model_name} "
                f"{ablation['name']} "
                f"{record.tiles_per_side}x{record.tiles_per_side} "
                f"permutation {record.tile_permutation_id}"
            )
            training_start = perf_counter()
            metrics = train_and_evaluate_model_configuration(
                config=config,
                model_name=model_name,
                train_loader=train_loader,
                val_loader=validation_loader,
                device=device,
                overrides={
                    "pretrained": bool(ablation["use_pretrained"]),
                    "freeze_backbone": bool(ablation["freeze_backbone"]),
                },
                progress_desc=progress_desc,
            )
            training_duration_seconds = perf_counter() - training_start
            metrics["training_duration_seconds"] = training_duration_seconds
            metrics["run_status"] = "completed"
            print(
                f"Done training model '{model_name}' for ablation={ablation['name']} "
                f"on tile_permutation_id={record.tile_permutation_id} "
                f"in {training_duration_seconds:.2f} seconds."
            )
            row = build_result_row(
                config=config,
                run_id=run_id,
                model_name=model_name,
                record=record,
                seed=config.seed,
                metrics=metrics,
            )
            row["ablation_name"] = ablation["name"]
            rows[row_indices[(str(ablation["name"]), record.tiles_per_side, record.tile_permutation_id)]] = row
            if raw_results_output_path:
                save_model_tile_permutation_progress(
                    rows=rows,
                    output_path=raw_results_output_path,
                    run_id=run_id,
                    model_name=model_name,
                )
                print(f"Saved {len(rows)} completed row(s) for model '{model_name}' to {raw_results_output_path}.")
    return rows


def run_part2_improvement_experiments(
    config: CVExperimentConfig,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Run Part 2 ablations, save raw/aggregated results, and write the comparison plot."""

    model_name = getattr(config, "model_name", config.model_names[0])
    resolved_device = device or get_device(config)
    train_samples, validation_samples, _ = load_experiment_samples(config, seed=config.seed)
    tile_permutation_records = build_tile_permutation_records(
        tiles_per_side_values=config.tiles_per_side_values,
        num_tile_permutations=config.num_tile_permutations,
        seed=config.seed,
        include_baseline=True,
    )
    run_id = datetime.now(timezone.utc).strftime("part2_%Y%m%d_%H%M%S")
    output_paths = experiment_output_paths(config.results_dir, config.figures_dir, config.part)

    rows = load_part1_model_baseline_raw_rows(config, model_name)
    rows.extend(
        collect_part2_ablation_results(
            config=config,
            ablations=getattr(config, "ablations"),
            train_samples=train_samples,
            validation_samples=validation_samples,
            tile_permutation_records=tile_permutation_records,
            device=resolved_device,
            run_id=run_id,
            raw_results_output_path=output_paths["raw_results"],
        )
    )

    save_rows(rows, output_paths["raw_results"])
    raw_results = pd.DataFrame(rows)
    aggregated_results = save_aggregated_accuracy(
        raw_results,
        group_columns=["model_name", "ablation_name", "tiles_per_side", "num_tiles"],
        output_path=output_paths["aggregated_results"],
    )

    part1_baseline_aggregated = load_part1_model_baseline_aggregated(config, model_name)
    has_regular_baseline = (
        "ablation_name" in aggregated_results.columns
        and (aggregated_results["ablation_name"] == "regular_part1").any()
    )
    if not has_regular_baseline and not part1_baseline_aggregated.empty:
        aggregated_results = pd.concat([part1_baseline_aggregated, aggregated_results], ignore_index=True, sort=False)
        aggregated_results.to_csv(output_paths["aggregated_results"], index=False)

    plot_ablation_results(aggregated_results, output_paths["figure"])
    return aggregated_results
