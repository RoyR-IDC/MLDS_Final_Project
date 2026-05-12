"""Part 1 model and tile-permutation experiments."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
from torch import nn
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader

from src.evaluation.experiment_results import get_device, load_experiment_samples, plot_accuracy_vs_tiles
from src.experiments.results import (
    build_result_row,
    build_training_result_row,
    experiment_output_paths,
    save_aggregated_accuracy,
    save_rows,
    save_run_rows,
)
from src.models.factory import get_model
from src.preprocessing.dataloaders import build_dataloaders
from src.preprocessing.tile_permutations import TilePermutationRecord, build_tile_permutation_records
from src.training.run import CheckpointConfig, TrainingConfig, TrainingResult, TrainingRunSpec
from src.training.trainer import ModelTrainer
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir


def get_executable_tile_permutation_records(records: Sequence[TilePermutationRecord]) -> list[TilePermutationRecord]:
    """Return tile-permutation records that correspond to actual runs."""

    return list(records)


def _clean_path_part(value: Any) -> str:
    return str(value).replace(os.sep, "_").replace(" ", "_").replace(":", "_")


def _checkpoint_config(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    ablation_name: str | None = None,
) -> CheckpointConfig:
    outputs_dir = getattr(config, "outputs_dir", "") or os.path.dirname(getattr(config, "results_dir", "")) or "outputs"
    checkpoint_dir = ensure_dir(os.path.join(outputs_dir, "checkpoints", str(config.part), run_id))
    name_parts = [
        _clean_path_part(model_name),
        _clean_path_part(ablation_name) if ablation_name else None,
        f"tiles_{record.tiles_per_side or 1}",
        f"perm_{record.tile_permutation_id}",
    ]
    stem = "__".join(part for part in name_parts if part)
    return CheckpointConfig(
        best_path=os.path.join(checkpoint_dir, f"{stem}__best.pt"),
        last_path=os.path.join(checkpoint_dir, f"{stem}__last.pt"),
    )


def _training_config(config: CVExperimentConfig) -> TrainingConfig:
    return TrainingConfig(
        epochs=int(getattr(config, "epochs", 1)),
        optimizer_name=str(getattr(config, "optimizer", "adamw")),
        learning_rate=float(getattr(config, "learning_rate", 0.0003)),
        weight_decay=float(getattr(config, "weight_decay", 0.0)),
        use_amp=bool(getattr(config, "use_amp", False)),
    )


def build_training_spec(
    *,
    config: CVExperimentConfig,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: TorchDevice,
    run_id: str,
    record: TilePermutationRecord,
    seed: int,
    overrides: Optional[Mapping[str, Any]] = None,
    ablation_name: str | None = None,
    progress_desc: Optional[str] = None,
) -> TrainingRunSpec:
    """Build the shared OOP training specification for one run."""

    run_options = dict(overrides or {})
    print(f"Building model '{model_name}'...")
    model = get_model(
        model_name,
        num_classes=int(getattr(config, "num_classes", 2)),
        pretrained=bool(run_options.get("pretrained", getattr(config, "pretrained", False))),
        device=device,
        freeze_backbone=bool(run_options.get("freeze_backbone", getattr(config, "freeze_backbone", False))),
    )
    metadata = {
        "part": config.part,
        "config_name": config.config_name,
        "run_id": run_id,
        "model_name": model_name,
        "ablation_name": ablation_name,
        "tiles_per_side": record.tiles_per_side,
        "tile_permutation_id": record.tile_permutation_id,
        "tile_permutation_seed": record.tile_permutation_seed,
        "seed": seed,
    }
    return TrainingRunSpec(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config=_training_config(config),
        checkpoint_config=_checkpoint_config(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            ablation_name=ablation_name,
        ),
        metadata=metadata,
        progress_desc=progress_desc or f"{model_name} epochs",
    )


def build_pending_result_row(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    seed: int,
    ablation_name: str | None = None,
) -> dict[str, Any]:
    """Build an empty result row before a training run starts."""

    result = TrainingResult.pending(
        model_name=model_name,
        metadata={
            "part": config.part,
            "config_name": config.config_name,
            "run_id": run_id,
            "model_name": model_name,
            "ablation_name": ablation_name,
            "seed": seed,
        },
    )
    return build_result_row(
        config=config,
        run_id=run_id,
        model_name=model_name,
        record=record,
        seed=seed,
        metrics=result.latest_metrics(),
        ablation_name=ablation_name,
    )


def _result_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    tiles_per_side = 0 if pd.isna(row["tiles_per_side"]) else int(row["tiles_per_side"])
    ablation_name = row.get("ablation_name")
    if ablation_name is None or (isinstance(ablation_name, float) and pd.isna(ablation_name)):
        return tiles_per_side, int(row["tile_permutation_id"])
    return str(ablation_name), tiles_per_side, int(row["tile_permutation_id"])


def _run_training_with_progress_saves(
    *,
    spec: TrainingRunSpec,
    config: CVExperimentConfig,
    run_id: str,
    record: TilePermutationRecord,
    seed: int,
    rows: list[dict[str, Any]],
    row_index: int,
    raw_results_output_path: str | None,
    ablation_name: str | None = None,
) -> TrainingResult:
    """Train one spec and persist row updates whenever the trainer reports progress."""

    def persist_progress(result: TrainingResult) -> None:
        rows[row_index] = build_training_result_row(
            config=config,
            run_id=run_id,
            model_name=spec.model_name,
            record=record,
            seed=seed,
            result=result,
            ablation_name=ablation_name,
        )
        if raw_results_output_path:
            save_run_rows(
                rows=rows,
                output_path=raw_results_output_path,
                run_id=run_id,
                model_name=spec.model_name,
            )

    return ModelTrainer(spec).fit(on_progress=persist_progress)


def collect_model_tile_permutation_results(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[tuple[str, int]],
    validation_samples: Sequence[tuple[str, int]],
    tile_permutation_records: Sequence[TilePermutationRecord],
    seed: int,
    device: TorchDevice,
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
        save_run_rows(rows=rows, output_path=raw_results_output_path, run_id=run_id, model_name=model_name)
        print(f"Saved {len(rows)} pending row(s) for model '{model_name}' to {raw_results_output_path}.")

    for record_index, record in enumerate(executable_records, start=1):
        print()
        print("=" * 80)
        print(
            f"[{record_index}/{len(executable_records)}] model={model_name}, "
            f"tiles_per_side={record.tiles_per_side}, tile_permutation_id={record.tile_permutation_id}, "
            f"seed={record.tile_permutation_seed}"
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
        print(f"Built dataloaders: {len(train_loader)} train batches, {len(validation_loader)} validation batches.")
        tiles_label = record.tiles_per_side or 1
        progress_desc = f"{model_name} {tiles_label}x{tiles_label} permutation {record.tile_permutation_id}"
        spec = build_training_spec(
            config=config,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            run_id=run_id,
            record=record,
            seed=seed,
            overrides={"pretrained": config.pretrained},
            progress_desc=progress_desc,
        )
        checkpoint_config = getattr(spec, "checkpoint_config", None)
        print(f"Best checkpoint path: {getattr(checkpoint_config, 'best_path', None)}")
        print(f"Raw results path: {raw_results_output_path}")
        _run_training_with_progress_saves(
            spec=spec,
            config=config,
            run_id=run_id,
            record=record,
            seed=seed,
            rows=rows,
            row_index=row_indices[(record.tiles_per_side or 0, record.tile_permutation_id)],
            raw_results_output_path=raw_results_output_path,
        )
    return rows


def run_part1_experiments(config: CVExperimentConfig, device: Optional[TorchDevice] = None) -> pd.DataFrame:
    """Run Part 1 model comparison experiments and save raw/aggregated outputs."""

    resolved_device = device or get_device(config)
    train_samples, validation_samples, _ = load_experiment_samples(config, seed=config.seed)
    tile_permutation_records = build_tile_permutation_records(
        tiles_per_side_values=config.tiles_per_side_values,
        num_tile_permutations=config.num_tile_permutations,
        seed=config.seed,
        include_baseline=True,
    )
    run_id = datetime.now(timezone.utc).strftime("part1_%Y%m%d_%H%M%S")
    output_paths = experiment_output_paths(config.results_dir, config.figures_dir, config.part)

    rows: list[dict[str, Any]] = []
    for model_name in config.model_names:
        rows.extend(
            collect_model_tile_permutation_results(
                config=config,
                model_name=model_name,
                run_id=run_id,
                train_samples=train_samples,
                validation_samples=validation_samples,
                tile_permutation_records=tile_permutation_records,
                seed=config.seed,
                device=resolved_device,
                raw_results_output_path=output_paths["raw_results"],
            )
        )

    save_rows(rows, output_paths["raw_results"])
    raw_results = pd.DataFrame(rows)
    aggregated_results = save_aggregated_accuracy(
        raw_results,
        group_columns=["model_name", "tiles_per_side", "num_tiles"],
        output_path=output_paths["aggregated_results"],
    )
    plot_accuracy_vs_tiles(aggregated_results, output_paths["figure"])
    return aggregated_results
