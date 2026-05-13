"""Shared helpers for experiment training runs."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
import os
from typing import Any, Mapping, Optional

import pandas as pd
from torch import nn
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader

from src.experiments.results import build_result_row, build_training_result_row, save_run_rows
from src.models.factory import get_model
from src.preprocessing.augmentations import BatchAugmentation
from src.preprocessing.tile_permutations import TilePermutationRecord
from src.training.curriculum import CurriculumSchedule
from src.training.run import CheckpointConfig, TrainingConfig, TrainingResult, TrainingRunSpec
from src.training.trainer import ModelTrainer
from src.utils.config import CVExperimentConfig
from src.utils.io import ensure_dir


def format_elapsed_time(seconds: float) -> str:
    """Return a compact human-readable elapsed time string."""

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def format_epoch_count(epochs: int) -> str:
    """Return an epoch count with singular/plural text."""

    return f"{epochs} epoch" if epochs == 1 else f"{epochs} epochs"


def format_stage_summary(stages: Mapping[str, int] | Sequence[tuple[str, int]]) -> str:
    """Return a one-line summary of training stages."""

    items = list(stages.items()) if isinstance(stages, MappingABC) else list(stages)
    stage_text = ", ".join(f"{name}: {format_epoch_count(epochs)}" for name, epochs in items)
    return f"stages={len(items)} [{stage_text}]"


def format_dataloader_summary(train_loader: DataLoader, val_loader: DataLoader) -> str:
    """Return sample and batch counts for train and validation loaders."""

    return (
        f"\ttrain={len(train_loader.dataset)} samples in {len(train_loader)} batches"
        f"\n\tvalidation={len(val_loader.dataset)} samples in {len(val_loader)} batches"
    )


def format_stage_dataloader_summary(stages: Sequence[Any]) -> str:
    """Return sample and batch counts for each stage train loader."""

    return "; ".join(
        f"{stage.name}={len(stage.train_loader.dataset)} samples / {len(stage.train_loader)} batches"
        for stage in stages
    )


def clean_checkpoint_name_part(value: Any) -> str:
    """Return a value safe to use inside a checkpoint filename."""

    return str(value).replace(os.sep, "_").replace(" ", "_").replace(":", "_")


def checkpoint_dir_path(*, config: CVExperimentConfig, run_id: str) -> str:
    """Return the shared checkpoint directory for one experiment session."""

    outputs_dir = (
        getattr(config, "outputs_dir", "")
        or os.path.dirname(getattr(config, "results_dir", ""))
        or "outputs"
    )
    return ensure_dir(os.path.join(outputs_dir, "checkpoints", str(config.part), run_id))


def checkpoints_enabled(config: CVExperimentConfig) -> bool:
    """Return whether model checkpoints should be written for this runtime."""

    return bool(getattr(config, "using_google_colab", False))


def build_checkpoint_config(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    ablation_name: str | None = None,
) -> CheckpointConfig:
    """Build checkpoint destinations for one experiment run."""

    if not checkpoints_enabled(config):
        return CheckpointConfig(save_best=False, save_last=False)

    checkpoint_dir = checkpoint_dir_path(config=config, run_id=run_id)
    name_parts = [
        clean_checkpoint_name_part(model_name),
        clean_checkpoint_name_part(ablation_name) if ablation_name else None,
        f"tiles_{record.tiles_per_side or 1}",
        f"perm_{record.tile_permutation_id}",
    ]
    stem = "__".join(part for part in name_parts if part)
    return CheckpointConfig(
        best_path=os.path.join(checkpoint_dir, f"{stem}__best.pt"),
        last_path=os.path.join(checkpoint_dir, f"{stem}__last.pt"),
    )


def build_training_config(config: CVExperimentConfig) -> TrainingConfig:
    """Build training hyperparameters from an experiment config."""

    return TrainingConfig(
        epochs=int(getattr(config, "epochs", 1)),
        optimizer_name=str(getattr(config, "optimizer", "adamw")),
        learning_rate=float(getattr(config, "learning_rate", 0.0003)),
        weight_decay=float(getattr(config, "weight_decay", 0.0)),
        use_amp=bool(getattr(config, "use_amp", False)),
    )


def build_training_metadata(
    *,
    config: CVExperimentConfig,
    run_id: str,
    model_name: str,
    record: TilePermutationRecord,
    seed: int,
    ablation_name: str | None = None,
    metadata_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build common metadata for one training run."""

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
    metadata.update(dict(metadata_overrides or {}))
    return metadata


def build_training_run_spec(
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
    batch_augmentation: BatchAugmentation | None = None,
    curriculum_schedule: CurriculumSchedule | None = None,
    metadata_overrides: Optional[Mapping[str, Any]] = None,
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
    return TrainingRunSpec(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        config=build_training_config(config),
        checkpoint_config=build_checkpoint_config(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            ablation_name=ablation_name,
        ),
        metadata=build_training_metadata(
            config=config,
            run_id=run_id,
            model_name=model_name,
            record=record,
            seed=seed,
            ablation_name=ablation_name,
            metadata_overrides=metadata_overrides,
        ),
        progress_desc=progress_desc or f"{model_name} epochs",
        batch_augmentation=batch_augmentation,
        curriculum_schedule=curriculum_schedule,
    )


def build_pending_training_result_row(
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


def training_result_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the stable lookup key used to update a training result row."""

    tiles_per_side = 0 if pd.isna(row["tiles_per_side"]) else int(row["tiles_per_side"])
    ablation_name = row.get("ablation_name")
    if ablation_name is None or (isinstance(ablation_name, float) and pd.isna(ablation_name)):
        return tiles_per_side, int(row["tile_permutation_id"])
    return str(ablation_name), tiles_per_side, int(row["tile_permutation_id"])


def train_model_and_save_progress(
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
    """Train one model and persist result rows whenever progress is reported."""

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
