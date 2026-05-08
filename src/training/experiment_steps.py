"""Generic training steps shared by notebook-owned experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.experiment_results import build_result_row
from src.models.factory import get_model
from src.preprocessing.dogs_cats import build_dataloaders
from src.preprocessing.permutations import PermutationRecord
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
    pretrained = bool(run_options.get("pretrained", config.pretrained))
    if model_name == "convmixer":
        pretrained = False

    model = get_model(
        model_name,
        num_classes=config.num_classes,
        pretrained=pretrained,
        device=device,
        freeze_backbone=bool(run_options.get("freeze_backbone", getattr(config, "freeze_backbone", False))),
        convmixer_dim=config.convmixer_dim,
        convmixer_depth=config.convmixer_depth,
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
    )
    metrics = train_and_validate(training_run)
    print(f"Finished training model '{model_name}'. Metrics: {metrics}")
    return metrics


def collect_model_permutation_results(
    *,
    config: CVExperimentConfig,
    model_name: str,
    run_id: str,
    train_samples: Sequence[tuple[str, int]],
    validation_samples: Sequence[tuple[str, int]],
    permutation_records: Sequence[PermutationRecord],
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Train one model across permutation records and collect result rows."""

    rows: list[dict[str, Any]] = []
    executable_records = [
        record
        for record in permutation_records
        if not (record.grid_size == 1 and record.permutation_id > 0)
    ]
    for record_index, record in enumerate(executable_records, start=1):
        print()
        print("=" * 80)
        print(
            f"[{record_index}/{len(executable_records)}] Running permutation "
            f"grid={record.grid_size}x{record.grid_size}, "
            f"permutation_id={record.permutation_id}, seed={record.permutation_seed}"
        )
        print("Building dataloaders...")
        train_loader, validation_loader = build_dataloaders(
            train_samples=train_samples,
            val_samples=validation_samples,
            image_size=config.image_size,
            grid_size=record.grid_size,
            permutation=record.permutation,
            seed=seed,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            standard_augmentation=False,
        )
        print(
            "Finished building dataloaders: "
            f"{len(train_loader)} train batches, {len(validation_loader)} validation batches."
        )
        print(f"Training model '{model_name}' on the current permutation...")
        metrics = train_and_evaluate_model_configuration(
            config=config,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            overrides={"pretrained": config.pretrained and model_name != "convmixer"},
        )
        print(f"Done training model '{model_name}' on permutation_id={record.permutation_id}.")
        rows.append(
            build_result_row(
                config=config,
                run_id=run_id,
                model_name=model_name,
                record=record,
                seed=seed,
                metrics=metrics,
            )
        )
    return rows
