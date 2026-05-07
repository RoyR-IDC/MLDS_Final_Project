"""Generic training steps shared by notebook-owned experiments."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.factory import get_model
from src.training.engine import build_optimizer, fit
from src.utils.config import CVExperimentConfig


def build_training_components(
    config: CVExperimentConfig,
    model_name: str,
    device: torch.device,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, object]:
    """Build model, optimizer, and criterion objects for one training run.

    Args:
        config: Normalized experiment configuration.
        model_name: Name of the model architecture to train.
        device: Device where the model should run.
        overrides: Optional per-run settings such as pretrained or freeze flags.

    Returns:
        Dictionary containing ``model``, ``optimizer``, and ``criterion``.
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
    components = {"model": model, "optimizer": optimizer, "criterion": criterion}
    return components


def train_model_configuration(
    config: CVExperimentConfig,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Train one model configuration and return final/best metrics.

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
    metrics = fit(
        components["model"],
        train_loader,
        val_loader,
        epochs=config.epochs,
        optimizer=components["optimizer"],
        criterion=components["criterion"],
        device=device,
        use_amp=config.use_amp,
    )
    print(f"Finished training model '{model_name}'. Metrics: {metrics}")
    return metrics
