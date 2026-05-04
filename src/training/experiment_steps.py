"""Generic training steps shared by notebook-owned experiments."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.factory import get_model
from src.training.engine import build_optimizer, fit


def build_training_components(config: Dict, model_name: str, device: torch.device, overrides: Optional[Dict] = None) -> Dict[str, object]:
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
    pretrained = bool(run_options.get("pretrained", config.get("pretrained", False)))
    if model_name == "convmixer":
        pretrained = False

    model = get_model(
        model_name,
        num_classes=int(config.get("num_classes", 2)),
        pretrained=pretrained,
        device=device,
        freeze_backbone=bool(run_options.get("freeze_backbone", config.get("freeze_backbone", False))),
        convmixer_dim=int(config.get("convmixer_dim", 256)),
        convmixer_depth=int(config.get("convmixer_depth", 8)),
    )
    optimizer = build_optimizer(
        model,
        name=config.get("optimizer", "adamw"),
        learning_rate=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    components = {"model": model, "optimizer": optimizer, "criterion": criterion}
    return components


def train_model_configuration(
    config: Dict,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    overrides: Optional[Dict] = None,
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

    components = build_training_components(config, model_name, device, overrides=overrides)
    metrics = fit(
        components["model"],
        train_loader,
        val_loader,
        epochs=int(config.get("epochs", 1)),
        optimizer=components["optimizer"],
        criterion=components["criterion"],
        device=device,
        use_amp=bool(config.get("use_amp", False)),
    )
    return metrics
