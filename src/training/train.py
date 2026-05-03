"""Compatibility wrappers for the shared training engine."""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.training.engine import evaluate, fit, train_one_epoch as _engine_train_one_epoch


def train_one_epoch(model, loader, optimizer, device, criterion, epoch=0, mixup_alpha: float = 0.0) -> Dict[str, float]:
    """Train for one epoch and return legacy metric keys.

    Args:
        model: Model to train.
        loader: Training dataloader.
        optimizer: Optimizer.
        device: Training device.
        criterion: Loss function.
        epoch: Unused legacy epoch index.
        mixup_alpha: Unsupported legacy mixup setting, ignored when nonzero.

    Returns:
        Dictionary with ``loss`` and ``acc`` keys.
    """

    del epoch, mixup_alpha
    metrics = _engine_train_one_epoch(model, loader, optimizer, criterion, device)
    return {"loss": metrics["train_loss"], "acc": metrics["train_accuracy"]}


def validate(model, loader, device, criterion) -> Dict[str, float]:
    """Evaluate a model and return legacy metric keys."""

    metrics = evaluate(model, loader, criterion, device)
    return {"loss": metrics["val_loss"], "acc": metrics["val_accuracy"]}


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save a PyTorch checkpoint."""

    torch.save(state, path)

__all__ = ["train_one_epoch", "evaluate", "fit", "validate", "save_checkpoint"]
