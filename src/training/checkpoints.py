"""Checkpoint helpers for PyTorch training runs."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch._C import device as TorchDevice

from src.training.run import EpochResult, TrainingResult


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Any | None = None,
    epoch: int,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    planned_total_epochs: int | None = None,
) -> None:
    """Save a complete training checkpoint."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics),
        "metadata": dict(metadata),
    }
    if planned_total_epochs is not None:
        state["planned_total_epochs"] = int(planned_total_epochs)
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str | TorchDevice | None = None) -> dict[str, Any]:
    """Load a PyTorch checkpoint."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint_if_available(
    path: str | None,
    map_location: str | TorchDevice | None = None,
) -> dict[str, Any] | None:
    """Load a checkpoint when the path exists."""

    if not path or not os.path.exists(path):
        return None
    return load_checkpoint(path, map_location=map_location)


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    expected_metadata: Mapping[str, Any],
    planned_total_epochs: int,
) -> bool:
    """Return whether a checkpoint belongs to the current experiment step."""

    checkpoint_metadata = checkpoint.get("metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        return False

    keys_to_match = (
        "part",
        "config_name",
        "run_id",
        "model_name",
        "ablation_name",
        "tiles_per_side",
        "tile_permutation_id",
        "tile_permutation_seed",
        "seed",
        "augmentation_name",
        "batch_augmentation_name",
        "curriculum_name",
        "curriculum_stages",
        "optimizer_name",
        "learning_rate",
        "weight_decay",
        "use_amp",
        "pretrained",
        "freeze_backbone",
        "epochs",
        "classification_head",
        "p_original",
        "tile_permutation_probability",
    )
    for key in keys_to_match:
        if key == "run_id":
            compatible_run_ids = set(expected_metadata.get("_compatible_run_ids", ()))
            if (
                expected_metadata.get(key) != checkpoint_metadata.get(key)
                and checkpoint_metadata.get(key) in compatible_run_ids
            ):
                continue
        if expected_metadata.get(key) != checkpoint_metadata.get(key):
            return False

    checkpoint_total_epochs = checkpoint.get("planned_total_epochs")
    return checkpoint_total_epochs is None or int(checkpoint_total_epochs) == int(planned_total_epochs)


def checkpoint_to_completed_result(
    *,
    checkpoint: Mapping[str, Any],
    model_name: str,
    metadata: Mapping[str, Any],
    best_checkpoint_path: str | None,
    last_checkpoint_path: str | None,
    skipped_from_checkpoint: str,
    duration_seconds: float,
) -> TrainingResult:
    """Build a completed training result from a complete checkpoint."""

    metrics = dict(checkpoint.get("metrics") or {})
    epoch = int(checkpoint.get("epoch", 0))
    best_val_accuracy = metrics.get("best_val_accuracy", metrics.get("val_accuracy"))
    result = TrainingResult.pending(model_name=model_name, metadata=metadata)
    if epoch > 0:
        result.update_from_epoch(
            EpochResult(
                epoch=epoch,
                train_loss=float(metrics.get("train_loss", 0.0)),
                train_accuracy=float(metrics.get("train_accuracy", 0.0)),
                val_loss=float(metrics.get("val_loss", 0.0)),
                val_accuracy=float(metrics.get("val_accuracy", 0.0)),
                best_val_accuracy=float(best_val_accuracy or 0.0),
            )
        )
    result.best_checkpoint_path = best_checkpoint_path
    result.last_checkpoint_path = last_checkpoint_path
    result.skipped_from_checkpoint = skipped_from_checkpoint
    result.mark_completed(duration_seconds)
    return result
