"""Checkpoint helpers for PyTorch training runs."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import torch
from torch import nn


def save_checkpoint(
    *,
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Save a complete training checkpoint."""

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics),
        "metadata": dict(metadata),
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Load a PyTorch checkpoint."""

    return torch.load(path, map_location=map_location)
