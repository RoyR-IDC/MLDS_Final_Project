"""Small training utilities."""

from __future__ import annotations

from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save a PyTorch checkpoint."""

    torch.save(state, path)

__all__ = ["save_checkpoint"]
