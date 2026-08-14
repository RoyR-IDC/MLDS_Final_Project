"""Optimizer builders for training runs."""

from __future__ import annotations

import torch
from torch import nn


def build_optimizer(model: nn.Module, name: str, learning_rate: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Build an optimizer from a short config name."""

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(trainable, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")
