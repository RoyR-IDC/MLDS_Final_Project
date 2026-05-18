"""Loss functions used by the training pipeline."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with scalar alpha weighting."""

    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if reduction != "mean":
            raise ValueError("FocalLoss currently supports only reduction='mean'")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        losses = -self.alpha * (1.0 - target_probabilities).pow(self.gamma) * target_log_probabilities
        return losses.mean()
