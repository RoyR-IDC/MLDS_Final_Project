"""Training metric helpers."""

from __future__ import annotations

from typing import Dict

import torch


def classification_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 classification accuracy for a batch."""

    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


class AverageMeter:
    """Track a running weighted average.

    Attributes:
        total: Weighted sum of observed values.
        count: Number of examples included.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int) -> None:
        """Add a weighted observation."""

        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        """Return the running average."""

        return self.total / max(1, self.count)


def make_result_row(**metadata) -> Dict[str, object]:
    """Return metadata as a plain result row dictionary."""

    return dict(metadata)

