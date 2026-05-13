"""Curriculum-learning schedule objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingStage:
    """One stage in a curriculum schedule."""

    name: str
    epochs: int
    train_loader: DataLoader
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("stage epochs must be at least 1")


@dataclass(frozen=True)
class CurriculumSchedule:
    """Ordered stages that train one model continuously."""

    stages: Sequence[TrainingStage]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("curriculum schedule must include at least one stage")

    @property
    def total_epochs(self) -> int:
        """Return the total number of epochs across all stages."""

        return sum(stage.epochs for stage in self.stages)

    @property
    def stage_names(self) -> list[str]:
        """Return stage names in training order."""

        return [stage.name for stage in self.stages]
