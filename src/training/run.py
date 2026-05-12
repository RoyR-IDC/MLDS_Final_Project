"""Structured training run inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


RunStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class CheckpointConfig:
    """Checkpoint destinations for one training run."""

    best_path: Optional[str] = None
    last_path: Optional[str] = None
    save_best: bool = True
    save_last: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters used by the shared trainer."""

    epochs: int
    optimizer_name: str
    learning_rate: float
    weight_decay: float = 0.0
    use_amp: bool = False


@dataclass
class EpochResult:
    """Metrics produced by one training epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    best_val_accuracy: float


@dataclass
class TrainingRunSpec:
    """All objects and metadata required to train one model."""

    model_name: str
    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    device: torch.device
    config: TrainingConfig
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    metadata: dict[str, Any] = field(default_factory=dict)
    progress_desc: str = "Epoch"
    progress_leave: bool = True


@dataclass
class TrainingResult:
    """Structured output returned by every shared training run."""

    status: RunStatus
    model_name: str
    metadata: dict[str, Any]
    epoch_history: list[EpochResult] = field(default_factory=list)
    train_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    best_checkpoint_path: Optional[str] = None
    last_checkpoint_path: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

    @classmethod
    def pending(cls, *, model_name: str, metadata: Mapping[str, Any]) -> "TrainingResult":
        """Build a pending result placeholder."""

        return cls(status="pending", model_name=model_name, metadata=dict(metadata))

    def latest_metrics(self) -> dict[str, Any]:
        """Return flat metric fields suitable for CSV result rows."""

        return {
            "run_status": self.status,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "best_checkpoint_path": self.best_checkpoint_path,
            "last_checkpoint_path": self.last_checkpoint_path,
            "training_duration_seconds": self.training_duration_seconds,
            "error_message": self.error_message,
        }

    def mark_running(self) -> None:
        """Mark this result as running."""

        self.status = "running"

    def update_from_epoch(self, epoch_result: EpochResult) -> None:
        """Refresh final metric fields from the latest epoch."""

        self.epoch_history.append(epoch_result)
        self.train_loss = epoch_result.train_loss
        self.train_accuracy = epoch_result.train_accuracy
        self.val_loss = epoch_result.val_loss
        self.val_accuracy = epoch_result.val_accuracy
        self.best_val_accuracy = epoch_result.best_val_accuracy

    def mark_completed(self, duration_seconds: float) -> None:
        """Mark this result as completed."""

        self.status = "completed"
        self.training_duration_seconds = duration_seconds

    def mark_failed(self, duration_seconds: float, error: BaseException) -> None:
        """Mark this result as failed while preserving partial metrics."""

        self.status = "failed"
        self.training_duration_seconds = duration_seconds
        self.error_message = str(error)
