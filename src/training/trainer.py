"""Object-oriented PyTorch model training."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.checkpoints import save_checkpoint
from src.training.metrics import AverageMeter
from src.training.optimizers import build_optimizer
from src.training.run import EpochResult, TrainingResult, TrainingRunSpec


ProgressCallback = Callable[[TrainingResult], None]


class ModelTrainer:
    """Train and evaluate one model using a shared run specification."""

    def __init__(self, spec: TrainingRunSpec) -> None:
        self.spec = spec
        self.optimizer = build_optimizer(
            spec.model,
            name=spec.config.optimizer_name,
            learning_rate=spec.config.learning_rate,
            weight_decay=spec.config.weight_decay,
        )
        amp_enabled = spec.config.use_amp and spec.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled

    def fit(self, on_progress: Optional[ProgressCallback] = None) -> TrainingResult:
        """Train the configured model and return a structured result."""

        result = TrainingResult.pending(model_name=self.spec.model_name, metadata=self.spec.metadata)
        result.mark_running()
        self._notify(on_progress, result)
        best_val_accuracy = 0.0
        start_time = perf_counter()

        try:
            with tqdm(
                total=self.spec.config.epochs,
                desc=self.spec.progress_desc,
                unit="epoch",
                leave=self.spec.progress_leave,
            ) as progress:
                for epoch in range(1, self.spec.config.epochs + 1):
                    train_metrics = self.train_one_epoch()
                    val_metrics = self.evaluate()
                    best_val_accuracy = max(best_val_accuracy, val_metrics["val_accuracy"])
                    epoch_result = EpochResult(
                        epoch=epoch,
                        train_loss=train_metrics["train_loss"],
                        train_accuracy=train_metrics["train_accuracy"],
                        val_loss=val_metrics["val_loss"],
                        val_accuracy=val_metrics["val_accuracy"],
                        best_val_accuracy=best_val_accuracy,
                    )
                    result.update_from_epoch(epoch_result)
                    self._save_epoch_checkpoints(result, epoch_result)
                    self._notify(on_progress, result)
                    progress.set_postfix(
                        train_loss=epoch_result.train_loss,
                        train_accuracy=epoch_result.train_accuracy,
                        val_loss=epoch_result.val_loss,
                        val_accuracy=epoch_result.val_accuracy,
                        best_val_accuracy=epoch_result.best_val_accuracy,
                    )
                    progress.update(1)
        except Exception as exc:
            result.mark_failed(perf_counter() - start_time, exc)
            self._notify(on_progress, result)
            raise

        result.mark_completed(perf_counter() - start_time)
        self._notify(on_progress, result)
        return result

    def train_one_epoch(self) -> dict[str, float]:
        """Train the model for one epoch."""

        self.spec.model.train()
        return self._run_batches(self.spec.train_loader, training=True)

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> dict[str, float]:
        """Evaluate the model on a dataloader."""

        self.spec.model.eval()
        with torch.no_grad():
            return self._run_batches(dataloader or self.spec.val_loader, training=False)

    def _run_batches(self, dataloader: DataLoader, *, training: bool) -> dict[str, float]:
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        for images, targets in dataloader:
            images = images.to(self.spec.device)
            targets = targets.to(self.spec.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.amp_enabled and training):
                logits = self.spec.model(images)
                loss = self.spec.criterion(logits, targets)

            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            batch_size = targets.numel()
            loss_meter.update(float(loss.item()), batch_size)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size

        prefix = "train" if training else "val"
        return {f"{prefix}_loss": loss_meter.average, f"{prefix}_accuracy": correct / max(1, total)}

    def _save_epoch_checkpoints(self, result: TrainingResult, epoch_result: EpochResult) -> None:
        checkpoint_config = self.spec.checkpoint_config
        metrics: dict[str, Any] = {
            "train_loss": epoch_result.train_loss,
            "train_accuracy": epoch_result.train_accuracy,
            "val_loss": epoch_result.val_loss,
            "val_accuracy": epoch_result.val_accuracy,
            "best_val_accuracy": epoch_result.best_val_accuracy,
        }
        is_best = epoch_result.val_accuracy >= epoch_result.best_val_accuracy
        if checkpoint_config.save_best and checkpoint_config.best_path and is_best:
            save_checkpoint(
                path=checkpoint_config.best_path,
                model=self.spec.model,
                optimizer=self.optimizer,
                epoch=epoch_result.epoch,
                metrics=metrics,
                metadata=self.spec.metadata,
            )
            result.best_checkpoint_path = checkpoint_config.best_path
        if checkpoint_config.save_last and checkpoint_config.last_path:
            save_checkpoint(
                path=checkpoint_config.last_path,
                model=self.spec.model,
                optimizer=self.optimizer,
                epoch=epoch_result.epoch,
                metrics=metrics,
                metadata=self.spec.metadata,
            )
            result.last_checkpoint_path = checkpoint_config.last_path

    @staticmethod
    def _notify(callback: Optional[ProgressCallback], result: TrainingResult) -> None:
        if callback is not None:
            callback(result)
