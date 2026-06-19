"""Object-oriented PyTorch model training."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.checkpoints import (
    checkpoint_to_completed_result,
    load_checkpoint_if_available,
    save_checkpoint,
    validate_checkpoint_metadata,
)
from src.training.curriculum import TrainingStage
from src.training.metrics import AverageMeter
from src.training.optimizers import build_optimizer
from src.training.profiling import PhaseProfiler, timed_seconds
from src.training.run import EpochResult, TrainingResult, TrainingRunSpec


ProgressCallback = Callable[[TrainingResult], None]


class ModelTrainer:
    """Train and evaluate one model using a shared run specification."""

    def __init__(self, spec: TrainingRunSpec) -> None:
        self.spec = spec
        self.spec.model = self.spec.model.to(self.spec.device)
        self.spec.criterion = self.spec.criterion.to(self.spec.device)
        self.optimizer = build_optimizer(
            spec.model,
            name=spec.config.optimizer_name,
            learning_rate=spec.config.learning_rate,
            weight_decay=spec.config.weight_decay,
        )
        amp_enabled = spec.config.use_amp and spec.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled
        self._profile_rows: list[dict[str, Any]] = []
        self._validated_input_batch = False
        self._log_runtime_summary()

    def fit(self, on_progress: Optional[ProgressCallback] = None) -> TrainingResult:
        """Train the configured model and return a structured result."""

        result = TrainingResult.pending(model_name=self.spec.model_name, metadata=self.spec.metadata)
        start_time = perf_counter()

        try:
            stages = self._training_stages()
            planned_total_epochs = sum(stage.epochs for stage in stages)
            resume_checkpoint = self._load_resume_checkpoint(planned_total_epochs)
            start_epoch = int(resume_checkpoint.get("epoch", 0)) if resume_checkpoint else 0
            best_val_accuracy = self._checkpoint_best_val_accuracy(resume_checkpoint)

            if resume_checkpoint and start_epoch >= planned_total_epochs:
                result = checkpoint_to_completed_result(
                    checkpoint=resume_checkpoint,
                    model_name=self.spec.model_name,
                    metadata=self.spec.metadata,
                    best_checkpoint_path=self.spec.checkpoint_config.best_path,
                    last_checkpoint_path=self.spec.checkpoint_config.last_path,
                    skipped_from_checkpoint=self.spec.checkpoint_config.resume_path or "",
                    duration_seconds=perf_counter() - start_time,
                )
                print(
                    f"Checkpoint complete; skipping step at epoch "
                    f"{start_epoch}/{planned_total_epochs}: {self.spec.checkpoint_config.resume_path}"
                )
                self._notify(on_progress, result)
                return result

            if resume_checkpoint:
                self._restore_training_state(resume_checkpoint)
                result.resumed_from_checkpoint = self.spec.checkpoint_config.resume_path
                result.resumed_from_epoch = start_epoch
                result.best_checkpoint_path = self.spec.checkpoint_config.best_path
                result.last_checkpoint_path = self.spec.checkpoint_config.last_path
                self._apply_checkpoint_metrics(result, resume_checkpoint)
                print(
                    f"Resuming from epoch {start_epoch}/{planned_total_epochs}: "
                    f"{self.spec.checkpoint_config.resume_path}"
                )

            result.mark_running()
            self._notify(on_progress, result)
            self._validate_first_training_batch(stages, start_epoch=start_epoch)
            with tqdm(
                total=planned_total_epochs,
                initial=start_epoch,
                desc=self.spec.progress_desc,
                unit="epoch",
                position=0,
                leave=self.spec.progress_leave,
            ) as progress:
                epoch = 0
                for stage in stages:
                    for _ in range(stage.epochs):
                        epoch += 1
                        if epoch <= start_epoch:
                            continue
                        with tqdm(
                            total=len(stage.train_loader),
                            desc=f"{self.spec.progress_desc} epoch {epoch} batches",
                            unit="batch",
                            position=1,
                            leave=False,
                        ) as batch_progress:
                            train_metrics = self.train_one_epoch(
                                stage.train_loader,
                                batch_progress=batch_progress,
                                epoch=epoch,
                                stage_name=stage.name,
                            )
                        val_metrics = self.evaluate(epoch=epoch, stage_name=stage.name)
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
                        result.profile_rows = list(self._profile_rows)
                        self._save_epoch_checkpoints(result, epoch_result, planned_total_epochs)
                        self._notify(on_progress, result)
                        progress.set_postfix(
                            stage=stage.name,
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
        result.profile_rows = list(self._profile_rows)
        self._notify(on_progress, result)
        return result

    def train_one_epoch(
        self,
        dataloader: Optional[DataLoader] = None,
        *,
        batch_progress: Any | None = None,
        epoch: int | None = None,
        stage_name: str | None = None,
    ) -> dict[str, float]:
        """Train the model for one epoch."""

        self.spec.model.train()
        return self._run_batches(
            dataloader or self.spec.train_loader,
            training=True,
            batch_progress=batch_progress,
            epoch=epoch,
            stage_name=stage_name,
        )

    def evaluate(
        self,
        dataloader: Optional[DataLoader] = None,
        *,
        epoch: int | None = None,
        stage_name: str | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on a dataloader."""

        self.spec.model.eval()
        with torch.inference_mode():
            return self._run_batches(
                dataloader or self.spec.val_loader,
                training=False,
                epoch=epoch,
                stage_name=stage_name,
            )

    def _run_batches(
        self,
        dataloader: DataLoader,
        *,
        training: bool,
        batch_progress: Any | None = None,
        epoch: int | None = None,
        stage_name: str | None = None,
    ) -> dict[str, float]:
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        phase = "train" if training else "val"
        profiler = PhaseProfiler(
            enabled=self.spec.config.profile_performance,
            device=self.spec.device,
            phase=phase,
            epoch=epoch,
            warmup_batches=max(0, int(self.spec.config.profile_warmup_batches)),
        )
        profiler.start()

        iterator = iter(dataloader)
        while True:
            with timed_seconds() as data_timer:
                try:
                    images, targets = next(iterator)
                except StopIteration:
                    break
            profiler.count_batch()
            profiler.add_data_loading(data_timer.elapsed_seconds)

            with timed_seconds(self.spec.device) as transfer_timer:
                images = images.to(self.spec.device, non_blocking=True)
                targets = targets.to(self.spec.device, non_blocking=True)
            profiler.add_transfer(transfer_timer.elapsed_seconds)
            self._validate_runtime_batch(images)

            if training:
                if self.spec.batch_augmentation is not None:
                    images, targets = self.spec.batch_augmentation(images, targets)
                self.optimizer.zero_grad(set_to_none=True)

            with timed_seconds(self.spec.device) as compute_timer:
                with autocast("cuda", enabled=self.amp_enabled and training):
                    logits = self.spec.model(images)
                    loss = self.spec.criterion(logits, targets)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            profiler.add_compute(compute_timer.elapsed_seconds)

            batch_size = targets.numel()
            loss_meter.update(float(loss.item()), batch_size)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size
            if training and batch_progress is not None:
                batch_progress.update(1)

        profiler.finish()
        self._append_profile_row(profiler, stage_name=stage_name)
        return {f"{phase}_loss": loss_meter.average, f"{phase}_accuracy": correct / max(1, total)}

    def _log_runtime_summary(self) -> None:
        model_device = self._model_device()
        cuda_name = (
            torch.cuda.get_device_name(self.spec.device)
            if self.spec.device.type == "cuda" and torch.cuda.is_available()
            else None
        )
        total_params = sum(parameter.numel() for parameter in self.spec.model.parameters())
        trainable_params = sum(
            parameter.numel()
            for parameter in self.spec.model.parameters()
            if parameter.requires_grad
        )
        optimizer_params = sum(
            parameter.numel()
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )
        print(
            f"Runtime: selected_device={self.spec.device}, model_device={model_device}, "
            f"use_amp={self.spec.config.use_amp}, amp_enabled={self.amp_enabled}, cuda_device={cuda_name}"
        )
        print(
            f"Model summary: model={self.spec.model_name}, total_params={total_params:,}, "
            f"trainable_params={trainable_params:,}, optimizer_params={optimizer_params:,}, "
            f"freeze_backbone={bool(self.spec.metadata.get('freeze_backbone', False))}"
        )
        if self.spec.device.type == "cuda" and model_device.type != "cuda":
            raise RuntimeError(f"CUDA was selected, but model is on {model_device}.")

    def _validate_first_training_batch(self, stages: list[TrainingStage], *, start_epoch: int) -> None:
        if self._validated_input_batch:
            return

        epoch = 0
        for stage in stages:
            for _ in range(stage.epochs):
                epoch += 1
                if epoch <= start_epoch:
                    continue
                iterator = iter(stage.train_loader)
                try:
                    images, _ = next(iterator)
                except StopIteration:
                    return
                images = images.to(self.spec.device, non_blocking=True)
                self._validate_runtime_batch(images)
                self._validate_forward_logits(images)
                return

    def _model_device(self) -> torch.device:
        try:
            return next(self.spec.model.parameters()).device
        except StopIteration:
            return self.spec.device

    def _validate_runtime_batch(self, images: torch.Tensor) -> None:
        if self._validated_input_batch:
            return
        self._validated_input_batch = True
        print(f"First batch: tensor_device={images.device}, tensor_shape={tuple(images.shape)}")
        if self.spec.device.type == "cuda" and images.device.type != "cuda":
            raise RuntimeError(f"CUDA was selected, but input batch is on {images.device}.")
        if self.spec.expected_input_size is None and images.ndim != 4:
            return
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected image batch shape [N, 3, H, W], got {tuple(images.shape)}.")
        if self.spec.expected_input_size is not None:
            expected_shape = (3, int(self.spec.expected_input_size), int(self.spec.expected_input_size))
            actual_shape = tuple(int(value) for value in images.shape[1:])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Expected image tensors with shape [N, {expected_shape[0]}, {expected_shape[1]}, "
                    f"{expected_shape[2]}], got {tuple(images.shape)}."
                )

    def _append_profile_row(self, profiler: PhaseProfiler, *, stage_name: str | None) -> None:
        row = profiler.summary_row(
            {
                "model_name": self.spec.model_name,
                "stage": stage_name,
                **self.spec.metadata,
            }
        )
        if row is not None:
            self._profile_rows.append(row)

    def _save_epoch_checkpoints(
        self,
        result: TrainingResult,
        epoch_result: EpochResult,
        planned_total_epochs: int,
    ) -> None:
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
                scaler=self.scaler,
                epoch=epoch_result.epoch,
                metrics=metrics,
                metadata=self.spec.metadata,
                planned_total_epochs=planned_total_epochs,
            )
            result.best_checkpoint_path = checkpoint_config.best_path
        if checkpoint_config.save_last and checkpoint_config.last_path:
            save_checkpoint(
                path=checkpoint_config.last_path,
                model=self.spec.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch_result.epoch,
                metrics=metrics,
                metadata=self.spec.metadata,
                planned_total_epochs=planned_total_epochs,
            )
            result.last_checkpoint_path = checkpoint_config.last_path

    @staticmethod
    def _notify(callback: Optional[ProgressCallback], result: TrainingResult) -> None:
        if callback is not None:
            callback(result)

    def _training_stages(self) -> list[TrainingStage]:
        if self.spec.curriculum_schedule is not None:
            return list(self.spec.curriculum_schedule.stages)
        return [TrainingStage(name="standard", epochs=self.spec.config.epochs, train_loader=self.spec.train_loader)]

    def _validate_forward_logits(self, images: torch.Tensor) -> None:
        expected_num_classes = self.spec.expected_num_classes
        if expected_num_classes is None:
            return
        was_training = self.spec.model.training
        self.spec.model.eval()
        with torch.inference_mode():
            logits = self.spec.model(images)
        if was_training:
            self.spec.model.train()
        expected_shape = (int(images.shape[0]), int(expected_num_classes))
        actual_shape = tuple(int(value) for value in logits.shape)
        print(f"Sanity forward: logits_shape={actual_shape}, expected_shape={expected_shape}")
        if actual_shape != expected_shape:
            raise ValueError(
                f"Expected model logits shape {expected_shape}, got {actual_shape} "
                f"for model '{self.spec.model_name}'."
            )

    def _load_resume_checkpoint(self, planned_total_epochs: int) -> dict[str, Any] | None:
        checkpoint_config = self.spec.checkpoint_config
        if not checkpoint_config.resume or not checkpoint_config.resume_path:
            return None
        checkpoint = load_checkpoint_if_available(checkpoint_config.resume_path, map_location=self.spec.device)
        if checkpoint is None:
            return None
        if validate_checkpoint_metadata(
            checkpoint,
            expected_metadata=self.spec.metadata,
            planned_total_epochs=planned_total_epochs,
        ):
            return checkpoint
        print(f"Ignoring checkpoint with mismatched metadata: {checkpoint_config.resume_path}")
        return None

    def _restore_training_state(self, checkpoint: dict[str, Any]) -> None:
        self.spec.model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        scaler_state = checkpoint.get("scaler_state")
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)

    @staticmethod
    def _checkpoint_best_val_accuracy(checkpoint: dict[str, Any] | None) -> float:
        if checkpoint is None:
            return 0.0
        metrics = checkpoint.get("metrics") or {}
        return float(metrics.get("best_val_accuracy", metrics.get("val_accuracy", 0.0)) or 0.0)

    @staticmethod
    def _apply_checkpoint_metrics(result: TrainingResult, checkpoint: dict[str, Any]) -> None:
        metrics = checkpoint.get("metrics") or {}
        epoch = int(checkpoint.get("epoch", 0))
        if epoch <= 0:
            return
        result.update_from_epoch(
            EpochResult(
                epoch=epoch,
                train_loss=float(metrics.get("train_loss", 0.0)),
                train_accuracy=float(metrics.get("train_accuracy", 0.0)),
                val_loss=float(metrics.get("val_loss", 0.0)),
                val_accuracy=float(metrics.get("val_accuracy", 0.0)),
                best_val_accuracy=float(
                    metrics.get("best_val_accuracy", metrics.get("val_accuracy", 0.0)) or 0.0
                ),
            )
        )
