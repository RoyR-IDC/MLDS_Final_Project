"""Shared PyTorch training and evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch import nn
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.metrics import AverageMeter


@dataclass
class TrainingRunComponents:
    """All state needed for a train-and-validation run."""

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    optimizer: torch.optim.Optimizer
    criterion: nn.Module
    device: TorchDevice
    epochs: int
    use_amp: bool = False
    checkpoint_path: Optional[str] = None
    progress_desc: str = "Epoch"
    progress_leave: bool = True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: TorchDevice,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train a model for one epoch.

    Args:
        model: Model to train.
        dataloader: Training dataloader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Training device.
        use_amp: Whether to use CUDA automatic mixed precision.

    Returns:
        Dictionary with ``train_loss`` and ``train_accuracy``.
    """

    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    amp_enabled = use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.numel()
        loss_meter.update(float(loss.item()), batch_size)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size
    return {"train_loss": loss_meter.average, "train_accuracy": correct / max(1, total)}


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: TorchDevice) -> Dict[str, float]:
    """Evaluate a model on a dataloader."""

    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            batch_size = targets.numel()
            loss_meter.update(float(loss.item()), batch_size)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size
    return {"val_loss": loss_meter.average, "val_accuracy": correct / max(1, total)}


def train_and_validate(components: TrainingRunComponents) -> Dict[str, float]:
    """Train and validate a model, returning final and best metrics."""

    components.model = components.model.to(components.device)
    components.criterion = components.criterion.to(components.device)
    best_val_accuracy = 0.0
    last_train = {"train_loss": 0.0, "train_accuracy": 0.0}
    last_val = {"val_loss": 0.0, "val_accuracy": 0.0}
    with tqdm(
        total=components.epochs,
        desc=components.progress_desc,
        unit="epoch",
        leave=components.progress_leave,
    ) as progress:
        for _ in range(components.epochs):
            last_train = train_one_epoch(
                components.model,
                components.train_loader,
                components.optimizer,
                components.criterion,
                components.device,
                use_amp=components.use_amp,
            )
            last_val = evaluate(
                components.model,
                components.val_loader,
                components.criterion,
                components.device,
            )
            if last_val["val_accuracy"] > best_val_accuracy:
                best_val_accuracy = last_val["val_accuracy"]
                if components.checkpoint_path:
                    torch.save(
                        {"model_state": components.model.state_dict(), "val_accuracy": best_val_accuracy},
                        components.checkpoint_path,
                    )
            progress.set_postfix(
                train_loss=last_train["train_loss"],
                train_accuracy=last_train["train_accuracy"],
                val_loss=last_val["val_loss"],
                val_accuracy=last_val["val_accuracy"],
                best_val_accuracy=best_val_accuracy,
            )
            progress.update(1)
    return {**last_train, **last_val, "best_val_accuracy": best_val_accuracy}
