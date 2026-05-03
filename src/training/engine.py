"""Shared PyTorch training and evaluation engine."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import AverageMeter


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
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


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Evaluate a model on a dataloader."""

    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            batch_size = targets.numel()
            loss_meter.update(float(loss.item()), batch_size)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size
    return {"val_loss": loss_meter.average, "val_accuracy": correct / max(1, total)}


def build_optimizer(model: nn.Module, name: str, learning_rate: float, weight_decay: float = 0.0):
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


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """Train and validate a model, returning final and best metrics."""

    best_val_accuracy = 0.0
    last_train = {"train_loss": 0.0, "train_accuracy": 0.0}
    last_val = {"val_loss": 0.0, "val_accuracy": 0.0}
    for _ in range(epochs):
        last_train = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp=use_amp)
        last_val = evaluate(model, val_loader, criterion, device)
        if last_val["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = last_val["val_accuracy"]
            if checkpoint_path:
                torch.save({"model_state": model.state_dict(), "val_accuracy": best_val_accuracy}, checkpoint_path)
    return {**last_train, **last_val, "best_val_accuracy": best_val_accuracy}

