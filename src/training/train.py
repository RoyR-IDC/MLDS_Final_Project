from typing import Tuple, Dict, Any, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import random
import numpy as np


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Return mixed inputs, pairs of targets, and lambda for mixup augmentation."""
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    mixup_alpha: float = 0.0,
) -> Dict[str, float]:
    """Train model for one epoch and return loss/accuracy summary.

    Supports optional mixup augmentation controlled by `mixup_alpha`.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        if mixup_alpha > 0.0:
            xb_m, y_a, y_b, lam = mixup_data(xb, yb, mixup_alpha)
            out = model(xb_m)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            # For accuracy reporting, use predictions on non-mixed targets by majority (approx)
            preds = out.argmax(dim=1)
            # Count matches to blended targets conservatively against y_a
            correct += (preds == y_a).sum().item()
        else:
            out = model(xb)
            loss = criterion(out, yb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)
    elapsed = time.time() - start
    return {
        'loss': running_loss / max(1, total),
        'acc': correct / max(1, total),
        'time': elapsed,
    }


def validate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> Dict[str, float]:
    """Evaluate model on loader and return loss/accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            running_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return {'loss': running_loss / max(1, total), 'acc': correct / max(1, total)}


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Save training state dict to path."""
    torch.save(state, path)
