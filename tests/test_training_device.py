from typing import cast

import torch
from torch import nn
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader

from src.training import engine
from src.training.engine import TrainingRunComponents
from src.training.run import TrainingConfig, TrainingRunSpec
from src.training.trainer import ModelTrainer


class RecordingLinear(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class RecordingCriterion(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def test_model_trainer_moves_model_and_criterion_to_spec_device():
    device = TorchDevice("cpu")
    model = RecordingLinear()
    criterion = RecordingCriterion()
    spec = TrainingRunSpec(
        model_name="recording_model",
        model=model,
        train_loader=cast(DataLoader, []),
        val_loader=cast(DataLoader, []),
        criterion=criterion,
        device=device,
        config=TrainingConfig(epochs=1, optimizer_name="sgd", learning_rate=0.01),
    )

    ModelTrainer(spec)

    assert model.to_calls[0][0] == (device,)
    assert criterion.to_calls[0][0] == (device,)


def test_train_and_validate_moves_model_and_criterion_to_component_device(monkeypatch):
    device = TorchDevice("cpu")
    model = RecordingLinear()
    criterion = RecordingCriterion()
    components = TrainingRunComponents(
        model=model,
        train_loader=cast(DataLoader, []),
        val_loader=cast(DataLoader, []),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        criterion=criterion,
        device=device,
        epochs=1,
        progress_leave=False,
    )

    def fake_train_one_epoch(*args, **kwargs):
        return {"train_loss": 0.2, "train_accuracy": 0.9}

    def fake_evaluate(*args, **kwargs):
        return {"val_loss": 0.3, "val_accuracy": 0.8}

    monkeypatch.setattr(engine, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(engine, "evaluate", fake_evaluate)

    metrics = engine.train_and_validate(components)

    assert model.to_calls[0][0] == (device,)
    assert criterion.to_calls[0][0] == (device,)
    assert metrics["best_val_accuracy"] == 0.8
