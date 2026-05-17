from typing import cast

import torch
from torch import nn
from torch._C import device as TorchDevice
from torch.utils.data import DataLoader

from src.training import engine
import src.training.trainer as trainer_module
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


class NoOpToLinear(nn.Linear):
    def to(self, *args, **kwargs):
        return self


class NoOpToCriterion(nn.CrossEntropyLoss):
    def to(self, *args, **kwargs):
        return self


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


def test_model_trainer_fails_if_cuda_selected_but_model_remains_on_cpu():
    spec = TrainingRunSpec(
        model_name="linear",
        model=NoOpToLinear(2, 2),
        train_loader=cast(DataLoader, []),
        val_loader=cast(DataLoader, []),
        criterion=NoOpToCriterion(),
        device=torch.device("cuda"),
        config=TrainingConfig(epochs=1, optimizer_name="sgd", learning_rate=0.01),
    )

    try:
        ModelTrainer(spec)
    except RuntimeError as exc:
        assert "CUDA was selected, but model is on cpu" in str(exc)
    else:
        raise AssertionError("Expected CPU model with CUDA spec to fail")


def test_model_trainer_evaluate_uses_inference_mode(monkeypatch):
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=2)
    entered = []

    class FakeInferenceMode:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(trainer_module.torch, "inference_mode", lambda: FakeInferenceMode())
    spec = TrainingRunSpec(
        model_name="linear",
        model=nn.Linear(2, 2),
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=1, optimizer_name="sgd", learning_rate=0.01),
    )

    ModelTrainer(spec).evaluate()

    assert entered == [True]


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
