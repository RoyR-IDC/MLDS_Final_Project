from types import SimpleNamespace
import builtins

import pandas as pd
import pytest
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments import part1, part2, training_runs
from src.experiments.results import save_run_rows
import src.training.trainer as trainer_module
from src.experiments.part1 import (
    IMAGENET1K_DOMESTIC_CAT_CLASS_INDICES,
    IMAGENET1K_DOG_CLASS_INDICES,
    train_unfrozen_pretrained_binary_head_on_tile_permutation_records,
    get_executable_tile_permutation_records,
    train_model_on_tile_permutation_records,
    zero_shot_cat_dog_predictions,
)
from src.experiments.part2 import (
    CORRUPTION_PROBABILITY_SCHEDULE,
    build_curriculum_schedule,
    train_part2_ablation_experiments,
)
from src.preprocessing.tile_permutations import TilePermutationRecord, deterministic_tile_permutation, identity_tile_permutation
from src.training.checkpoints import load_checkpoint, save_checkpoint
from src.training.run import CheckpointConfig, TrainingConfig, TrainingResult, TrainingRunSpec
from src.training.trainer import ModelTrainer


def test_executable_tile_permutation_records_returns_records():
    records = [
        TilePermutationRecord(tiles_per_side=None, tile_permutation_id=0, tile_permutation_seed=42, tile_permutation=None),
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=1,
            tile_permutation_seed=42,
            tile_permutation=deterministic_tile_permutation(2, "medium"),
        ),
    ]

    executable_records = get_executable_tile_permutation_records(records)

    assert [(record.tiles_per_side, record.tile_permutation_id) for record in executable_records] == [
        (None, 0),
        (2, 1),
    ]


def test_checkpoint_config_disabled_outside_colab(tmp_path):
    config = SimpleNamespace(
        part="part1",
        outputs_dir=str(tmp_path / "outputs"),
        using_google_colab=False,
    )
    record = TilePermutationRecord(
        tiles_per_side=2,
        tile_permutation_id=3,
        tile_permutation_seed=99,
        tile_permutation=identity_tile_permutation(2),
    )

    checkpoint_config = training_runs.build_checkpoint_config(
        config=config,
        run_id="run",
        model_name="resnet18",
        record=record,
    )

    assert checkpoint_config.save_best is False
    assert checkpoint_config.save_last is False
    assert checkpoint_config.best_path is None
    assert checkpoint_config.last_path is None
    assert not (tmp_path / "outputs" / "checkpoints").exists()


def test_checkpoint_config_enabled_on_colab(tmp_path):
    config = SimpleNamespace(
        part="part1",
        outputs_dir=str(tmp_path / "outputs"),
        using_google_colab=True,
    )
    record = TilePermutationRecord(
        tiles_per_side=2,
        tile_permutation_id=3,
        tile_permutation_seed=99,
        tile_permutation=identity_tile_permutation(2),
    )

    checkpoint_config = training_runs.build_checkpoint_config(
        config=config,
        run_id="run",
        model_name="resnet18",
        record=record,
    )

    assert checkpoint_config.save_best is True
    assert checkpoint_config.save_last is True
    assert checkpoint_config.best_path.endswith("resnet18__tiles_2__perm_3__best.pt")
    assert checkpoint_config.last_path.endswith("resnet18__tiles_2__perm_3__last.pt")
    assert checkpoint_config.resume is True
    assert checkpoint_config.resume_path == checkpoint_config.last_path
    assert (tmp_path / "outputs" / "checkpoints" / "part1" / "run").is_dir()


def test_colab_run_id_is_stable_for_same_config():
    config = SimpleNamespace(
        part="part1",
        config_name="part1_default",
        seed=42,
        using_google_colab=True,
    )

    assert training_runs.build_experiment_run_id(config) == "part1_part1_default_seed_42"


def _resume_test_spec(tmp_path, *, epochs: int, checkpoint_epoch: int) -> tuple[TrainingRunSpec, str, dict]:
    metadata = {
        "part": "part1",
        "config_name": "test_config",
        "run_id": "part1_test_config_seed_42",
        "model_name": "linear",
        "ablation_name": None,
        "tiles_per_side": 2,
        "tile_permutation_id": 3,
        "tile_permutation_seed": 99,
        "seed": 42,
        "optimizer_name": "sgd",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "use_amp": False,
    }
    checkpoint_path = str(tmp_path / "linear__tiles_2__perm_3__last.pt")
    checkpoint_model = nn.Linear(2, 2)
    save_checkpoint(
        path=checkpoint_path,
        model=checkpoint_model,
        optimizer=None,
        epoch=checkpoint_epoch,
        metrics={
            "train_loss": 0.4,
            "train_accuracy": 0.8,
            "val_loss": 0.5,
            "val_accuracy": 0.75,
            "best_val_accuracy": 0.75,
        },
        metadata=metadata,
        planned_total_epochs=epochs,
    )

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    spec = TrainingRunSpec(
        model_name="linear",
        model=nn.Linear(2, 2),
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=epochs, optimizer_name="sgd", learning_rate=0.01),
        checkpoint_config=CheckpointConfig(
            best_path=str(tmp_path / "linear__tiles_2__perm_3__best.pt"),
            last_path=checkpoint_path,
        ),
        metadata=metadata,
        progress_leave=False,
    )
    return spec, checkpoint_path, metadata


def test_model_trainer_resumes_partial_last_checkpoint(tmp_path):
    spec, checkpoint_path, _ = _resume_test_spec(tmp_path, epochs=2, checkpoint_epoch=1)

    result = ModelTrainer(spec).fit()

    assert result.status == "completed"
    assert result.resumed_from_checkpoint == checkpoint_path
    assert result.resumed_from_epoch == 1
    assert [epoch_result.epoch for epoch_result in result.epoch_history] == [1, 2]
    resumed_checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    assert resumed_checkpoint["epoch"] == 2


def test_model_trainer_skips_complete_last_checkpoint(monkeypatch, tmp_path):
    spec, checkpoint_path, _ = _resume_test_spec(tmp_path, epochs=2, checkpoint_epoch=2)

    def fail_if_training_starts(*args, **kwargs):
        raise AssertionError("complete checkpoints should skip training")

    monkeypatch.setattr(ModelTrainer, "train_one_epoch", fail_if_training_starts)

    result = ModelTrainer(spec).fit()

    assert result.status == "completed"
    assert result.skipped_from_checkpoint == checkpoint_path
    assert result.val_accuracy == 0.75
    assert result.last_checkpoint_path == checkpoint_path


def test_model_trainer_ignores_mismatched_resume_checkpoint(tmp_path):
    spec, checkpoint_path, metadata = _resume_test_spec(tmp_path, epochs=1, checkpoint_epoch=1)
    mismatched_metadata = {**metadata, "tile_permutation_id": 999}
    save_checkpoint(
        path=checkpoint_path,
        model=nn.Linear(2, 2),
        optimizer=None,
        epoch=1,
        metrics={
            "train_loss": 0.4,
            "train_accuracy": 0.8,
            "val_loss": 0.5,
            "val_accuracy": 0.75,
            "best_val_accuracy": 0.75,
        },
        metadata=mismatched_metadata,
        planned_total_epochs=1,
    )

    result = ModelTrainer(spec).fit()

    assert result.status == "completed"
    assert result.resumed_from_checkpoint is None
    assert result.skipped_from_checkpoint is None


def test_build_training_run_spec_rejects_missing_pretrained_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(timm, "is_model_pretrained", lambda model_id: False)
    model_calls = []

    def fake_get_model(*args, **kwargs):
        model_calls.append((args, kwargs))
        return nn.Linear(2, 2)

    monkeypatch.setattr(training_runs, "get_model", fake_get_model)

    config = SimpleNamespace(
        part="part1",
        config_name="part1",
        outputs_dir=str(tmp_path / "outputs"),
        using_google_colab=False,
        epochs=1,
        optimizer="adamw",
        learning_rate=0.0003,
        weight_decay=0.0001,
        use_amp=False,
        profile_performance=False,
        profile_warmup_batches=0,
        image_size=224,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
    )
    loader = DataLoader(TensorDataset(torch.zeros(2, 3, 224, 224), torch.zeros(2, dtype=torch.long)))
    record = TilePermutationRecord(
        tiles_per_side=None,
        tile_permutation_id=0,
        tile_permutation_seed=42,
        tile_permutation=None,
    )

    with pytest.raises(ValueError, match="pretrained ImageNet-1K weights"):
        spec = training_runs.build_training_run_spec(
            config=config,
            model_name="gmlp_s16",
            train_loader=loader,
            val_loader=loader,
            device=torch.device("cpu"),
            run_id="run",
            record=record,
            seed=42,
        )
    assert model_calls == []


def test_build_training_run_spec_distinguishes_auto_from_disabled_expected_input_size(monkeypatch):
    monkeypatch.setattr(training_runs, "get_model", lambda *args, **kwargs: nn.Linear(2, 2))
    config = SimpleNamespace(
        part="part1",
        config_name="part1",
        outputs_dir="outputs",
        using_google_colab=False,
        epochs=1,
        optimizer="adamw",
        learning_rate=0.0003,
        weight_decay=0.0001,
        use_amp=False,
        profile_performance=False,
        profile_warmup_batches=0,
        image_size=224,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
    )
    loader = DataLoader(TensorDataset(torch.zeros(2, 3, 224, 224), torch.zeros(2, dtype=torch.long)))
    record = TilePermutationRecord(
        tiles_per_side=10,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(10, "medium"),
    )

    auto_spec = training_runs.build_training_run_spec(
        config=config,
        model_name="resnet18",
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        run_id="run",
        record=record,
        seed=42,
    )
    disabled_spec = training_runs.build_training_run_spec(
        config=config,
        model_name="resnet18",
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        run_id="run",
        record=record,
        seed=42,
        expected_input_size=None,
    )

    assert auto_spec.expected_input_size == 230
    assert disabled_spec.expected_input_size is None


def test_model_trainer_shows_training_batch_progress_per_epoch(monkeypatch):
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    spec = TrainingRunSpec(
        model_name="linear",
        model=nn.Linear(2, 2),
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=2, optimizer_name="sgd", learning_rate=0.01),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
        progress_desc="linear test epochs",
        progress_leave=False,
    )
    progress_bars = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.updated = 0
            self.postfixes = []
            progress_bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self, amount=1):
            self.updated += amount

        def set_postfix(self, **kwargs):
            self.postfixes.append(kwargs)

    monkeypatch.setattr(trainer_module, "tqdm", FakeTqdm)

    result = ModelTrainer(spec).fit()

    assert result.status == "completed"
    epoch_bars = [bar for bar in progress_bars if bar.kwargs["unit"] == "epoch"]
    batch_bars = [bar for bar in progress_bars if bar.kwargs["unit"] == "batch"]
    assert len(epoch_bars) == 1
    assert epoch_bars[0].updated == 2
    assert len(batch_bars) == 2
    assert [bar.kwargs["total"] for bar in batch_bars] == [len(loader), len(loader)]
    assert [bar.updated for bar in batch_bars] == [len(loader), len(loader)]
    assert all(bar.kwargs["position"] == 1 for bar in batch_bars)
    assert all(bar.kwargs["leave"] is False for bar in batch_bars)


def test_model_trainer_prints_first_batch_before_progress_bar(monkeypatch):
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    spec = TrainingRunSpec(
        model_name="linear",
        model=nn.Linear(2, 2),
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=1, optimizer_name="sgd", learning_rate=0.01),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
        progress_leave=False,
    )
    events = []
    original_print = builtins.print

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            events.append(f"tqdm:{kwargs['unit']}")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self, amount=1):
            pass

        def set_postfix(self, **kwargs):
            pass

    def tracked_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        if message.startswith("First batch:"):
            events.append("first_batch")
        original_print(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(builtins, "print", tracked_print)

    ModelTrainer(spec).fit()

    assert events.index("first_batch") < events.index("tqdm:epoch")


def test_model_trainer_validates_expected_image_shape():
    features = torch.rand(2, 3, 7, 8)
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    spec = TrainingRunSpec(
        model_name="flatten_linear",
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 7 * 8, 2)),
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config=TrainingConfig(epochs=1, optimizer_name="sgd", learning_rate=0.01),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
        expected_input_size=8,
        progress_leave=False,
    )

    try:
        ModelTrainer(spec).fit()
    except ValueError as exc:
        assert "Expected image tensors with shape" in str(exc)
    else:
        raise AssertionError("Expected invalid image shape to raise")


def test_model_trainer_profiles_train_and_validation_phases():
    features = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    spec = TrainingRunSpec(
        model_name="flatten_linear",
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2)),
        train_loader=loader,
        val_loader=loader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        config=TrainingConfig(
            epochs=1,
            optimizer_name="sgd",
            learning_rate=0.01,
            profile_performance=True,
        ),
        checkpoint_config=CheckpointConfig(save_best=False, save_last=False),
        metadata={"run_id": "run", "tile_permutation_id": 0},
        expected_input_size=8,
        progress_leave=False,
    )

    result = ModelTrainer(spec).fit()

    assert [row["phase"] for row in result.profile_rows] == ["train", "val"]
    assert all(row["total_batches"] == len(loader) for row in result.profile_rows)
    assert all(row["measured_batches"] == len(loader) for row in result.profile_rows)
    assert all(row["total_seconds"] >= row["compute_seconds"] for row in result.profile_rows)


def test_train_model_on_tile_permutation_records_saves_mid_run_updates(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
        epochs=1,
        outputs_dir=str(tmp_path / "outputs"),
    )
    record = TilePermutationRecord(
        tiles_per_side=2,
        tile_permutation_id=3,
        tile_permutation_seed=99,
        tile_permutation=identity_tile_permutation(2),
    )
    progress_descriptions = []

    class FakeLoader:
        dataset = [None]

        def __len__(self):
            return 1

    class FakeTrainer:
        def __init__(self, spec):
            self.spec = spec

        def fit(self, on_progress):
            result = TrainingResult.pending(model_name=self.spec.model_name, metadata={})
            result.mark_running()
            on_progress(result)
            result.train_loss = 0.4
            result.train_accuracy = 0.8
            result.val_loss = 0.5
            result.val_accuracy = 0.75
            result.best_val_accuracy = 0.75
            result.best_checkpoint_path = "best.pt"
            on_progress(result)
            result.mark_completed(2.5)
            on_progress(result)
            return result

    def fake_build_dataloaders(**kwargs):
        return FakeLoader(), FakeLoader()

    def fake_build_training_run_spec(**kwargs):
        progress_descriptions.append(kwargs["progress_desc"])
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part1, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(part1, "build_training_run_spec", fake_build_training_run_spec)
    monkeypatch.setattr(training_runs, "ModelTrainer", FakeTrainer)

    raw_results_path = tmp_path / "part1_raw_results.csv"
    rows = train_model_on_tile_permutation_records(
        config=config,
        model_name="resnet18",
        run_id="run",
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        tile_permutation_records=[record],
        seed=42,
        device="cpu",
        raw_results_output_path=str(raw_results_path),
    )

    assert rows[0]["training_duration_seconds"] == 2.5
    assert rows[0]["run_status"] == "completed"
    assert rows[0]["best_checkpoint_path"] == "best.pt"
    assert progress_descriptions == ["resnet18 2x2 permutation 3"]

    saved = pd.read_csv(raw_results_path)
    assert saved.loc[0, "run_status"] == "completed"
    assert saved.loc[0, "training_duration_seconds"] == 2.5


def test_save_run_rows_preserves_baseline_when_saving_ablation(tmp_path):
    raw_results_path = tmp_path / "part1_raw_results.csv"
    baseline_row = {
        "run_id": "run",
        "model_name": "resnet18",
        "tile_permutation_id": 1,
        "val_accuracy": 0.8,
    }
    ablation_row = {
        "run_id": "run",
        "model_name": "resnet18",
        "ablation_name": "unfrozen_pretrained_binary_head",
        "tile_permutation_id": 1,
        "val_accuracy": 0.9,
    }

    save_run_rows(rows=[baseline_row], output_path=str(raw_results_path), run_id="run", model_name="resnet18")
    save_run_rows(rows=[ablation_row], output_path=str(raw_results_path), run_id="run", model_name="resnet18")

    saved = pd.read_csv(raw_results_path)
    assert len(saved) == 2
    assert set(saved["val_accuracy"]) == {0.8, 0.9}
    assert saved["ablation_name"].isna().sum() == 1


def test_unfrozen_part1_variant_sets_freeze_backbone_false(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=True,
        epochs=1,
        outputs_dir=str(tmp_path / "outputs"),
    )
    record = TilePermutationRecord(
        tiles_per_side=2,
        tile_permutation_id=3,
        tile_permutation_seed=99,
        tile_permutation=identity_tile_permutation(2),
    )
    captured_kwargs = {}

    class FakeLoader:
        dataset = [None]

        def __len__(self):
            return 1

    class FakeTrainer:
        def __init__(self, spec):
            self.spec = spec

        def fit(self, on_progress):
            result = TrainingResult.pending(model_name=self.spec.model_name, metadata={})
            result.val_accuracy = 0.75
            result.best_val_accuracy = 0.75
            result.mark_completed(1.0)
            on_progress(result)
            return result

    def fake_build_training_run_spec(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part1, "build_dataloaders", lambda **kwargs: (FakeLoader(), FakeLoader()))
    monkeypatch.setattr(part1, "build_training_run_spec", fake_build_training_run_spec)
    monkeypatch.setattr(training_runs, "ModelTrainer", FakeTrainer)

    rows = train_unfrozen_pretrained_binary_head_on_tile_permutation_records(
        config=config,
        model_name="resnet18",
        run_id="run",
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        tile_permutation_records=[record],
        seed=42,
        device="cpu",
    )

    assert captured_kwargs["overrides"]["pretrained"] is True
    assert captured_kwargs["overrides"]["freeze_backbone"] is False
    assert captured_kwargs["ablation_name"] == "unfrozen_pretrained_binary_head"
    assert rows[0]["ablation_name"] == "unfrozen_pretrained_binary_head"


def test_zero_shot_cat_dog_predictions_sum_imagenet_probabilities():
    logits = torch.full((3, 1000), -20.0)
    logits[0, IMAGENET1K_DOG_CLASS_INDICES[0]] = 8.0
    logits[0, IMAGENET1K_DOMESTIC_CAT_CLASS_INDICES[0]] = 2.0
    logits[1, IMAGENET1K_DOG_CLASS_INDICES[0]] = 1.0
    logits[1, IMAGENET1K_DOMESTIC_CAT_CLASS_INDICES[0]] = 7.0
    logits[2, IMAGENET1K_DOG_CLASS_INDICES[0]] = 3.0
    logits[2, IMAGENET1K_DOMESTIC_CAT_CLASS_INDICES[0]] = 3.0

    predictions = zero_shot_cat_dog_predictions(logits)

    assert predictions.tolist() == [1, 0, 0]


def test_zero_shot_cat_dog_predictions_requires_full_imagenet_logits():
    with pytest.raises(ValueError, match="ImageNet-1k logits"):
        zero_shot_cat_dog_predictions(torch.zeros((2, 2)))


def test_train_model_on_tile_permutation_records_saves_failed_status(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
        epochs=1,
        outputs_dir=str(tmp_path / "outputs"),
    )
    records = [
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=0,
            tile_permutation_seed=99,
            tile_permutation=identity_tile_permutation(2),
        ),
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=1,
            tile_permutation_seed=100,
            tile_permutation=deterministic_tile_permutation(2, "medium"),
        ),
    ]
    calls = iter(["ok", "fail"])

    class FakeLoader:
        dataset = [None]

        def __len__(self):
            return 1

    class FakeTrainer:
        def __init__(self, spec):
            self.spec = spec

        def fit(self, on_progress):
            mode = next(calls)
            result = TrainingResult.pending(model_name=self.spec.model_name, metadata={})
            result.mark_running()
            on_progress(result)
            if mode == "fail":
                result.mark_failed(1.0, RuntimeError("simulated crash"))
                on_progress(result)
                raise RuntimeError("simulated crash")
            result.best_val_accuracy = 0.75
            result.mark_completed(2.5)
            on_progress(result)
            return result

    monkeypatch.setattr(part1, "build_dataloaders", lambda **kwargs: (FakeLoader(), FakeLoader()))
    monkeypatch.setattr(part1, "build_training_run_spec", lambda **kwargs: SimpleNamespace(model_name=kwargs["model_name"]))
    monkeypatch.setattr(training_runs, "ModelTrainer", FakeTrainer)

    raw_results_path = tmp_path / "part1_raw_results.csv"
    try:
        train_model_on_tile_permutation_records(
            config=config,
            model_name="resnet18",
            run_id="run",
            train_samples=[("cat.jpg", 0)],
            validation_samples=[("dog.jpg", 1)],
            tile_permutation_records=records,
            seed=42,
            device="cpu",
            raw_results_output_path=str(raw_results_path),
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated crash"
    else:
        raise AssertionError("Expected simulated crash")

    saved = pd.read_csv(raw_results_path).sort_values("tile_permutation_id")
    assert saved["run_status"].tolist() == ["completed", "failed"]
    assert saved["error_message"].tolist()[1] == "simulated crash"


def test_train_model_and_save_progress_writes_profile_rows(monkeypatch, tmp_path):
    config = SimpleNamespace(part="part1", config_name="test_config")
    record = TilePermutationRecord(
        tiles_per_side=2,
        tile_permutation_id=0,
        tile_permutation_seed=99,
        tile_permutation=identity_tile_permutation(2),
    )
    rows = [
        training_runs.build_pending_training_result_row(
            config=config,
            run_id="run",
            model_name="resnet18",
            record=record,
            seed=42,
        )
    ]
    profile_path = tmp_path / "profile.csv"
    spec = SimpleNamespace(
        model_name="resnet18",
        config=SimpleNamespace(profile_performance=True),
        profile_output_path=str(profile_path),
    )

    class FakeTrainer:
        def __init__(self, spec):
            self.spec = spec

        def fit(self, on_progress):
            result = TrainingResult.pending(model_name=self.spec.model_name, metadata={})
            result.profile_rows = [
                {
                    "run_id": "run",
                    "model_name": self.spec.model_name,
                    "phase": "train",
                    "total_seconds": 1.25,
                }
            ]
            result.mark_completed(1.25)
            on_progress(result)
            return result

    monkeypatch.setattr(training_runs, "ModelTrainer", FakeTrainer)

    training_runs.train_model_and_save_progress(
        spec=spec,
        config=config,
        run_id="run",
        record=record,
        seed=42,
        rows=rows,
        row_index=0,
        raw_results_output_path=None,
    )

    saved = pd.read_csv(profile_path)
    assert saved[["run_id", "model_name", "phase", "total_seconds"]].to_dict("records") == [
        {
            "run_id": "run",
            "model_name": "resnet18",
            "phase": "train",
            "total_seconds": 1.25,
        }
    ]


def test_train_part2_ablation_experiments_uses_same_trainer_core(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part2",
        config_name="part2_improvement",
        model_name="resnet18",
        model_names=["resnet18"],
        image_size=32,
        batch_size=4,
        num_workers=0,
        seed=42,
        pretrained=True,
        epochs=1,
        outputs_dir=str(tmp_path / "outputs"),
    )
    records = [
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=0,
            tile_permutation_seed=99,
            tile_permutation=identity_tile_permutation(2),
        )
    ]
    ablations = [
        {
            "name": "augmentation_patch_shuffle",
            "use_pretrained": True,
            "augmentation": "patch_shuffle",
        }
    ]
    progress_descriptions = []
    spec_kwargs = []
    plotted_paths = []
    callback_paths = []

    class FakeLoader:
        dataset = [None]

        def __len__(self):
            return 1

    class FakeTrainer:
        def __init__(self, spec):
            self.spec = spec

        def fit(self, on_progress):
            result = TrainingResult.pending(model_name=self.spec.model_name, metadata={})
            result.best_val_accuracy = 0.75
            result.mark_completed(2.5)
            on_progress(result)
            return result

    def fake_build_training_run_spec(**kwargs):
        progress_descriptions.append(kwargs["progress_desc"])
        spec_kwargs.append(kwargs)
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part2, "build_dataloaders", lambda **kwargs: (FakeLoader(), FakeLoader()))
    monkeypatch.setattr(part2, "build_training_run_spec", fake_build_training_run_spec)
    monkeypatch.setattr(part2, "plot_ablation_results", lambda aggregated, path, **kwargs: plotted_paths.append(path))
    monkeypatch.setattr(training_runs, "ModelTrainer", FakeTrainer)

    raw_results_path = tmp_path / "part2_raw_results.csv"
    rows = train_part2_ablation_experiments(
        config=config,
        ablations=ablations,
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        tile_permutation_records=records,
        device="cpu",
        run_id="part2_run",
        raw_results_output_path=str(raw_results_path),
        intermediate_figures_dir=str(tmp_path / "figures"),
        intermediate_figure_callback=callback_paths.append,
    )

    assert progress_descriptions == ["resnet18 [augmentation_patch_shuffle] 2x2 permutation #0. epochs progress"]
    assert spec_kwargs[0]["overrides"]["freeze_backbone"] is True
    assert spec_kwargs[0]["metadata_overrides"]["augmentation_name"] == "patch_shuffle"
    assert spec_kwargs[0]["metadata_overrides"]["hardness_level"] == "baseline"
    assert spec_kwargs[0]["metadata_overrides"]["epochs"] == 1
    assert rows[0]["run_status"] == "completed"
    assert rows[0]["ablation_name"] == "augmentation_patch_shuffle"
    assert plotted_paths == [str(tmp_path / "figures" / "intermediate" / "part2_ablation_augmentation_patch_shuffle.png")]
    assert callback_paths == plotted_paths


def test_part2_regular_augmentations_use_train_only_image_augmentation(monkeypatch):
    config = SimpleNamespace(image_size=32, batch_size=4, num_workers=0)
    record = TilePermutationRecord(
        tiles_per_side=4,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(4, "medium"),
    )
    captured_kwargs = {}

    class FakeLoader:
        pass

    def fake_build_dataloaders(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeLoader(), FakeLoader()

    monkeypatch.setattr(part2, "build_dataloaders", fake_build_dataloaders)

    part2.build_ablation_dataloaders(
        ablation={"name": "regular_augmentations", "augmentation": "regular_augmentations"},
        record=record,
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        config=config,
    )

    assert captured_kwargs["image_augmentation"] == "regular_augmentations"
    assert captured_kwargs["tile_permutation_probability"] == 1.0


def test_part2_mixed_original_permuted_maps_p_original_to_training_probability(monkeypatch):
    config = SimpleNamespace(image_size=32, batch_size=4, num_workers=0)
    record = TilePermutationRecord(
        tiles_per_side=10,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(10, "medium"),
    )
    captured_kwargs = {}

    class FakeLoader:
        pass

    def fake_build_dataloaders(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeLoader(), FakeLoader()

    monkeypatch.setattr(part2, "build_dataloaders", fake_build_dataloaders)

    part2.build_ablation_dataloaders(
        ablation={"name": "mixed_original_permuted", "p_original": 0.5},
        record=record,
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        config=config,
    )

    assert captured_kwargs["tiles_per_side"] == 10
    assert captured_kwargs["tile_permutation"] == record.tile_permutation
    assert captured_kwargs["tile_permutation_probability"] == 0.5


def test_part2_mixed_original_permuted_logs_p_original_metadata(monkeypatch):
    config = SimpleNamespace(
        part="part2",
        config_name="part2_improvement",
        image_size=224,
        batch_size=4,
        num_workers=0,
        seed=42,
        pretrained=True,
        freeze_backbone=True,
        optimizer="adamw",
        learning_rate=0.0003,
        weight_decay=0.0001,
        use_amp=False,
        profile_performance=False,
        profile_warmup_batches=0,
        outputs_dir="outputs",
        results_dir="outputs/results",
        epochs=10,
    )
    record = TilePermutationRecord(
        tiles_per_side=4,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(4, "medium"),
    )
    captured_kwargs = {}

    class FakeLoader:
        pass

    def fake_build_training_run_spec(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part2, "build_training_run_spec", fake_build_training_run_spec)

    part2.build_part2_training_run_spec(
        config=config,
        model_name="resnet18",
        run_id="part2_run",
        ablation={"name": "mixed_original_permuted", "p_original": 0.5},
        record=record,
        train_loader=FakeLoader(),
        validation_loader=FakeLoader(),
        device="cpu",
        batch_augmentation=None,
        curriculum_schedule=None,
    )

    metadata = captured_kwargs["metadata_overrides"]
    assert metadata["p_original"] == 0.5
    assert metadata["tile_permutation_probability"] == 0.5


def test_part2_mlp_head_ablation_sets_overrides_and_metadata(monkeypatch):
    config = SimpleNamespace(
        part="part2",
        config_name="part2_improvement",
        image_size=224,
        batch_size=4,
        num_workers=0,
        seed=42,
        pretrained=True,
        freeze_backbone=True,
        optimizer="adamw",
        learning_rate=0.0003,
        weight_decay=0.0001,
        use_amp=False,
        profile_performance=False,
        profile_warmup_batches=0,
        outputs_dir="outputs",
        results_dir="outputs/results",
        epochs=10,
    )
    record = TilePermutationRecord(
        tiles_per_side=4,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(4, "medium"),
    )
    captured_kwargs = {}

    class FakeLoader:
        pass

    def fake_build_training_run_spec(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part2, "build_training_run_spec", fake_build_training_run_spec)

    part2.build_part2_training_run_spec(
        config=config,
        model_name="resnet18",
        run_id="part2_run",
        ablation={"name": "cnn_mlp_head", "classification_head": "mlp"},
        record=record,
        train_loader=FakeLoader(),
        validation_loader=FakeLoader(),
        device="cpu",
        batch_augmentation=None,
        curriculum_schedule=None,
    )

    assert captured_kwargs["overrides"]["classification_head"] == "mlp"
    assert captured_kwargs["metadata_overrides"]["classification_head"] == "MLP"


def test_difficulty_curriculum_builds_stages_up_to_target(monkeypatch):
    config = SimpleNamespace(image_size=32, batch_size=4, num_workers=0, epochs=5, seed=42)
    record = TilePermutationRecord(
        tiles_per_side=4,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(4, "medium"),
    )

    class FakeLoader:
        pass

    monkeypatch.setattr(part2, "build_dataloaders", lambda **kwargs: (FakeLoader(), FakeLoader()))

    schedule = build_curriculum_schedule(
        ablation={"curriculum": "permutation_difficulty"},
        record=record,
        train_samples=[("cat.jpg", 0), ("dog.jpg", 1)],
        config=config,
    )

    assert schedule is not None
    assert schedule.stage_names == ["original", "2x2_permutation", "3x3_permutation", "4x4_permutation"]
    assert schedule.total_epochs == 5


def test_curriculum_ablation_epoch_override_controls_stage_total(monkeypatch):
    config = SimpleNamespace(image_size=32, batch_size=4, num_workers=0, epochs=10, seed=42)
    record = TilePermutationRecord(
        tiles_per_side=4,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(4, "medium"),
    )

    class FakeLoader:
        pass

    monkeypatch.setattr(part2, "build_dataloaders", lambda **kwargs: (FakeLoader(), FakeLoader()))

    schedule = build_curriculum_schedule(
        ablation={"curriculum": "permutation_difficulty", "epochs": 30},
        record=record,
        train_samples=[("cat.jpg", 0), ("dog.jpg", 1)],
        config=config,
    )

    assert schedule is not None
    assert schedule.total_epochs == 30


def test_difficulty_curriculum_includes_target_grid_stage(monkeypatch):
    config = SimpleNamespace(image_size=32, batch_size=4, num_workers=0, epochs=10, seed=42)
    record = TilePermutationRecord(
        tiles_per_side=10,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(10, "medium"),
    )
    loader_tiles = []

    class FakeLoader:
        pass

    def fake_build_dataloaders(**kwargs):
        loader_tiles.append(kwargs["tiles_per_side"])
        return FakeLoader(), FakeLoader()

    monkeypatch.setattr(part2, "build_dataloaders", fake_build_dataloaders)

    schedule = build_curriculum_schedule(
        ablation={"curriculum": "permutation_difficulty"},
        record=record,
        train_samples=[("cat.jpg", 0), ("dog.jpg", 1)],
        config=config,
    )

    assert schedule is not None
    assert schedule.stage_names == [
        "original",
        "2x2_permutation",
        "3x3_permutation",
        "4x4_permutation",
        "10x10_permutation",
    ]
    assert loader_tiles == [1, 2, 3, 4, 10]
    assert schedule.total_epochs == 10


def test_difficulty_curriculum_spec_allows_variable_stage_image_sizes(monkeypatch):
    config = SimpleNamespace(
        part="part2",
        config_name="part2_improvement",
        image_size=224,
        batch_size=4,
        num_workers=0,
        seed=42,
        pretrained=True,
        freeze_backbone=True,
        optimizer="adamw",
        learning_rate=0.0003,
        weight_decay=0.0001,
        use_amp=False,
        profile_performance=False,
        profile_warmup_batches=0,
        outputs_dir="outputs",
        results_dir="outputs/results",
    )
    record = TilePermutationRecord(
        tiles_per_side=10,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(10, "medium"),
    )
    captured_kwargs = {}

    class FakeLoader:
        pass

    def fake_build_training_run_spec(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part2, "build_training_run_spec", fake_build_training_run_spec)

    part2.build_part2_training_run_spec(
        config=config,
        model_name="resnet18",
        run_id="part2_run",
        ablation={"name": "curriculum_permutation_difficulty", "curriculum": "permutation_difficulty"},
        record=record,
        train_loader=FakeLoader(),
        validation_loader=FakeLoader(),
        device="cpu",
        batch_augmentation=None,
        curriculum_schedule=SimpleNamespace(stage_names=["original", "10x10_permutation"]),
    )

    assert captured_kwargs["expected_input_size"] == 224


def test_corruption_probability_curriculum_uses_expected_probabilities(monkeypatch):
    config = SimpleNamespace(image_size=32, batch_size=4, num_workers=0, epochs=4, seed=42)
    record = TilePermutationRecord(
        tiles_per_side=3,
        tile_permutation_id=1,
        tile_permutation_seed=42,
        tile_permutation=deterministic_tile_permutation(3, "medium"),
    )
    probabilities = []

    class FakeLoader:
        pass

    def fake_build_dataloaders(**kwargs):
        probabilities.append(kwargs.get("tile_permutation_probability"))
        return FakeLoader(), FakeLoader()

    monkeypatch.setattr(part2, "build_dataloaders", fake_build_dataloaders)

    schedule = build_curriculum_schedule(
        ablation={"curriculum": "corruption_probability"},
        record=record,
        train_samples=[("cat.jpg", 0), ("dog.jpg", 1)],
        config=config,
    )

    assert schedule is not None
    assert probabilities == CORRUPTION_PROBABILITY_SCHEDULE
    assert schedule.total_epochs == 4
