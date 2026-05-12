from types import SimpleNamespace

import pandas as pd

from src.experiments import part1, part2
from src.experiments.part1 import collect_model_tile_permutation_results, get_executable_tile_permutation_records
from src.experiments.part2 import collect_part2_ablation_results
from src.preprocessing.tile_permutations import TilePermutationRecord, identity_tile_permutation, random_tile_permutation
from src.training.run import TrainingResult


def test_executable_tile_permutation_records_returns_records():
    records = [
        TilePermutationRecord(tiles_per_side=None, tile_permutation_id=0, tile_permutation_seed=42, tile_permutation=None),
        TilePermutationRecord(
            tiles_per_side=2,
            tile_permutation_id=1,
            tile_permutation_seed=42,
            tile_permutation=random_tile_permutation(2, seed=42),
        ),
    ]

    executable_records = get_executable_tile_permutation_records(records)

    assert [(record.tiles_per_side, record.tile_permutation_id) for record in executable_records] == [
        (None, 0),
        (2, 1),
    ]


def test_collect_model_tile_permutation_results_saves_mid_run_updates(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
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

    def fake_build_training_spec(**kwargs):
        progress_descriptions.append(kwargs["progress_desc"])
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part1, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(part1, "build_training_spec", fake_build_training_spec)
    monkeypatch.setattr(part1, "ModelTrainer", FakeTrainer)

    raw_results_path = tmp_path / "part1_raw_results.csv"
    rows = collect_model_tile_permutation_results(
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


def test_collect_model_tile_permutation_results_saves_failed_status(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
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
            tile_permutation=random_tile_permutation(2, seed=100),
        ),
    ]
    calls = iter(["ok", "fail"])

    class FakeLoader:
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
    monkeypatch.setattr(part1, "build_training_spec", lambda **kwargs: SimpleNamespace(model_name=kwargs["model_name"]))
    monkeypatch.setattr(part1, "ModelTrainer", FakeTrainer)

    raw_results_path = tmp_path / "part1_raw_results.csv"
    try:
        collect_model_tile_permutation_results(
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


def test_collect_part2_ablation_results_uses_same_trainer_core(monkeypatch, tmp_path):
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
            "name": "augmentation_only",
            "use_pretrained": True,
            "use_standard_augmentation": True,
            "freeze_backbone": False,
        }
    ]
    progress_descriptions = []

    class FakeLoader:
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

    def fake_build_training_spec(**kwargs):
        progress_descriptions.append(kwargs["progress_desc"])
        return SimpleNamespace(model_name=kwargs["model_name"])

    monkeypatch.setattr(part2, "build_dataloaders", lambda **kwargs: (FakeLoader(), FakeLoader()))
    monkeypatch.setattr(part2, "build_training_spec", fake_build_training_spec)
    monkeypatch.setattr(part1, "ModelTrainer", FakeTrainer)

    raw_results_path = tmp_path / "part2_raw_results.csv"
    rows = collect_part2_ablation_results(
        config=config,
        ablations=ablations,
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        tile_permutation_records=records,
        device="cpu",
        run_id="part2_run",
        raw_results_output_path=str(raw_results_path),
    )

    assert progress_descriptions == ["resnet18 augmentation_only 2x2 permutation 0"]
    assert rows[0]["run_status"] == "completed"
    assert rows[0]["ablation_name"] == "augmentation_only"
