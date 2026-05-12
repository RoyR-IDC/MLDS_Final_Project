from types import SimpleNamespace

import pandas as pd

from src.preprocessing.tile_permutations import TilePermutationRecord, identity_tile_permutation, random_tile_permutation
from src.training import experiment_steps
from src.training.experiment_steps import (
    collect_model_tile_permutation_results,
    collect_part2_ablation_results,
    get_executable_tile_permutation_records,
)


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


def test_collect_model_tile_permutation_results_records_training_duration_and_progress_desc(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
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

    def fake_build_dataloaders(**kwargs):
        return FakeLoader(), FakeLoader()

    def fake_train_and_evaluate_model_configuration(**kwargs):
        progress_descriptions.append(kwargs["progress_desc"])
        return {"best_val_accuracy": 0.75}

    timer_values = iter([10.0, 12.5])

    monkeypatch.setattr(experiment_steps, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(
        experiment_steps,
        "train_and_evaluate_model_configuration",
        fake_train_and_evaluate_model_configuration,
    )
    monkeypatch.setattr(experiment_steps, "perf_counter", lambda: next(timer_values))

    raw_results_path = tmp_path / "part1_raw_results.csv"
    pd.DataFrame(
        [
            {
                "part": "part1",
                "run_id": "run",
                "config_name": "test_config",
                "model_name": "deit_tiny",
                "tiles_per_side": 1,
                "num_tiles": 1,
                "tile_permutation_id": 0,
                "tile_permutation_seed": 99,
                "seed": 42,
                "best_val_accuracy": 0.5,
            }
        ]
    ).to_csv(raw_results_path, index=False)

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
    assert progress_descriptions == ["resnet18 2x2 permutation 3"]

    saved = pd.read_csv(raw_results_path)
    assert sorted(saved["model_name"]) == ["deit_tiny", "resnet18"]
    assert saved.loc[saved["model_name"] == "resnet18", "training_duration_seconds"].iloc[0] == 2.5
    assert saved.loc[saved["model_name"] == "resnet18", "run_status"].iloc[0] == "completed"


def test_collect_model_tile_permutation_results_leaves_pending_placeholders_after_crash(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
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

    class FakeLoader:
        def __len__(self):
            return 1

    def fake_build_dataloaders(**kwargs):
        return FakeLoader(), FakeLoader()

    train_calls = iter(
        [
            {"best_val_accuracy": 0.75},
            RuntimeError("simulated crash"),
        ]
    )

    def fake_train_and_evaluate_model_configuration(**kwargs):
        result = next(train_calls)
        if isinstance(result, Exception):
            raise result
        return result

    timer_values = iter([10.0, 12.5, 20.0])

    monkeypatch.setattr(experiment_steps, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(
        experiment_steps,
        "train_and_evaluate_model_configuration",
        fake_train_and_evaluate_model_configuration,
    )
    monkeypatch.setattr(experiment_steps, "perf_counter", lambda: next(timer_values))

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

    assert saved["run_status"].tolist() == ["completed", "pending"]
    assert saved["training_duration_seconds"].tolist()[0] == 2.5
    assert pd.isna(saved["training_duration_seconds"].tolist()[1])


def test_collect_part2_ablation_results_saves_placeholders_and_completed_rows(monkeypatch, tmp_path):
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

    def fake_build_dataloaders(**kwargs):
        return FakeLoader(), FakeLoader()

    train_calls = iter(
        [
            {"best_val_accuracy": 0.75},
            RuntimeError("simulated crash"),
        ]
    )

    def fake_train_and_evaluate_model_configuration(**kwargs):
        progress_descriptions.append(kwargs["progress_desc"])
        result = next(train_calls)
        if isinstance(result, Exception):
            raise result
        return result

    timer_values = iter([10.0, 12.5, 20.0])

    monkeypatch.setattr(experiment_steps, "build_dataloaders", fake_build_dataloaders)
    monkeypatch.setattr(
        experiment_steps,
        "train_and_evaluate_model_configuration",
        fake_train_and_evaluate_model_configuration,
    )
    monkeypatch.setattr(experiment_steps, "perf_counter", lambda: next(timer_values))

    raw_results_path = tmp_path / "part2_raw_results.csv"

    try:
        collect_part2_ablation_results(
            config=config,
            ablations=ablations,
            train_samples=[("cat.jpg", 0)],
            validation_samples=[("dog.jpg", 1)],
            tile_permutation_records=records,
            device="cpu",
            run_id="part2_run",
            raw_results_output_path=str(raw_results_path),
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated crash"
    else:
        raise AssertionError("Expected simulated crash")

    saved = pd.read_csv(raw_results_path).sort_values("tile_permutation_id")

    assert progress_descriptions == [
        "resnet18 augmentation_only 2x2 permutation 0",
        "resnet18 augmentation_only 2x2 permutation 1",
    ]
    assert saved["run_status"].tolist() == ["completed", "pending"]
    assert saved["ablation_name"].tolist() == ["augmentation_only", "augmentation_only"]
    assert saved["training_duration_seconds"].tolist()[0] == 2.5
    assert pd.isna(saved["training_duration_seconds"].tolist()[1])
