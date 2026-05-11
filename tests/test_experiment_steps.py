from types import SimpleNamespace

import pandas as pd

from src.preprocessing.tile_orders import TileOrderRecord
from src.training import experiment_steps
from src.training.experiment_steps import (
    collect_model_tile_order_results,
    collect_part2_ablation_results,
    get_executable_tile_order_records,
)


def test_executable_tile_order_records_skip_duplicate_one_by_one_output_tile_orders():
    records = [
        TileOrderRecord(grid_side_length=1, tile_order_id=0, tile_order_seed=42, output_tile_order=[0]),
        TileOrderRecord(grid_side_length=1, tile_order_id=1, tile_order_seed=42, output_tile_order=[0]),
        TileOrderRecord(grid_side_length=2, tile_order_id=0, tile_order_seed=42, output_tile_order=[0, 1, 2, 3]),
        TileOrderRecord(grid_side_length=2, tile_order_id=1, tile_order_seed=42, output_tile_order=[1, 0, 3, 2]),
    ]

    executable_records = get_executable_tile_order_records(records)

    assert [(record.grid_side_length, record.tile_order_id) for record in executable_records] == [
        (1, 0),
        (2, 0),
        (2, 1),
    ]


def test_collect_model_tile_order_results_records_training_duration_and_progress_desc(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
    )
    record = TileOrderRecord(
        grid_side_length=2,
        tile_order_id=3,
        tile_order_seed=99,
        output_tile_order=[0, 1, 2, 3],
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
                "grid_side_length": 1,
                "tile_count": 1,
                "tile_order_id": 0,
                "tile_order_seed": 99,
                "seed": 42,
                "best_val_accuracy": 0.5,
            }
        ]
    ).to_csv(raw_results_path, index=False)

    rows = collect_model_tile_order_results(
        config=config,
        model_name="resnet18",
        run_id="run",
        train_samples=[("cat.jpg", 0)],
        validation_samples=[("dog.jpg", 1)],
        tile_order_records=[record],
        seed=42,
        device="cpu",
        raw_results_output_path=str(raw_results_path),
    )

    assert rows[0]["training_duration_seconds"] == 2.5
    assert progress_descriptions == ["resnet18 2x2 order 3"]

    saved = pd.read_csv(raw_results_path)
    assert sorted(saved["model_name"]) == ["deit_tiny", "resnet18"]
    assert saved.loc[saved["model_name"] == "resnet18", "training_duration_seconds"].iloc[0] == 2.5
    assert saved.loc[saved["model_name"] == "resnet18", "run_status"].iloc[0] == "completed"


def test_collect_model_tile_order_results_leaves_pending_placeholders_after_crash(monkeypatch, tmp_path):
    config = SimpleNamespace(
        part="part1",
        config_name="test_config",
        image_size=32,
        batch_size=4,
        num_workers=0,
        pretrained=False,
    )
    records = [
        TileOrderRecord(
            grid_side_length=2,
            tile_order_id=0,
            tile_order_seed=99,
            output_tile_order=[0, 1, 2, 3],
        ),
        TileOrderRecord(
            grid_side_length=2,
            tile_order_id=1,
            tile_order_seed=100,
            output_tile_order=[1, 0, 3, 2],
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
        collect_model_tile_order_results(
            config=config,
            model_name="resnet18",
            run_id="run",
            train_samples=[("cat.jpg", 0)],
            validation_samples=[("dog.jpg", 1)],
            tile_order_records=records,
            seed=42,
            device="cpu",
            raw_results_output_path=str(raw_results_path),
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated crash"
    else:
        raise AssertionError("Expected simulated crash")

    saved = pd.read_csv(raw_results_path).sort_values("tile_order_id")

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
        TileOrderRecord(
            grid_side_length=2,
            tile_order_id=0,
            tile_order_seed=99,
            output_tile_order=[0, 1, 2, 3],
        ),
        TileOrderRecord(
            grid_side_length=2,
            tile_order_id=1,
            tile_order_seed=100,
            output_tile_order=[1, 0, 3, 2],
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
            tile_order_records=records,
            device="cpu",
            run_id="part2_run",
            raw_results_output_path=str(raw_results_path),
        )
    except RuntimeError as exc:
        assert str(exc) == "simulated crash"
    else:
        raise AssertionError("Expected simulated crash")

    saved = pd.read_csv(raw_results_path).sort_values("tile_order_id")

    assert progress_descriptions == [
        "resnet18 augmentation_only 2x2 order 0",
        "resnet18 augmentation_only 2x2 order 1",
    ]
    assert saved["run_status"].tolist() == ["completed", "pending"]
    assert saved["ablation_name"].tolist() == ["augmentation_only", "augmentation_only"]
    assert saved["training_duration_seconds"].tolist()[0] == 2.5
    assert pd.isna(saved["training_duration_seconds"].tolist()[1])
