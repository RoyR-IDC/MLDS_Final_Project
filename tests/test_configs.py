from dataclasses import fields
from pathlib import Path

from src.utils.config import CVExperimentConfig, load_experiment_config, normalize_config


def test_grouped_config_normalizes_to_runner_keys():
    raw_config = {
        "general": {"part": "part1", "seeds": [0]},
        "input_output": {"data_dir": "data/dogs-vs-cats/train", "results_dir": "outputs/results"},
        "data": {"batch_size": 8, "image_size": 224},
        "models": {"model_names": ["resnet18"], "num_classes": 2},
        "experiment": {"grid_sizes": [1, 2], "num_permutations": 1},
    }

    normalized_config = normalize_config(raw_config)

    assert normalized_config["part"] == "part1"
    assert normalized_config["data_dir"] == "data/dogs-vs-cats/train"
    assert normalized_config["batch_size"] == 8
    assert normalized_config["model_names"] == ["resnet18"]
    assert normalized_config["grid_sizes"] == [1, 2]


def test_official_configs_are_not_duplicated():
    config_names = {path.name for path in Path("configs").glob("*.yaml")}

    assert config_names == {
        "part1_baselines.yaml",
        "part2_improvement.yaml",
        "part3_difficulty.yaml",
    }


def test_part2_config_exposes_single_model_name():
    normalized_config = load_experiment_config("configs/part2_improvement.yaml")

    assert normalized_config["part"] == "part2"
    assert normalized_config["model_name"] == "resnet18"
    assert normalized_config["seed"] == 42
    assert "permutation_seed" not in normalized_config
    assert "seeds" not in normalized_config
    assert "ablations" in normalized_config


def test_cv_experiment_config_exposes_single_seed_field():
    field_names = {field.name for field in fields(CVExperimentConfig)}

    assert "seed" in field_names
    assert "permutation_seed" not in field_names
