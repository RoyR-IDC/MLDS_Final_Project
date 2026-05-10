from dataclasses import fields
from pathlib import Path

from src.utils.config import CVExperimentConfig, Part2ExperimentConfig, find_project_root, normalize_config


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

    assert config_names == set()


def test_part2_config_defaults_to_resnet50_improvement_ablation_setup():
    dataclass_fields = Part2ExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()
    grid_sizes = dataclass_fields["grid_sizes"].default_factory()
    ablations = dataclass_fields["ablations"].default_factory()

    assert dataclass_fields["part"].default == "part2"
    assert dataclass_fields["config_name"].default == "part2_improvement"
    assert model_names == ["resnet50"]
    assert grid_sizes == [1, 2, 3, 4]
    assert dataclass_fields["num_permutations"].default == 3
    assert [ablation["name"] for ablation in ablations] == [
        "pretrained_feature_extractor",
        "pretrained_finetune",
        "augmentation_only",
        "pretrained_augmented_finetune",
    ]


def test_part2_config_exposes_single_model_name_property():
    config = Part2ExperimentConfig.__new__(Part2ExperimentConfig)
    config.model_names = ["resnet50"]

    assert config.model_name == "resnet50"


def test_cv_experiment_config_exposes_single_seed_field():
    field_names = {field.name for field in fields(CVExperimentConfig)}

    assert "seed" in field_names
    assert "permutation_seed" not in field_names


def test_find_project_root_walks_up_from_notebook_directory(tmp_path):
    project_root = tmp_path / "MLDS_Final_Project"
    notebook_dir = project_root / "src" / "notebooks"
    notebook_dir.mkdir(parents=True)
    (project_root / "requirements.txt").write_text("pytest\n")

    assert find_project_root(notebook_dir) == str(project_root)


def test_config_resolves_project_relative_paths():
    config = CVExperimentConfig.__new__(CVExperimentConfig)
    config.root_dir = "/project"
    resolved_path = CVExperimentConfig._resolve_project_path(config, "outputs/results", "unused")

    assert resolved_path == "/project/outputs/results"
