from dataclasses import fields
from pathlib import Path

from src.utils.config import (
    CVExperimentConfig,
    Part2ExperimentConfig,
    Part3ExperimentConfig,
    find_project_root,
    normalize_config,
)


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


def test_part2_config_defaults_to_resnet18_improvement_ablation_setup():
    dataclass_fields = Part2ExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()
    grid_sizes = dataclass_fields["grid_sizes"].default_factory()
    ablations = dataclass_fields["ablations"].default_factory()

    assert dataclass_fields["part"].default == "part2"
    assert dataclass_fields["config_name"].default == "part2_improvement"
    assert model_names == ["resnet18"]
    assert grid_sizes == [1, 2, 3, 4]
    assert dataclass_fields["num_permutations"].default == 3
    assert [ablation["name"] for ablation in ablations] == [
        "augmentation_only",
        "finetune_only",
        "augmentation_finetune",
    ]


def test_part2_config_exposes_single_model_name_property():
    config = Part2ExperimentConfig.__new__(Part2ExperimentConfig)
    config.model_names = ["resnet18"]

    assert config.model_name == "resnet18"


def test_part3_config_defaults_to_resnet18_hardness_analysis_setup():
    dataclass_fields = Part3ExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()

    assert dataclass_fields["part"].default == "part3"
    assert dataclass_fields["config_name"].default == "part3_hardness_analysis"
    assert model_names == ["resnet18"]
    assert dataclass_fields["alpha_center"].default == 1.0
    assert dataclass_fields["weight_center"].default == 0.5
    assert dataclass_fields["weight_dist"].default == 0.5


def test_part3_config_exposes_single_model_name_property():
    config = Part3ExperimentConfig.__new__(Part3ExperimentConfig)
    config.model_names = ["resnet18"]

    assert config.model_name == "resnet18"


def test_cv_experiment_config_exposes_single_seed_field():
    field_names = {field.name for field in fields(CVExperimentConfig)}

    assert "seed" in field_names
    assert "permutation_seed" not in field_names


def test_part1_model_defaults_to_lightweight_pretrained_trio():
    dataclass_fields = CVExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()

    assert model_names == ["resnet18", "deit_tiny", "mlp_mixer_small"]
    assert dataclass_fields["pretrained"].default is True


def test_local_testing_defaults_use_larger_part1_signal():
    config = CVExperimentConfig.__new__(CVExperimentConfig)

    CVExperimentConfig.update_configs_for_local_testing(config)

    assert config.sample_data is True
    assert config.sample_limit == 256
    assert config.grid_sizes == [1, 3]
    assert config.num_permutations == 2
    assert config.epochs == 5
    assert config.plot_samples is True


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
