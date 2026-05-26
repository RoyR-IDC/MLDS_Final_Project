from dataclasses import fields
from pathlib import Path

from src.utils.config import (
    COLAB_BATCH_SIZE,
    COLAB_NUM_WORKERS,
    CVExperimentConfig,
    PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES,
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
        "experiment": {"tiles_per_side_values": [1, 2], "num_tile_permutations": 1},
    }

    normalized_config = normalize_config(raw_config)

    assert normalized_config["part"] == "part1"
    assert normalized_config["data_dir"] == "data/dogs-vs-cats/train"
    assert normalized_config["batch_size"] == 8
    assert normalized_config["model_names"] == ["resnet18"]
    assert normalized_config["model_name"] == "resnet18"
    assert normalized_config["tiles_per_side_values"] == [1, 2]


def test_config_normalization_rejects_models_outside_supported_trio():
    raw_config = {
        "part": "part1",
        "model_names": ["resnet18", "resnet50"],
    }

    try:
        normalize_config(raw_config)
    except ValueError as exc:
        assert "Unsupported model_name='resnet50'" in str(exc)
        assert "resnet18, deit_tiny, mlp_mixer_base, mlp_mixer_small" in str(exc)
    else:
        raise AssertionError("Expected unsupported model_name to raise")


def test_config_normalization_rejects_inconsistent_single_model_fields():
    raw_config = {
        "part": "part1",
        "model_name": "deit_tiny",
        "model_names": ["resnet18"],
    }

    try:
        normalize_config(raw_config)
    except ValueError as exc:
        assert "must be one of model_names" in str(exc)
    else:
        raise AssertionError("Expected inconsistent model fields to raise")


def test_part3_normalized_config_is_resnet18_only():
    raw_config = {
        "part": "part3",
        "model_names": ["deit_tiny"],
    }

    try:
        normalize_config(raw_config)
    except ValueError as exc:
        assert "part3 configs support only model_names=['resnet18']" in str(exc)
    else:
        raise AssertionError("Expected Part 3 config to reject non-resnet18 model")


def test_part3_normalized_config_rejects_single_resnet50_model_name():
    raw_config = {
        "part": "part3",
        "model_name": "resnet50",
    }

    try:
        normalize_config(raw_config)
    except ValueError as exc:
        assert "Unsupported model_name='resnet50'" in str(exc)
    else:
        raise AssertionError("Expected Part 3 config to reject resnet50")


def test_official_configs_are_not_duplicated():
    config_names = {path.name for path in Path("configs").glob("*.yaml")}

    assert config_names == set()


def test_part2_config_defaults_to_resnet18_improvement_ablation_setup():
    dataclass_fields = Part2ExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()
    tiles_per_side_values = dataclass_fields["tiles_per_side_values"].default_factory()
    ablations = dataclass_fields["ablations"].default_factory()

    assert dataclass_fields["part"].default == "part2"
    assert dataclass_fields["config_name"].default == "part2_improvement"
    assert model_names == ["resnet18"]
    assert tiles_per_side_values == PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES
    assert dataclass_fields["num_tile_permutations"].default == 3
    assert [ablation["name"] for ablation in ablations] == [
        "augmentation_patch_shuffle",
        "regular_augmentations",
        "mixed_original_permuted",
        "resnet18_mlp_head",
        "curriculum_corruption_probability",
        "curriculum_permutation_difficulty",
    ]
    assert ablations[1]["augmentation"] == "regular_augmentations"
    assert ablations[2]["p_original"] == 0.5
    assert ablations[3]["classification_head"] == "mlp"
    assert ablations[4]["epochs"] == 30
    assert ablations[5]["epochs"] == 30


def test_part2_config_exposes_single_model_name_property():
    config = Part2ExperimentConfig.__new__(Part2ExperimentConfig)
    config.model_names = ["resnet18"]

    assert config.model_name == "resnet18"


def test_part2_config_rejects_non_resnet18_supported_model():
    config = Part2ExperimentConfig.__new__(Part2ExperimentConfig)
    config.config_name = "part2_improvement"
    config.model_names = ["deit_tiny"]

    try:
        _ = config.model_name
    except ValueError as exc:
        assert "supports only model_names=['resnet18']" in str(exc)
    else:
        raise AssertionError("Expected Part 2 config to reject non-resnet18 model")


def test_part3_config_defaults_to_resnet18_hardness_analysis_setup():
    dataclass_fields = Part3ExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()
    tiles_per_side_values = dataclass_fields["tiles_per_side_values"].default_factory()

    assert dataclass_fields["part"].default == "part3"
    assert dataclass_fields["config_name"].default == "part3_hardness_analysis"
    assert model_names == ["resnet18"]
    assert tiles_per_side_values == PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES
    assert dataclass_fields["num_tile_permutations"].default == 3
    assert dataclass_fields["weight_adj"].default == 0.5
    assert dataclass_fields["weight_entropy"].default == 0.3
    assert dataclass_fields["weight_dist"].default == 0.2


def test_part3_config_exposes_single_model_name_property():
    config = Part3ExperimentConfig.__new__(Part3ExperimentConfig)
    config.model_names = ["resnet18"]

    assert config.model_name == "resnet18"


def test_part3_config_rejects_resnet50():
    config = Part3ExperimentConfig.__new__(Part3ExperimentConfig)
    config.config_name = "part3_hardness_analysis"
    config.model_names = ["resnet50"]

    try:
        _ = config.model_name
    except ValueError as exc:
        assert "Unsupported model_name='resnet50'" in str(exc)
    else:
        raise AssertionError("Expected Part 3 config to reject resnet50")


def test_cv_experiment_config_exposes_single_seed_field():
    field_names = {field.name for field in fields(CVExperimentConfig)}

    assert "seed" in field_names
    assert "tile_permutation_seed" not in field_names


def test_part1_model_defaults_to_lightweight_pretrained_trio():
    dataclass_fields = CVExperimentConfig.__dataclass_fields__
    model_names = dataclass_fields["model_names"].default_factory()
    tiles_per_side_values = dataclass_fields["tiles_per_side_values"].default_factory()

    assert model_names == ["resnet18", "deit_tiny", "mlp_mixer_base"]
    assert tiles_per_side_values == PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES
    assert dataclass_fields["num_tile_permutations"].default == 3
    assert dataclass_fields["pretrained"].default is True
    assert dataclass_fields["freeze_backbone"].default is True


def test_local_testing_preserves_configured_tile_scope():
    config = CVExperimentConfig.__new__(CVExperimentConfig)
    config.tiles_per_side_values = [1, 4, 7, 10]
    config.num_tile_permutations = 3

    CVExperimentConfig.update_configs_for_local_testing(config)

    assert config.sample_data is True
    assert config.sample_limit == 256
    assert config.tiles_per_side_values == [1, 4, 7, 10]
    assert config.num_tile_permutations == 3
    assert config.epochs == 1
    assert config.plot_samples is True


def test_local_testing_falls_back_to_remote_tile_scope_when_missing():
    config = CVExperimentConfig.__new__(CVExperimentConfig)

    CVExperimentConfig.update_configs_for_local_testing(config)

    assert config.tiles_per_side_values == PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES


def test_colab_runtime_defaults_enable_t4_speed_settings():
    config = CVExperimentConfig.__new__(CVExperimentConfig)
    config.batch_size = 16
    config.num_workers = 0
    config.use_amp = False

    CVExperimentConfig.update_configs_for_colab_runtime(config)

    assert config.batch_size == COLAB_BATCH_SIZE
    assert config.num_workers == COLAB_NUM_WORKERS
    assert config.use_amp is True


def test_part3_local_testing_keeps_hardness_notebook_tile_scope():
    config = Part3ExperimentConfig.__new__(Part3ExperimentConfig)

    Part3ExperimentConfig.update_configs_for_local_testing(config)

    assert config.tiles_per_side_values == PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES


def test_find_project_root_walks_up_from_notebook_directory(tmp_path):
    project_root = tmp_path / "MLDS_Final_Project"
    notebook_dir = project_root / "src" / "notebooks"
    notebook_dir.mkdir(parents=True)
    (project_root / "src" / "utils").mkdir(parents=True)
    (project_root / "src" / "evaluation").mkdir(parents=True)
    (project_root / "src" / "__init__.py").write_text("")
    (project_root / "src" / "utils" / "notebook_setup.py").write_text("")
    (project_root / "src" / "evaluation" / "experiment_results.py").write_text("")

    assert find_project_root(notebook_dir) == str(project_root)


def test_config_resolves_project_relative_paths():
    config = CVExperimentConfig.__new__(CVExperimentConfig)
    config.root_dir = "/project"
    resolved_path = CVExperimentConfig._resolve_project_path(config, "outputs/results", "unused")

    assert resolved_path == "/project/outputs/results"


def test_colab_config_accepts_zip_only_dataset_input(tmp_path):
    config = CVExperimentConfig.__new__(CVExperimentConfig)
    config.data_dir = str(tmp_path / "data" / "dogs-vs-cats" / "train")
    config.using_google_colab = True
    config.stage_colab_data_to_local_disk = True
    zip_path = Path(f"{config.data_dir}.zip")
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(b"zip")

    assert CVExperimentConfig._data_input_exists(config) is True
