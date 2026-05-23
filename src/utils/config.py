"""Configuration loading and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping
import os

from src.models.registry import validate_model_name, validate_model_names
from src.utils.colab import (
    DEFAULT_COLAB_LOCAL_DATA_DIR,
    DEFAULT_COLAB_DRIVE_ROOT,
    colab_data_zip_path,
    find_project_root,
    is_google_colab_runtime,
    mount_colab_drive_if_available,
)
from src.utils.io import load_yaml


GROUPED_CONFIG_KEYS = {"general", "input_output", "data", "models", "experiment", "ablations"}
DEFAULT_LOCAL_ROOT = "/Users/royrubin/Documents/GitHub/MLDS_Final_Project"
PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES = [1, 4, 10]# [1, 2, 4, 6, 8, 10, 12]
COLAB_BATCH_SIZE = 64
COLAB_NUM_WORKERS = 4


def normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert grouped notebook configs into the flat internal runner format.

    Args:
        config: Raw YAML configuration, either grouped or already flat.

    Returns:
        Flat dictionary consumed by the experiment runners.
    """

    if not GROUPED_CONFIG_KEYS.intersection(config.keys()):
        normalized = dict(config)
        return _normalize_model_config(normalized)

    general = dict(config.get("general", {}))
    input_output = dict(config.get("input_output", {}))
    data = dict(config.get("data", {}))
    models = dict(config.get("models", {}))
    experiment = dict(config.get("experiment", {}))

    normalized: Dict[str, Any] = {}
    normalized.update(general)
    normalized.update(input_output)
    normalized.update(data)
    normalized.update(models)
    normalized.update(experiment)

    if "ablations" in config:
        normalized["ablations"] = config["ablations"]

    return _normalize_model_config(normalized)


def _normalize_model_config(normalized: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and canonicalize model fields in flat configs."""

    if "model_names" in normalized:
        normalized["model_names"] = validate_model_names(normalized["model_names"])
    if "model_name" in normalized:
        normalized["model_name"] = validate_model_name(normalized["model_name"])

    # Keep single-model notebooks ergonomic in YAML while preserving the
    # internal key used by the runner.
    if "model_name" not in normalized and "model_names" in normalized:
        model_names = normalized["model_names"]
        if isinstance(model_names, list) and len(model_names) == 1:
            normalized["model_name"] = model_names[0]

    if "model_name" in normalized and "model_names" in normalized:
        model_name = normalized["model_name"]
        model_names = normalized["model_names"]
        if model_name not in model_names:
            raise ValueError(
                f"model_name='{model_name}' must be one of model_names={model_names}"
            )

    if normalized.get("part") in {"part2", "part3"} and "model_names" in normalized:
        model_names = normalized["model_names"]
        if model_names != ["resnet18"]:
            raise ValueError(
                f"{normalized['part']} configs support only model_names=['resnet18']; "
                f"got model_names={model_names}"
            )
    if normalized.get("part") in {"part2", "part3"} and "model_name" in normalized:
        model_name = normalized["model_name"]
        if model_name != "resnet18":
            raise ValueError(
                f"{normalized['part']} configs support only model_name='resnet18'; "
                f"got model_name='{model_name}'"
            )

    return normalized


def load_experiment_config(path: str) -> Dict[str, Any]:
    """Load a YAML experiment config and normalize it for internal runners."""

    return normalize_config(load_yaml(path))


@dataclass
class CVExperimentConfig:
    # General experiment configuration
    part: str = "part1"
    config_name: str = "part1"
    device: str = "auto"
    deterministic: bool = False

    # Directory configuration
    root_dir: str = DEFAULT_LOCAL_ROOT
    data_dir: str = ""
    outputs_dir: str = ""
    results_dir: str = ""
    figures_dir: str = ""

    # Dataset splitting configuration
    sample_data: bool = False
    sample_limit: int = 256
    val_fraction: float = 0.2
    test_fraction: float = 0.0

    # Image preprocessing configuration
    image_size: int = 224

    # Data loading configuration
    batch_size: int = 16
    num_workers: int = 2

    # Model configuration
    num_classes: int = 2
    model_names: list[str] = field(
        default_factory=lambda: ["resnet18", "deit_tiny", "mlp_mixer_small"]
    )
    pretrained: bool = True
    freeze_backbone: bool = True

    # Tile permutation experiment configuration
    tiles_per_side_values: list[int] = field(default_factory=lambda: PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES.copy())
    num_tile_permutations: int = 5

    # Training configuration
    epochs: int = 10
    learning_rate: float = 0.0003
    optimizer: Literal["adamw"] = "adamw"
    weight_decay: float = 0.0001
    use_amp: bool = False

    # reproduction configuration
    seed: int = 42
    max_threads: int = 4

    # environment configuration
    using_google_colab: bool = False
    plot_samples: bool = False
    stage_colab_data_to_local_disk: bool = True
    colab_local_data_dir: str = DEFAULT_COLAB_LOCAL_DATA_DIR
    profile_performance: bool = False
    profile_warmup_batches: int = 0
    profile_output_dir: str = ""

    def __post_init__(self) -> None:
        self._validate_config_model_names()

        if self._is_code_running_on_colab():
            print("Running on Google Colab, adjusting configs...")
            self._mount_colab_drive_if_available()
            self._update_root_for_colab()
            self.update_configs_for_colab_runtime()
        else:
            self._update_root_for_local_runtime()
            self.update_configs_for_local_testing()

        self._validate_config_model_names()

        # set_paths
        self._set_paths()

        # Validate data input exists. Colab may stage from a sibling train.zip
        # even when the extracted Drive train directory is not present.
        if not self._data_input_exists():
            raise FileNotFoundError(
                "Dataset directory does not exist: "
                f"{self.data_dir}. Expected Kaggle Dogs vs Cats images under "
                "<project-root>/data/dogs-vs-cats/train or a Colab staging ZIP at "
                "<project-root>/data/dogs-vs-cats/train.zip. In Colab, upload or "
                "mount the project/data folder before starting the training cells."
            )
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    ## Code to move to utils
    def _is_code_running_on_colab(self) -> bool:
        self.using_google_colab = is_google_colab_runtime()
        return self.using_google_colab

    def _mount_colab_drive_if_available(self) -> None:
        """Mount Google Drive in Colab when the Drive API is available."""

        mount_colab_drive_if_available()

    def _update_root_for_colab(self) -> None:
        """Resolve the project root for either Drive-backed or cloned Colab runs."""

        if self.root_dir and self.root_dir != DEFAULT_LOCAL_ROOT:
            return
        drive_root = Path(DEFAULT_COLAB_DRIVE_ROOT)
        if drive_root.exists():
            self.root_dir = str(drive_root)
            return
        self.root_dir = find_project_root()

    def _update_root_for_local_runtime(self) -> None:
        """Avoid carrying the developer machine path into other local runtimes."""

        if self.root_dir == DEFAULT_LOCAL_ROOT and not Path(self.root_dir).exists():
            self.root_dir = find_project_root()

    def _resolve_project_path(self, value: str, default_relative: str) -> str:
        """Resolve absolute and project-relative config paths."""

        path_value = value or default_relative
        path = Path(path_value)
        if path.is_absolute():
            return str(path)
        return str(Path(self.root_dir) / path)

    def _set_paths(self) -> None:
        self.data_dir = self._resolve_project_path(self.data_dir, os.path.join("data", "dogs-vs-cats", "train"))
        self.outputs_dir = self._resolve_project_path(self.outputs_dir, "outputs")
        self.results_dir = self._resolve_project_path(self.results_dir, os.path.join("outputs", "results"))
        self.figures_dir = self._resolve_project_path(self.figures_dir, os.path.join("outputs", "figures"))
        self.profile_output_dir = self._resolve_project_path(self.profile_output_dir, self.results_dir)

    def _data_input_exists(self) -> bool:
        """Return whether configured image data or its Colab staging ZIP exists."""

        if os.path.exists(self.data_dir):
            return True
        if self.using_google_colab and self.stage_colab_data_to_local_disk:
            return colab_data_zip_path(self.data_dir).exists()
        return False

    def update_configs_for_local_testing(self) -> None:
        """Update configs for local testing."""
        self.sample_data = True
        self.sample_limit = 256
        self.model_names = ["resnet18"]
        if not getattr(self, "tiles_per_side_values", None):
            self.tiles_per_side_values = PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES.copy()
        self.num_tile_permutations = 2
        self.epochs = 1
        self.plot_samples = True

    def update_configs_for_colab_runtime(self) -> None:
        """Tune default runtime settings for Colab GPU sessions."""

        self.batch_size = max(int(self.batch_size), COLAB_BATCH_SIZE)
        self.num_workers = max(int(self.num_workers), COLAB_NUM_WORKERS)
        self.use_amp = True

    def _validate_config_model_names(self) -> None:
        """Validate and canonicalize the models requested by this config."""

        self.model_names = validate_model_names(self.model_names)


@dataclass
class Part2ExperimentConfig(CVExperimentConfig):
    """Notebook-owned configuration for Part 2 ResNet-18 improvement ablations."""

    part: str = "part2"
    config_name: str = "part2_improvement"
    model_names: list[str] = field(default_factory=lambda: ["resnet18"])
    tiles_per_side_values: list[int] = field(default_factory=lambda: PART1_PART2_REMOTE_TILES_PER_SIDE_VALUES.copy())
    num_tile_permutations: int = 3
    ablations: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "augmentation_patch_shuffle",
                "use_pretrained": True,
                "augmentation": "patch_shuffle",
            },
            {
                "name": "augmentation_random_erasing",
                "use_pretrained": True,
                "augmentation": "random_erasing",
            },
            {
                "name": "augmentation_same_label_cutmix",
                "use_pretrained": True,
                "augmentation": "same_label_cutmix",
            },
            {
                "name": "loss_focal_loss",
                "use_pretrained": True,
                "augmentation": "none",
                "loss": "focal_loss",
                "focal_gamma": 2.0,
                "focal_alpha": 1.0,
            },
            {
                "name": "curriculum_corruption_probability",
                "use_pretrained": True,
                "augmentation": "none",
                "curriculum": "corruption_probability",
            },
            {
                "name": "curriculum_permutation_difficulty",
                "use_pretrained": True,
                "augmentation": "none",
                "curriculum": "permutation_difficulty",
            },
        ]
    )

    @property
    def model_name(self) -> str:
        """Return the single model used by the Part 2 ablation notebook."""

        return self._validate_resnet18_only_model_names()[0]

    def _validate_config_model_names(self) -> None:
        """Part 2 is a ResNet-18-only ablation setup."""

        self.model_names = self._validate_resnet18_only_model_names()

    def _validate_resnet18_only_model_names(self) -> list[str]:
        model_names = validate_model_names(self.model_names)
        if model_names != ["resnet18"]:
            config_name = getattr(self, "config_name", type(self).__name__)
            raise ValueError(
                f"{config_name} supports only model_names=['resnet18']; "
                f"got model_names={model_names}"
            )
        return model_names

    def __post_init__(self) -> None:
        return super().__post_init__()


@dataclass
class Part3ExperimentConfig(CVExperimentConfig):
    """Notebook-owned configuration for Part 3 ResNet-18 hardness analysis."""

    part: str = "part3"
    config_name: str = "part3_hardness_analysis"
    model_names: list[str] = field(default_factory=lambda: ["resnet18"])
    tiles_per_side_values: list[int] = field(default_factory=lambda: [1, 3, 4])
    weight_adj: float = 0.5
    weight_entropy: float = 0.3
    weight_dist: float = 0.2

    def update_configs_for_local_testing(self) -> None:
        """Keep Part 3 local smoke runs aligned with the hardness notebook scope."""

        super().update_configs_for_local_testing()
        self.tiles_per_side_values = [1, 3]

    @property
    def model_name(self) -> str:
        """Return the single model used by the Part 3 hardness notebook."""

        return self._validate_resnet18_only_model_names()[0]

    def _validate_config_model_names(self) -> None:
        """Part 3 is a ResNet-18-only hardness analysis setup."""

        self.model_names = self._validate_resnet18_only_model_names()

    def _validate_resnet18_only_model_names(self) -> list[str]:
        model_names = validate_model_names(self.model_names)
        if model_names != ["resnet18"]:
            config_name = getattr(self, "config_name", type(self).__name__)
            raise ValueError(
                f"{config_name} supports only model_names=['resnet18']; "
                f"got model_names={model_names}"
            )
        return model_names
