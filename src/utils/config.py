"""Configuration loading and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import socket
from typing import Any, Dict, Literal, Mapping
import os

from src.utils.io import load_yaml


GROUPED_CONFIG_KEYS = {"general", "input_output", "data", "models", "experiment", "ablations"}


def normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert grouped notebook configs into the flat internal runner format.

    Args:
        config: Raw YAML configuration, either grouped or already flat.

    Returns:
        Flat dictionary consumed by the experiment runners.
    """

    if not GROUPED_CONFIG_KEYS.intersection(config.keys()):
        return dict(config)

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

    # Keep Part 2's single model name ergonomic in YAML while preserving the
    # internal key used by the runner.
    if "model_name" not in normalized and "model_names" in normalized:
        model_names = normalized["model_names"]
        if isinstance(model_names, list) and len(model_names) == 1:
            normalized["model_name"] = model_names[0]

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
    root_dir: str = "/Users/royrubin/Documents/GitHub/MLDS_Final_Project"
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
        default_factory=lambda: ["resnet50", "deit_small", "mlp_mixer"]
    )
    pretrained: bool = False

    # ConvMixer-specific configuration
    convmixer_dim: int = 128
    convmixer_depth: int = 4

    # Grid and permutation experiment configuration
    grid_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_permutations: int = 5

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

    def __post_init__(self) -> None:
        # Adjust paths if running on Google Colab
        if self._is_code_running_on_colab():
            print("Running on Google Colab, adjusting configs...")

            # Adjust paths
            # mount google drive
            try:
                from google.colab import drive  # type: ignore # this import is only available in Colab, so if it succeeds we're in Colab
                drive.mount('/content/drive')
            except ImportError:
                raise ImportError("Google Colab environment detected but google.colab module not found")

            # now that drive was mounted, update paths
            self.root_dir = "/content/drive/MyDrive/MLDS_Final_Project"

        else:
            self.update_configs_for_local_testing()

        # set_paths
        self._set_paths()

        # Validate dirs exist
        assert os.path.exists(self.data_dir), f"Data dir {self.data_dir} does not exist"
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    ## Code to move to utils
    def _is_code_running_on_colab(self) -> bool:
        if socket.gethostname() == 'MACs-MacBook-Pro.local':
            return False
        try:
            from google.colab import drive  # type: ignore # this import is only available in Colab, so if it succeeds we're in Colab
            self.using_google_colab = True
            return True
        except ImportError:
            return False

    def _set_paths(self) -> None:
        self.data_dir = os.path.join(self.root_dir, "data", "dogs-vs-cats", "train")
        self.outputs_dir = os.path.join(self.root_dir, "outputs")
        self.results_dir = os.path.join(self.outputs_dir, "results")
        self.figures_dir = os.path.join(self.outputs_dir, "figures")

    def update_configs_for_local_testing(self) -> None:
        """Update configs for local testing."""
        self.sample_data = True
        self.sample_limit = 32
        self.grid_sizes = [1, 2]
        self.num_permutations = 3
        self.epochs = 5
        self.plot_samples = True


@dataclass
class Part2ExperimentConfig(CVExperimentConfig):
    """Notebook-owned configuration for Part 2 ResNet50 improvement ablations."""

    part: str = "part2"
    config_name: str = "part2_improvement"
    model_names: list[str] = field(default_factory=lambda: ["resnet50"])
    grid_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_permutations: int = 3
    ablations: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "pretrained_feature_extractor",
                "use_pretrained": True,
                "use_standard_augmentation": False,
                "freeze_backbone": True,
            },
            {
                "name": "pretrained_finetune",
                "use_pretrained": True,
                "use_standard_augmentation": False,
                "freeze_backbone": False,
            },
            {
                "name": "augmentation_only",
                "use_pretrained": False,
                "use_standard_augmentation": True,
                "freeze_backbone": False,
            },
            {
                "name": "pretrained_augmented_finetune",
                "use_pretrained": True,
                "use_standard_augmentation": True,
                "freeze_backbone": False,
            },
        ]
    )

    @property
    def model_name(self) -> str:
        """Return the single model used by the Part 2 ablation notebook."""

        return self.model_names[0]
