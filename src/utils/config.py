"""Configuration loading and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    config_name: str = "part1_baselines"
    device: str = "auto"
    deterministic: bool = False

    # Directory configuration
    data_dir: str = "/Users/royrubin/Documents/GitHub/MLDS_Final_Project/data/dogs-vs-cats/train"
    outputs_dir: str = "/Users/royrubin/Documents/GitHub/MLDS_Final_Project/outputs"
    results_dir: str = "/Users/royrubin/Documents/GitHub/MLDS_Final_Project/outputs/results"
    figures_dir: str = "/Users/royrubin/Documents/GitHub/MLDS_Final_Project/outputs/figures"

    # Dataset splitting configuration
    sample_data: bool = True
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
        default_factory=lambda: ["resnet18", "swin_t", "convmixer"]
    )
    pretrained: bool = False

    # ConvMixer-specific configuration
    convmixer_dim: int = 128
    convmixer_depth: int = 4

    # Grid and permutation experiment configuration
    grid_sizes: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    num_permutations: int = 2
    permutation_seed: int = 42

    # Training configuration
    epochs: int = 1
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

    def __post_init__(self):
        # Adjust paths if running on Google Colab
        if self._is_code_running_on_colab():
            print("Running on Google Colab, adjusting configs...")
            
            # Adjust paths 
            # mount google drive
            try:
                from google.colab import drive
                drive.mount('/content/drive')
            except ImportError:
                raise ImportError("Google Colab environment detected but google.colab module not found")

            # now that drive was mounted, update paths
            self.data_dir = "/content/drive/MyDrive/MLDS_Final_Project/data"      
            self.outputs_dir = "/content/drive/MyDrive/MLDS_Final_Project/outputs"
            self.results_dir = os.path.join(self.outputs_dir, "results")
            self.figures_dir = os.path.join(self.outputs_dir, "figures")

            # Set sample_data to False to speed up development on Colab
            self.sample_data = False

        # Validate dirs exist
        assert os.path.exists(self.data_dir), f"Data dir {self.data_dir} does not exist"
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    ## Code to move to utils
    def _is_code_running_on_colab(self) -> bool:
        try:
            from google.colab import drive
            self.using_google_colab = True
            return True
        except ImportError:
            return False

