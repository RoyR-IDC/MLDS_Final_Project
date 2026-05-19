"""Notebook setup helpers for the MLDS final project."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys


def ensure_project_on_path(project_root: str | Path | None = None) -> Path:
    """Resolve the project root and add it to ``sys.path`` once."""

    if project_root is None:
        root = Path(__file__).resolve().parents[2]
    else:
        root = Path(project_root).resolve()

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return root


def bootstrap_notebook(
    project_root: str | Path | None = None,
    *,
    install_requirements: bool = True,
    print_diagnostics: bool = True,
) -> Path:
    """Prepare notebook imports, optional Colab deps, and runtime diagnostics."""

    root = ensure_project_on_path(project_root)
    from src.utils.colab import bootstrap_notebook_runtime

    return bootstrap_notebook_runtime(
        root,
        install_requirements=install_requirements,
        print_diagnostics=print_diagnostics,
    )


def part1_output_paths(config: Any) -> dict[str, str]:
    """Build stable Part 1 output paths for notebook display."""

    from src.evaluation.experiment_results import experiment_output_paths

    paths = experiment_output_paths(
        results_dir=config.results_dir,
        figures_dir=config.figures_dir,
        part_name="part1",
    )
    paths["accuracy_plot"] = paths["figure"]
    return paths


def setup_part1_config() -> SimpleNamespace:
    """Create Part 1 config and runtime objects used by the notebook."""

    from src.evaluation.experiment_results import get_device
    from src.utils.colab import (
        print_colab_runtime_diagnostics,
        warn_if_colab_runtime_without_cuda,
    )
    from src.utils.config import CVExperimentConfig

    config = CVExperimentConfig()
    device = get_device(config=config)
    warn_if_colab_runtime_without_cuda(device)
    print_colab_runtime_diagnostics(device)
    return SimpleNamespace(config=config, configs=config, device=device, output_paths=part1_output_paths(config))


def setup_part2_config() -> SimpleNamespace:
    """Create Part 2 config, runtime objects, and a compact summary table."""

    import pandas as pd

    from src.evaluation.experiment_results import experiment_output_paths, get_device
    from src.utils.colab import (
        print_colab_runtime_diagnostics,
        warn_if_colab_runtime_without_cuda,
    )
    from src.utils.config import Part2ExperimentConfig
    from src.utils.reproducibility import seed_everything

    config = Part2ExperimentConfig()
    device = get_device(config)
    warn_if_colab_runtime_without_cuda(device)
    print_colab_runtime_diagnostics(device)
    output_paths = experiment_output_paths(config.results_dir, config.figures_dir, config.part)
    seed_everything(config.seed, deterministic=config.deterministic)
    summary = pd.DataFrame(
        [
            {
                "part": config.part,
                "config_name": config.config_name,
                "model_name": config.model_name,
                "tiles_per_side_values": config.tiles_per_side_values,
                "num_tile_permutations": config.num_tile_permutations,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "use_amp": config.use_amp,
                "stage_colab_data_to_local_disk": config.stage_colab_data_to_local_disk,
                "colab_local_data_dir": config.colab_local_data_dir,
                "device": str(device),
            }
        ]
    )
    return SimpleNamespace(config=config, device=device, output_paths=output_paths, summary=summary)


def setup_part3_config() -> SimpleNamespace:
    """Create Part 3 config, paths, and a compact summary table."""

    import pandas as pd

    from src.evaluation.experiment_results import part3_output_paths
    from src.utils.config import Part3ExperimentConfig

    config = Part3ExperimentConfig()
    results_dir = Path(config.results_dir)
    figures_dir = Path(config.figures_dir)
    output_paths = part3_output_paths(str(results_dir), str(figures_dir))
    summary = pd.DataFrame(
        [
            {
                "part": config.part,
                "config_name": config.config_name,
                "model_name": config.model_name,
                "tiles_per_side_values": config.tiles_per_side_values,
                "num_tile_permutations": config.num_tile_permutations,
                "seed": config.seed,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "use_amp": config.use_amp,
                "stage_colab_data_to_local_disk": config.stage_colab_data_to_local_disk,
                "colab_local_data_dir": config.colab_local_data_dir,
                "weight_adj": config.weight_adj,
                "weight_entropy": config.weight_entropy,
                "weight_dist": config.weight_dist,
            }
        ]
    )
    return SimpleNamespace(
        config=config,
        results_dir=results_dir,
        figures_dir=figures_dir,
        part1_results_csv=results_dir / "part1_raw_results.csv",
        tile_permutation_csv=results_dir / "part1_tile_permutations.csv",
        output_paths=output_paths,
        summary=summary,
    )
