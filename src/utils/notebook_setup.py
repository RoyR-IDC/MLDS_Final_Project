"""Notebook setup helpers for the MLDS final project."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
import shutil
import subprocess
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
    from src.preprocessing.tile_permutations import TILE_PERMUTATION_NAMES
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
                "tile_permutation_names": list(TILE_PERMUTATION_NAMES[: config.num_tile_permutations]),
                "grid_x_permutation_runs": len(config.tiles_per_side_values) * config.num_tile_permutations,
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
    from src.preprocessing.tile_permutations import TILE_PERMUTATION_NAMES
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
                "tile_permutation_names": list(TILE_PERMUTATION_NAMES[: config.num_tile_permutations]),
                "grid_x_permutation_records": len(config.tiles_per_side_values) * config.num_tile_permutations,
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


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _ensure_colab_pdf_export_dependencies() -> None:
    """Install the LaTeX pieces nbconvert needs in a fresh Colab runtime."""

    from src.utils.colab import is_google_colab_runtime

    if not is_google_colab_runtime():
        return

    try:
        import nbconvert  # noqa: F401
    except ImportError:
        print("Installing nbconvert for PDF export...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nbconvert"])

    missing_latex_command = not _command_exists("xelatex")
    missing_pandoc = not _command_exists("pandoc")
    missing_nbconvert_tex_packages = subprocess.run(
        ["kpsewhich", "adjustbox.sty"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0

    if not (missing_latex_command or missing_pandoc or missing_nbconvert_tex_packages):
        return

    if not _command_exists("apt-get"):
        print(
            "PDF export dependencies are missing, and apt-get is unavailable. "
            "Install xelatex, pandoc, and the LaTeX packages required by nbconvert."
        )
        return

    print("Installing Colab PDF export dependencies. This can take a few minutes...")
    subprocess.check_call(["apt-get", "update", "-qq"])
    subprocess.check_call(
        [
            "apt-get",
            "install",
            "-y",
            "-qq",
            "pandoc",
            "texlive-xetex",
            "texlive-latex-extra",
            "texlive-fonts-recommended",
            "texlive-plain-generic",
        ]
    )


def export_notebook_to_pdf(
    notebook_path: str | Path,
    export_dir: str | Path | None = None,
    *,
    install_colab_dependencies: bool = True,
) -> Path | None:
    """Export a saved notebook to PDF, with Colab-friendly dependency setup."""

    notebook_path = Path(notebook_path).resolve()
    if export_dir is None:
        export_dir = notebook_path.parent
    export_dir = Path(export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    if not notebook_path.exists():
        print(f"PDF export failed. Notebook file was not found: {notebook_path}")
        return None

    if install_colab_dependencies:
        _ensure_colab_pdf_export_dependencies()

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "pdf",
        str(notebook_path),
        "--output-dir",
        str(export_dir),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        pdf_path = export_dir / notebook_path.with_suffix(".pdf").name
        print(f"Generated PDF: {pdf_path}")
        return pdf_path

    print("PDF export failed.")
    print(f"Command exited with status {result.returncode}.")
    output_lines = result.stdout.splitlines()
    if output_lines:
        print("Last nbconvert output lines:")
        for line in output_lines[-25:]:
            print(line)
    else:
        print("nbconvert produced no output.")
    return None
