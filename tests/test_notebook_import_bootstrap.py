import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "src/notebooks/part1_solution.ipynb",
    ROOT / "src/notebooks/part2_solution.ipynb",
    ROOT / "src/notebooks/part3_solution.ipynb",
)


def _code_cells(notebook_path):
    notebook = json.loads(notebook_path.read_text())
    return ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def test_part_notebooks_bootstrap_full_project_before_src_imports():
    for notebook_path in NOTEBOOKS:
        cells = _code_cells(notebook_path)
        setup_index = next(
            index
            for index, source in enumerate(cells)
            if "ROOT = _colab_bootstrap.bootstrap_notebook_runtime(_PROJECT_ROOT" in source
        )
        setup_source = cells[setup_index]

        assert "src/__init__.py" in setup_source
        assert "src/utils/notebook_setup.py" in setup_source
        assert "src/evaluation/experiment_results.py" in setup_source
        if notebook_path.name in {"part1_solution.ipynb", "part2_solution.ipynb"}:
            assert "src/experiments/enhanced_confidence.py" in setup_source
        assert "force_remount=True" in setup_source
        assert "def _safe_cwd()" in setup_source
        assert "def _safe_exists(path)" in setup_source
        assert "except OSError:" in setup_source
        assert "spec_from_file_location('_mlds_colab_bootstrap'" in setup_source
        assert "bootstrap_notebook_runtime(_PROJECT_ROOT" in setup_source
        assert "notebook_setup = importlib.import_module('src.utils.notebook_setup')" in setup_source
        assert "ROOT = notebook_setup.bootstrap_notebook(_PROJECT_ROOT)" not in setup_source

        earlier_sources = "\n".join(cells[:setup_index])
        assert "import src." not in earlier_sources
        assert "from src." not in earlier_sources


def test_part1_notebook_keeps_easy_medium_hard_sample_records():
    cells = _code_cells(ROOT / "src/notebooks/part1_solution.ipynb")
    config_source = next(
        source
        for source in cells
        if "part1_setup = notebook_setup.setup_part1_config()" in source
    )
    sample_source = next(source for source in cells if "plot_tile_permutation_samples(" in source)

    assert "configs.num_tile_permutations = 3" in config_source
    assert "num_tile_permutations=configs.num_tile_permutations" in sample_source
    assert "max_records=sum(1 for record in tile_permutation_records if record.tile_permutation is not None)" in sample_source


def test_part1_training_loop_saves_without_inline_image_display():
    cells = _code_cells(ROOT / "src/notebooks/part1_solution.ipynb")
    training_source = next(
        source
        for source in cells
        if "train_model_on_tile_permutation_records(" in source
        and "for model_name in configs.model_names" in source
    )
    display_source = next(
        source
        for source in cells
        if "part1_generated_figure_paths" in source
        and "display(Image(filename=figure_path))" in source
    )

    assert "display(Image(filename=intermediate_figure_path))" not in training_source
    assert "Saved intermediate model plot" in training_source
    assert "part1_final_figure_paths" in display_source
    assert "part1_intermediate_experiment_figure_paths" in display_source


def test_part1_plotting_filters_to_configured_models_and_conditions():
    cells = _code_cells(ROOT / "src/notebooks/part1_solution.ipynb")
    plotting_source = next(
        source
        for source in cells
        if "PART1_EXPERIMENT_CONDITIONS" in source
        and "part1_final_figure_paths" in source
    )

    assert "part1_all_raw['model_name'].isin(configs.model_names)" in plotting_source
    assert "experiment_condition" in plotting_source
    assert "frozen_pretrained_binary_head" in plotting_source
    assert "model_column='experiment_condition'" in plotting_source
