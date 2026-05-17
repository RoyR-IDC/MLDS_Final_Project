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
