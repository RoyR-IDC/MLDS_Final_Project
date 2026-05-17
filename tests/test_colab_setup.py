import sys
from types import ModuleType

import src.utils.colab as colab
from src.utils.colab import (
    COLAB_PREINSTALLED_REQUIREMENT_PREFIXES,
    filter_colab_requirements_lines,
    find_project_root,
    prepare_project_imports,
    requirement_package_name,
)


def write_minimal_project(project_root):
    (project_root / "src" / "utils").mkdir(parents=True)
    (project_root / "src" / "evaluation").mkdir(parents=True)
    (project_root / "src" / "__init__.py").write_text("")
    (project_root / "src" / "utils" / "notebook_setup.py").write_text("")
    (project_root / "src" / "evaluation" / "experiment_results.py").write_text("")


def test_requirement_package_name_normalizes_common_requirement_forms():
    assert requirement_package_name("numpy==1.26.4") == "numpy"
    assert requirement_package_name("scikit_learn>=1.0") == "scikit-learn"
    assert requirement_package_name("  # comment") is None


def test_colab_requirements_filter_skips_binary_stack_and_keeps_lightweight_deps():
    requirements = [
        "kaggle",
        "python-dotenv",
        "numpy==1.26.4",
        "pandas==2.1.4",
        "scipy==1.11.4",
        "# PyTorch ecosystem for experiments",
        "torch>=1.12.0",
        "torchvision",
        "timm",
        "einops",
        "matplotlib",
        "scikit-learn",
        "pyyaml",
        "tqdm",
    ]

    filtered = filter_colab_requirements_lines(requirements)

    assert filtered == ["kaggle", "python-dotenv", "timm", "einops", "pyyaml", "tqdm"]
    for skipped_package in COLAB_PREINSTALLED_REQUIREMENT_PREFIXES:
        assert all(not line.lower().startswith(skipped_package) for line in filtered)


def test_colab_find_project_root_walks_up_from_notebook_directory(tmp_path):
    project_root = tmp_path / "MLDS_Final_Project"
    notebook_dir = project_root / "src" / "notebooks"
    notebook_dir.mkdir(parents=True)
    write_minimal_project(project_root)

    assert find_project_root(notebook_dir) == str(project_root)


def test_colab_find_project_root_checks_nearby_drive_style_locations(tmp_path, monkeypatch):
    project_root = tmp_path / "Colab Notebooks" / "MLDS_Final_Project"
    write_minimal_project(project_root)
    monkeypatch.setattr(colab, "COMMON_COLAB_PROJECT_ROOTS", (str(project_root),))
    monkeypatch.chdir(tmp_path)

    assert find_project_root(tmp_path / "unrelated") == str(project_root)


def test_colab_find_project_root_survives_disconnected_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "MLDS_Final_Project"
    write_minimal_project(project_root)
    monkeypatch.setattr(colab, "COMMON_COLAB_PROJECT_ROOTS", (str(project_root),))
    monkeypatch.setattr(colab.Path, "cwd", classmethod(lambda cls: (_ for _ in ()).throw(OSError(107, "Transport endpoint is not connected"))))

    assert find_project_root() == str(project_root)


def test_prepare_project_imports_adds_root_once_and_changes_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "MLDS_Final_Project"
    notebook_dir = project_root / "src" / "notebooks"
    notebook_dir.mkdir(parents=True)
    write_minimal_project(project_root)
    original_sys_path = list(sys.path)
    monkeypatch.chdir(tmp_path)

    try:
        first_root = prepare_project_imports(notebook_dir)
        second_root = prepare_project_imports(notebook_dir)

        assert first_root == project_root
        assert second_root == project_root
        assert sys.path.count(str(project_root)) == 1
    finally:
        sys.path[:] = original_sys_path


def test_prepare_project_imports_removes_stale_src_modules(tmp_path, monkeypatch):
    project_root = tmp_path / "MLDS_Final_Project"
    stale_root = tmp_path / "stale"
    write_minimal_project(project_root)
    (stale_root / "src").mkdir(parents=True)
    (stale_root / "src" / "__init__.py").write_text("")
    stale_src = ModuleType("src")
    stale_src.__file__ = str(stale_root / "src" / "__init__.py")
    stale_child = ModuleType("src.evaluation")
    stale_child.__file__ = str(stale_root / "src" / "evaluation" / "__init__.py")
    monkeypatch.setitem(sys.modules, "src", stale_src)
    monkeypatch.setitem(sys.modules, "src.evaluation", stale_child)

    prepare_project_imports(project_root)

    assert "src" not in sys.modules
    assert "src.evaluation" not in sys.modules


def test_prepare_project_imports_rejects_partial_project_copy(tmp_path):
    partial_root = tmp_path / "MLDS_Final_Project"
    (partial_root / "src" / "utils").mkdir(parents=True)
    (partial_root / "src" / "__init__.py").write_text("")
    (partial_root / "src" / "utils" / "notebook_setup.py").write_text("")

    try:
        prepare_project_imports(partial_root)
    except ModuleNotFoundError as exc:
        assert "src/evaluation/experiment_results.py" in str(exc)
    else:
        raise AssertionError("prepare_project_imports should reject a partial project copy")
