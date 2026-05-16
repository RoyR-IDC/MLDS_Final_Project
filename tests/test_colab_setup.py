import sys

from src.utils.colab import (
    COLAB_PREINSTALLED_REQUIREMENT_PREFIXES,
    filter_colab_requirements_lines,
    find_project_root,
    prepare_project_imports,
    requirement_package_name,
)


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
    (project_root / "requirements.txt").write_text("pytest\n")

    assert find_project_root(notebook_dir) == str(project_root)


def test_prepare_project_imports_adds_root_once_and_changes_cwd(tmp_path, monkeypatch):
    project_root = tmp_path / "MLDS_Final_Project"
    notebook_dir = project_root / "src" / "notebooks"
    notebook_dir.mkdir(parents=True)
    (project_root / "requirements.txt").write_text("pytest\n")
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
