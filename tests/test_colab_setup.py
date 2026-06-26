import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace
import zipfile

import src.utils.notebook_setup as notebook_setup
import src.utils.colab as colab
from src.utils.colab import (
    COLAB_PREINSTALLED_REQUIREMENT_PREFIXES,
    colab_data_zip_path,
    filter_colab_requirements_lines,
    find_project_root,
    mount_colab_drive_if_available,
    prepare_project_imports,
    requirement_package_name,
    stage_colab_data_to_local_disk,
)


def write_flat_image_zip(zip_path, filenames):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        for filename in filenames:
            archive.writestr(filename, filename.encode())


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
    original_sys_path = list(sys.path)
    (stale_root / "src").mkdir(parents=True)
    (stale_root / "src" / "__init__.py").write_text("")
    stale_src = ModuleType("src")
    stale_src.__file__ = str(stale_root / "src" / "__init__.py")
    stale_child = ModuleType("src.evaluation")
    stale_child.__file__ = str(stale_root / "src" / "evaluation" / "__init__.py")
    monkeypatch.setitem(sys.modules, "src", stale_src)
    monkeypatch.setitem(sys.modules, "src.evaluation", stale_child)

    try:
        prepare_project_imports(project_root)

        assert "src" not in sys.modules
        assert "src.evaluation" not in sys.modules
    finally:
        sys.path[:] = original_sys_path


def test_prepare_project_imports_invalidates_import_caches(tmp_path, monkeypatch):
    project_root = tmp_path / "MLDS_Final_Project"
    write_minimal_project(project_root)
    calls = []
    original_sys_path = list(sys.path)
    monkeypatch.setattr(colab, "invalidate_caches", lambda: calls.append(True))

    try:
        prepare_project_imports(project_root)

        assert calls == [True]
    finally:
        sys.path[:] = original_sys_path


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


def test_colab_mount_passes_force_remount(monkeypatch):
    mount_calls = []

    google_module = ModuleType("google")
    colab_module = ModuleType("google.colab")
    setattr(google_module, "__path__", [])
    setattr(google_module, "colab", colab_module)
    setattr(
        colab_module,
        "drive",
        SimpleNamespace(mount=lambda path, force_remount=False: mount_calls.append((path, force_remount))),
    )
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.colab", colab_module)

    mount_colab_drive_if_available(force_remount=True)

    assert mount_calls == [("/content/drive", True)]


def test_stage_colab_data_copies_drive_zip_and_extracts_to_local_disk(tmp_path, monkeypatch):
    source = tmp_path / "drive" / "train"
    destination = tmp_path / "content" / "train"
    source.parent.mkdir(parents=True)
    write_flat_image_zip(colab_data_zip_path(source), ["cat.0.jpg", "dog.1.jpg"])
    monkeypatch.setattr(colab, "path_is_under_colab_drive", lambda path: True)

    staged = stage_colab_data_to_local_disk(
        source,
        local_data_dir=destination,
        using_google_colab=True,
    )

    assert staged == str(destination)
    assert sorted(path.name for path in destination.iterdir()) == ["cat.0.jpg", "dog.1.jpg"]
    assert colab_data_zip_path(destination).exists()


def test_stage_colab_data_reuses_existing_local_copy(tmp_path, monkeypatch):
    source = tmp_path / "drive" / "train"
    destination = tmp_path / "content" / "train"
    source.parent.mkdir(parents=True)
    destination.mkdir(parents=True)
    write_flat_image_zip(colab_data_zip_path(source), ["cat.0.jpg", "dog.1.jpg"])
    for filename in ("cat.0.jpg", "dog.1.jpg"):
        (destination / filename).write_bytes(b"local")
    monkeypatch.setattr(colab, "path_is_under_colab_drive", lambda path: True)

    staged = stage_colab_data_to_local_disk(
        source,
        local_data_dir=destination,
        using_google_colab=True,
    )

    assert staged == str(destination)
    assert (destination / "cat.0.jpg").read_bytes() == b"local"


def test_stage_colab_data_can_skip_local_copy_when_disabled(tmp_path, monkeypatch, capsys):
    source = tmp_path / "drive" / "train"
    destination = tmp_path / "content" / "train"
    source.mkdir(parents=True)
    (source / "cat.0.jpg").write_bytes(b"cat")
    monkeypatch.setattr(colab, "path_is_under_colab_drive", lambda path: True)

    staged = stage_colab_data_to_local_disk(
        source,
        local_data_dir=destination,
        enabled=False,
        using_google_colab=True,
    )

    assert staged == str(source)
    assert not destination.exists()
    assert "staging disabled" in capsys.readouterr().out


def test_stage_colab_data_leaves_non_colab_path_unchanged(tmp_path):
    source = tmp_path / "train"
    source.mkdir()

    staged = stage_colab_data_to_local_disk(source, using_google_colab=False)

    assert staged == str(source)


def test_stage_colab_data_missing_zip_warns_and_keeps_drive_path(tmp_path, monkeypatch, capsys):
    source = tmp_path / "drive" / "train"
    destination = tmp_path / "content" / "train"
    source.mkdir(parents=True)
    (source / "cat.0.jpg").write_bytes(b"cat")
    monkeypatch.setattr(colab, "path_is_under_colab_drive", lambda path: True)

    staged = stage_colab_data_to_local_disk(
        source,
        local_data_dir=destination,
        using_google_colab=True,
    )

    assert staged == str(source)
    assert not destination.exists()
    assert "ZIP was not found" in capsys.readouterr().out


def test_nbconvert_tex_package_check_treats_missing_kpsewhich_as_missing(monkeypatch):
    monkeypatch.setattr(notebook_setup, "_command_exists", lambda command: False)
    monkeypatch.setattr(
        notebook_setup.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("kpsewhich should not be invoked when missing")),
    )

    assert notebook_setup._nbconvert_tex_packages_missing() is True


def test_zip_train_images_script_creates_flat_archive(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "zip_train_images.py"
    spec = importlib.util.spec_from_file_location("zip_train_images", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "train"
    output = tmp_path / "train.zip"
    source.mkdir()
    (source / "cat.0.jpg").write_bytes(b"cat")
    (source / "dog.1.jpg").write_bytes(b"dog")
    (source / "notes.txt").write_text("skip")

    count = module.zip_train_images(source, output)

    assert count == 2
    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == ["cat.0.jpg", "dog.1.jpg"]
