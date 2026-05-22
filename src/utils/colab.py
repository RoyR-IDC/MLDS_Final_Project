"""Google Colab setup helpers that avoid binary package churn."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pathlib import PurePosixPath
import os
import re
import shutil
import subprocess
import sys
import zipfile
from time import perf_counter
from typing import Iterable


DEFAULT_COLAB_DRIVE_ROOT = "/content/drive/MyDrive/MLDS_Final_Project"
DEFAULT_COLAB_LOCAL_DATA_DIR = "/content/MLDS_Final_Project/data/dogs-vs-cats/train"
COMMON_COLAB_PROJECT_ROOTS = (
    DEFAULT_COLAB_DRIVE_ROOT,
    "/content/MLDS_Final_Project",
    "/content/drive/MyDrive/Colab Notebooks/MLDS_Final_Project",
)

COLAB_PREINSTALLED_REQUIREMENT_PREFIXES = (
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "torchvision",
)

COLAB_SMOKE_CHECK_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "pyyaml": "yaml",
}

_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _path_looks_like_project_root(path: Path) -> bool:
    """Return whether ``path`` looks like this repository root."""

    try:
        return (
            (path / "src" / "__init__.py").is_file()
            and (path / "src" / "utils" / "notebook_setup.py").is_file()
            and (path / "src" / "evaluation" / "experiment_results.py").is_file()
        )
    except OSError:
        return False


def _safe_resolve(path: str | os.PathLike[str]) -> Path | None:
    """Resolve a path, returning None when a Colab Drive mount is disconnected."""

    try:
        return Path(path).resolve()
    except OSError:
        return None


def _safe_cwd() -> Path:
    """Return the current directory, falling back when cwd is a broken mount."""

    try:
        return Path.cwd().resolve()
    except OSError:
        return Path("/content") if Path("/content").exists() else Path.home()


def _safe_glob(path: Path, pattern: str) -> list[Path]:
    """Glob a path without letting disconnected mounts break notebook setup."""

    try:
        return list(path.glob(pattern))
    except OSError:
        return []


def _safe_exists(path: Path) -> bool:
    """Check path existence without failing on disconnected mounts."""

    try:
        return path.exists()
    except OSError:
        return False


def _module_file(module_name: str) -> Path | None:
    """Return the module file path when the module is already imported."""

    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    try:
        return Path(module_file).resolve()
    except OSError:
        return None


def _remove_stale_src_modules(project_root: Path) -> None:
    """Drop cached ``src`` modules that were imported from outside ``project_root``."""

    expected_src = project_root / "src"
    stale_modules: list[str] = []
    for module_name in list(sys.modules):
        if module_name != "src" and not module_name.startswith("src."):
            continue
        module_file = _module_file(module_name)
        if module_file is None:
            continue
        try:
            module_file.relative_to(expected_src)
        except ValueError:
            stale_modules.append(module_name)

    for module_name in stale_modules:
        sys.modules.pop(module_name, None)


def _candidate_project_roots(start: str | os.PathLike[str] | None = None) -> list[Path]:
    """Return likely project roots without doing an expensive full Drive walk."""

    current = _safe_resolve(start) if start is not None else _safe_cwd()
    candidates = [current, *current.parents] if current is not None else []
    candidates.extend(Path(root) for root in COMMON_COLAB_PROJECT_ROOTS)

    drive_root = Path("/content/drive/MyDrive")
    if _safe_exists(drive_root):
        candidates.extend(_safe_glob(drive_root, "MLDS_Final_Project"))
        candidates.extend(_safe_glob(drive_root, "*/MLDS_Final_Project"))

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = _safe_resolve(candidate)
        if resolved_candidate is None:
            continue
        candidate = resolved_candidate
        if candidate in seen:
            continue
        unique_candidates.append(candidate)
        seen.add(candidate)
    return unique_candidates


def is_google_colab_runtime() -> bool:
    """Return True when code is executing inside Google Colab."""

    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def mount_colab_drive_if_available(*, force_remount: bool = False) -> None:
    """Mount Google Drive in Colab when the Drive API is available."""

    if not is_google_colab_runtime():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=force_remount)
    except Exception as exc:
        print(f"Google Drive was not mounted automatically: {exc}")


def find_project_root(start: str | os.PathLike[str] | None = None) -> str:
    """Find the repository root from the current notebook/script location."""

    for candidate in _candidate_project_roots(start):
        if _path_looks_like_project_root(candidate):
            return str(candidate)
    fallback = _safe_resolve(start) if start is not None else _safe_cwd()
    return str(fallback)


def prepare_project_imports(project_root: str | Path | None = None) -> Path:
    """Resolve the project root, add it to ``sys.path``, and make it the cwd."""

    if project_root is None:
        root = Path(find_project_root()).resolve()
    else:
        root = _safe_resolve(project_root)
        if root is None:
            root = Path(find_project_root(project_root)).resolve()
        if not _path_looks_like_project_root(root):
            root = Path(find_project_root(root)).resolve()
    if not _path_looks_like_project_root(root):
        raise ModuleNotFoundError(
            "Could not find the full MLDS_Final_Project repository. "
            "Expected src/__init__.py, src/utils/notebook_setup.py, and "
            "src/evaluation/experiment_results.py under the project root. "
            "In Colab, upload or clone the full repo, then rerun the setup cell."
        )

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    _remove_stale_src_modules(root)
    invalidate_caches()
    os.chdir(root)
    return root


def requirement_package_name(line: str) -> str | None:
    """Return the normalized package name from one requirements line."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("-"):
        return None
    match = _REQUIREMENT_NAME_RE.match(stripped)
    if match is None:
        return None
    return match.group(1).lower().replace("_", "-")


def filter_colab_requirements_lines(
    lines: Iterable[str],
    *,
    skip_packages: Iterable[str] = COLAB_PREINSTALLED_REQUIREMENT_PREFIXES,
) -> list[str]:
    """Return requirements lines that are safe to install into a live Colab kernel."""

    skip = {package.lower().replace("_", "-") for package in skip_packages}
    filtered: list[str] = []
    for line in lines:
        package_name = requirement_package_name(line)
        if package_name is None:
            continue
        if package_name in skip:
            continue
        filtered.append(line)
    return filtered


def write_filtered_colab_requirements(
    project_root: str | Path,
    output_path: str | Path = "/tmp/mlds_colab_requirements.txt",
) -> Path:
    """Write a Colab-safe requirements file and return its path."""

    project_root = Path(project_root)
    filtered_requirements = Path(output_path)
    requirements_path = project_root / "requirements.txt"
    filtered_lines = filter_colab_requirements_lines(requirements_path.read_text().splitlines())
    filtered_requirements.write_text("\n".join(filtered_lines) + "\n")
    return filtered_requirements


def install_project_requirements_for_colab(project_root: str | Path) -> None:
    """Install lightweight project requirements without replacing Colab binary wheels."""

    if not is_google_colab_runtime():
        return

    filtered_requirements = write_filtered_colab_requirements(project_root)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(filtered_requirements)])


def bootstrap_notebook_runtime(
    project_root: str | Path | None = None,
    *,
    install_requirements: bool = True,
    print_diagnostics: bool = True,
    force_remount: bool = False,
) -> Path:
    """Prepare imports, install Colab-safe deps, and print runtime diagnostics."""

    if is_google_colab_runtime():
        mount_colab_drive_if_available(force_remount=force_remount)
    root = prepare_project_imports(project_root)
    if install_requirements:
        install_project_requirements_for_colab(root)
    if print_diagnostics:
        print_colab_runtime_diagnostics()
        print(f"Project root: {root}")
    return root


def smoke_check_colab_packages(*, strict: bool = True) -> dict[str, str]:
    """Import key packages and return their installed versions."""

    versions: dict[str, str] = {}
    for package_name, import_name in COLAB_SMOKE_CHECK_IMPORTS.items():
        try:
            import_module(import_name)
        except ImportError:
            if strict:
                raise
            versions[package_name] = "missing"
            continue
        except OSError as exc:
            versions[package_name] = f"import failed: {exc}"
            continue
        try:
            versions[package_name] = version(package_name)
        except PackageNotFoundError:
            versions[package_name] = "installed"
    return versions


def print_colab_runtime_diagnostics(selected_device: object | None = None) -> None:
    """Print concise Colab package and CUDA diagnostics."""

    import torch

    versions = smoke_check_colab_packages(strict=is_google_colab_runtime())
    print(f"Python: {sys.version.split()[0]}")
    print("Package versions:")
    for package_name, package_version in versions.items():
        print(f"  {package_name}: {package_version}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if selected_device is not None:
        print(f"Selected device: {selected_device}")


def warn_if_colab_runtime_without_cuda(selected_device: object) -> None:
    """Print a clear warning when a Colab run is not using CUDA."""

    if not is_google_colab_runtime():
        return

    import torch

    device_type = getattr(selected_device, "type", str(selected_device).split(":", maxsplit=1)[0])
    if device_type != "cuda" or not torch.cuda.is_available():
        print(
            "WARNING: This Colab runtime is not using CUDA. "
            "Choose Runtime > Change runtime type > T4 GPU, reconnect, and rerun setup."
        )


def _is_labeled_image_path(path: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name.lower()
    return suffix in {".jpg", ".jpeg", ".png"} and ("cat" in name or "dog" in name)


def _count_labeled_images(path: Path) -> int:
    try:
        if not path.is_dir():
            return 0
        return sum(1 for item in path.iterdir() if item.is_file() and _is_labeled_image_path(item))
    except OSError:
        return 0


def colab_data_zip_path(data_dir: str | Path) -> Path:
    """Return the expected ZIP path for a Dogs vs Cats train directory."""

    return Path(f"{data_dir}.zip")


def _format_file_size(size_bytes: int) -> str:
    """Return a compact human-readable file size."""

    units = ("B", "KiB", "MiB", "GiB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _zip_labeled_image_members(zip_path: Path) -> list[zipfile.ZipInfo]:
    """Return safe labeled image members from a flat dataset ZIP."""

    members: list[zipfile.ZipInfo] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    continue
                member_name = member_path.name
                if not member_name or member_name != member.filename:
                    continue
                if _is_labeled_image_path(Path(member_name)):
                    members.append(member)
    except (OSError, zipfile.BadZipFile):
        return []
    return members


def _extract_labeled_images_from_zip(zip_path: Path, destination: Path) -> int:
    """Extract safe labeled image members from ``zip_path`` into ``destination``."""

    extracted_count = 0
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            member_name = member_path.name
            if not member_name or member_name != member.filename:
                continue
            if not _is_labeled_image_path(Path(member_name)):
                continue
            with archive.open(member) as source_file:
                with (destination / member_name).open("wb") as destination_file:
                    shutil.copyfileobj(source_file, destination_file)
            extracted_count += 1
    return extracted_count


def path_is_under_colab_drive(path: str | Path) -> bool:
    """Return whether a path points at a Google Drive mount in Colab."""

    resolved = _safe_resolve(path) or Path(path)
    return str(resolved).startswith("/content/drive/")


def stage_colab_data_to_local_disk(
    data_dir: str | Path,
    *,
    local_data_dir: str | Path = DEFAULT_COLAB_LOCAL_DATA_DIR,
    enabled: bool = True,
    using_google_colab: bool | None = None,
) -> str:
    """Copy a Drive-backed dataset ZIP to local Colab disk and extract it."""

    in_colab = is_google_colab_runtime() if using_google_colab is None else using_google_colab
    if not enabled:
        if in_colab and path_is_under_colab_drive(data_dir):
            print(f"Colab local data staging disabled; reading dataset from Google Drive: {data_dir}")
        return str(data_dir)
    if not in_colab or not path_is_under_colab_drive(data_dir):
        return str(data_dir)

    start_time = perf_counter()
    source = Path(data_dir)
    destination = Path(local_data_dir)
    source_zip = colab_data_zip_path(source)
    local_zip = colab_data_zip_path(destination)
    source_count = _count_labeled_images(source)
    destination_count = _count_labeled_images(destination)
    zip_members = _zip_labeled_image_members(source_zip)
    zip_count = len(zip_members)
    expected_count = source_count or zip_count

    print("Starting Colab dataset staging.")
    print(f"  Source data directory: {source}")
    print(f"  Expected source ZIP: {source_zip}")
    print(f"  Local ZIP destination: {local_zip}")
    print(f"  Local extraction directory: {destination}")
    print(f"  Source directory labeled images: {source_count}")
    print(f"  Source ZIP labeled images: {zip_count}")
    print(f"  Existing local labeled images: {destination_count}")

    if expected_count > 0 and destination_count >= expected_count:
        elapsed_seconds = perf_counter() - start_time
        print(
            "Using existing staged local Colab dataset: "
            f"{destination} ({destination_count} images, {elapsed_seconds:.2f}s staging check)"
        )
        return str(destination)

    if not source_zip.exists():
        elapsed_seconds = perf_counter() - start_time
        print(
            "Colab dataset ZIP was not found; leaving data on Google Drive instead of "
            f"copying {source_count} individual files. Expected ZIP: {source_zip} "
            f"({elapsed_seconds:.2f}s)"
        )
        return str(data_dir)

    if zip_count == 0:
        elapsed_seconds = perf_counter() - start_time
        print(
            "Colab dataset ZIP exists but contains no flat labeled cat/dog image files: "
            f"{source_zip} ({elapsed_seconds:.2f}s)"
        )
        return str(data_dir)

    local_zip.parent.mkdir(parents=True, exist_ok=True)
    source_zip_size = source_zip.stat().st_size
    copy_start_time = perf_counter()
    if local_zip.exists() and local_zip.stat().st_size == source_zip_size:
        print(
            "Local dataset ZIP already matches source size; reusing it: "
            f"{local_zip} ({_format_file_size(source_zip_size)})"
        )
    else:
        print(
            "Copying dataset ZIP from Google Drive to local Colab disk: "
            f"{source_zip} -> {local_zip} ({_format_file_size(source_zip_size)})"
        )
        shutil.copy2(source_zip, local_zip)
        copy_elapsed_seconds = perf_counter() - copy_start_time
        print(f"Finished copying dataset ZIP in {copy_elapsed_seconds:.2f}s.")

    extract_start_time = perf_counter()
    print(f"Extracting {zip_count} labeled images from {local_zip} into {destination}.")
    extracted_count = _extract_labeled_images_from_zip(local_zip, destination)
    extract_elapsed_seconds = perf_counter() - extract_start_time
    final_count = _count_labeled_images(destination)
    print(
        "Finished extracting Colab dataset: "
        f"{extracted_count} files extracted in {extract_elapsed_seconds:.2f}s; "
        f"{final_count} labeled images now available locally."
    )

    elapsed_seconds = perf_counter() - start_time
    print(f"Finished Colab dataset staging in {elapsed_seconds:.2f}s: {destination}")
    return str(destination)
