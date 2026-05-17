"""Google Colab setup helpers that avoid binary package churn."""

from __future__ import annotations

from importlib import import_module, invalidate_caches
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import re
import subprocess
import sys
from typing import Iterable


DEFAULT_COLAB_DRIVE_ROOT = "/content/drive/MyDrive/MLDS_Final_Project"
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


def mount_colab_drive_if_available() -> None:
    """Mount Google Drive in Colab when the Drive API is available."""

    if not is_google_colab_runtime():
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
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
) -> Path:
    """Prepare imports, install Colab-safe deps, and print runtime diagnostics."""

    if is_google_colab_runtime():
        mount_colab_drive_if_available()
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
