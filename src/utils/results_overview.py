"""Display helpers for the final results overview notebook."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from IPython.display import Image as DisplayImage
from IPython.display import Markdown, display


TIMESTAMP_SUFFIX = re.compile(r"_(\d{8}_\d{6})$")
PART3_HARDNESS_EXAMPLES_FILENAME = "part3_hardness_examples.png"

PART_LABELS = {
    "part1": "Part 1",
    "part2": "Part 2",
    "part3": "Part 3",
}
KNOWN_TITLE_PARTS = {
    "accuracy": "Accuracy",
    "tiles": "Tiles",
    "deit": "DeiT",
    "tiny": "Tiny",
    "mobilenetv3": "MobileNetV3",
    "small": "Small",
    "gmlp": "gMLP",
    "s16": "S16",
    "mlp": "MLP",
    "mixer": "Mixer",
    "base": "Base",
    "resnet18": "ResNet-18",
    "ablation": "Ablation",
    "augmentation": "Augmentation",
    "patch": "Patch",
    "shuffle": "Shuffle",
    "curriculum": "Curriculum",
    "corruption": "Corruption",
    "probability": "Probability",
    "permutation": "Permutation",
    "difficulty": "Difficulty",
    "mixed": "Mixed",
    "original": "Original",
    "permuted": "Permuted",
    "regular": "Regular",
    "augmentations": "Augmentations",
    "head": "Head",
    "comparison": "Comparison",
    "global": "Global",
    "tile": "Tile",
    "displacement": "Displacement",
    "grid": "Grid",
    "hardness": "Hardness",
    "accuracy": "Accuracy",
    "3d": "3D",
    "adjacency": "Adjacency",
    "destruction": "Destruction",
    "spatial": "Spatial",
    "entropy": "Entropy",
    "combined": "Combined",
    "examples": "Examples",
}


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find the project root by walking upward from ``start``."""

    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "outputs").exists():
            return candidate
    raise FileNotFoundError("Could not find the project root containing src/ and outputs/.")


def overview_paths(project_root: Path) -> dict[str, Path]:
    """Return standard paths used by the overview notebook."""

    figures_dir = project_root / "outputs" / "figures"
    return {
        "results_dir": project_root / "outputs" / "results",
        "figures_dir": figures_dir,
        "intermediate_figures_dir": figures_dir / "intermediate",
        "part3_hardness_examples": figures_dir / PART3_HARDNESS_EXAMPLES_FILENAME,
    }


def relative_path(path: Path, project_root: Path) -> str:
    """Return a notebook-friendly path relative to the project root when possible."""

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def intermediate_family(path: Path) -> str:
    """Return a stable family name by stripping timestamp suffixes."""

    return TIMESTAMP_SUFFIX.sub("", path.stem)


def latest_intermediate_figures(intermediate_figures_dir: Path, prefix: Optional[str] = None) -> list[Path]:
    """Return the newest or canonical intermediate figure for each figure family."""

    figures = sorted(intermediate_figures_dir.glob("*.png"))
    if prefix is not None:
        figures = [path for path in figures if path.name.startswith(prefix)]

    latest_by_family: dict[str, Path] = {}
    for path in figures:
        family = intermediate_family(path)
        current = latest_by_family.get(family)
        if current is None:
            latest_by_family[family] = path
            continue

        current_is_canonical = current.stem == family
        path_is_canonical = path.stem == family
        if path_is_canonical and not current_is_canonical:
            latest_by_family[family] = path
        elif path_is_canonical == current_is_canonical and path.stat().st_mtime > current.stat().st_mtime:
            latest_by_family[family] = path

    return [latest_by_family[key] for key in sorted(latest_by_family)]


def _title_piece(piece: str) -> str:
    return KNOWN_TITLE_PARTS.get(piece, piece.replace("-", " ").title())


def _humanize_slug(slug: str) -> str:
    return " ".join(_title_piece(piece) for piece in slug.split("_") if piece)


def figure_title(path: Path) -> str:
    """Return a readable experiment/run title for a saved figure path."""

    stem = TIMESTAMP_SUFFIX.sub("", path.stem)
    if stem == "part3_hardness_examples":
        return "Part 3: Hardness Examples"

    for part_name, part_label in PART_LABELS.items():
        prefix = f"{part_name}_"
        if stem.startswith(prefix):
            title_slug = stem[len(prefix) :]
            return f"{part_label}: {_humanize_slug(title_slug)}"

    return _humanize_slug(stem)


def show_image(path: Path, *, project_root: Path, title: Optional[str] = None) -> None:
    """Display one image with a small heading and relative path label."""

    display(Markdown(f"#### {title or figure_title(path)}"))
    if not path.exists():
        display(Markdown(f"**Missing:** `{relative_path(path, project_root)}`"))
        return
    display(Markdown(f"`{relative_path(path, project_root)}`"))
    display(DisplayImage(filename=str(path)))


def show_images(paths: Sequence[Path], *, project_root: Path) -> None:
    """Display images with headings."""

    if not paths:
        display(Markdown("No figures found."))
        return
    for path in paths:
        show_image(path, project_root=project_root)


def show_table(path: Path, *, project_root: Path, rows: int = 20) -> None:
    """Display the first rows of a CSV table."""

    if not path.exists():
        display(Markdown(f"**Missing:** `{relative_path(path, project_root)}`"))
        return
    frame = pd.read_csv(path)
    display(Markdown(f"**{relative_path(path, project_root)}** - {len(frame)} rows x {len(frame.columns)} columns"))
    display(frame.head(rows))


def show_tables(paths: Sequence[Path], *, project_root: Path, rows: int = 20) -> None:
    """Display a sequence of CSV tables."""

    for path in paths:
        show_table(path, project_root=project_root, rows=rows)


def _part3_figure_sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == PART3_HARDNESS_EXAMPLES_FILENAME:
        return (0, 0, path.name)
    is_3d = path.name.endswith("_3d.png")
    return (1, 1 if is_3d else 0, path.name)


def part3_figures(figures_dir: Path) -> list[Path]:
    """Return Part 3 figure paths, including the hardness examples plot first."""

    paths = list(figures_dir.glob("part3_*.png"))
    hardness_examples = figures_dir / PART3_HARDNESS_EXAMPLES_FILENAME
    if hardness_examples not in paths:
        paths.append(hardness_examples)
    return sorted(paths, key=_part3_figure_sort_key)


def _deduplicate_part3_baseline_rows(tile_permutations: pd.DataFrame) -> pd.DataFrame:
    """Keep one no-permutation baseline row while preserving all real permutations."""

    if tile_permutations.empty:
        return tile_permutations

    is_baseline = tile_permutations["tiles_per_side"].isna()
    if "tile_permutation" in tile_permutations.columns:
        is_baseline = is_baseline | tile_permutations["tile_permutation"].isna()

    baseline_rows = tile_permutations[is_baseline].head(1)
    permutation_rows = tile_permutations[~is_baseline]
    return pd.concat([baseline_rows, permutation_rows], ignore_index=True)


def ensure_part3_hardness_examples_figure(project_root: Path, *, force: bool = False) -> Path:
    """Save the Part 3 example-image hardness plot if it is missing or forced."""

    output_path = overview_paths(project_root)["part3_hardness_examples"]
    if output_path.exists() and not force:
        return output_path

    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    from src.evaluation.experiment_results import (
        load_experiment_samples,
        load_or_build_part1_tile_permutations,
    )
    from src.evaluation.tile_permutation_difficulty import (
        compute_adjacency_destruction_hardness,
        compute_combined_hardness,
        compute_global_displacement,
        compute_spatial_permutation_entropy,
    )
    from src.preprocessing.image_transforms import PILToFloatTensor, make_tile_compatible_image_size
    from src.preprocessing.tile_permutations import tile_permutation_from_jsonable
    from src.preprocessing.tile_transforms import apply_tile_permutation
    from src.utils.config import Part3ExperimentConfig

    config = Part3ExperimentConfig(root_dir=str(project_root))
    _, validation_samples, _ = load_experiment_samples(config, seed=config.seed)
    if not validation_samples:
        raise ValueError("No validation samples available for the Part 3 hardness examples plot.")

    image_path = Path(sorted(validation_samples, key=lambda sample: str(sample[0]))[0][0])
    tile_permutations = load_or_build_part1_tile_permutations(
        tile_permutation_csv=str(project_root / "outputs" / "results" / "part1_tile_permutations.csv"),
        tiles_per_side_values=config.tiles_per_side_values,
        num_tile_permutations=config.num_tile_permutations,
        seed=config.seed,
    ).copy()
    tile_permutations = _deduplicate_part3_baseline_rows(tile_permutations)
    tile_permutations["_tiles_sort"] = tile_permutations["tiles_per_side"].fillna(0).astype(int)
    tile_permutations = tile_permutations.sort_values(["_tiles_sort", "tile_permutation_id"])

    rows = []
    for _, row in tile_permutations.iterrows():
        raw_tiles_per_side = row["tiles_per_side"]
        tiles_per_side = None if pd.isna(raw_tiles_per_side) else int(raw_tiles_per_side)
        effective_tiles_per_side = 1 if tiles_per_side is None else tiles_per_side
        raw_tile_permutation = row["tile_permutation"]
        if raw_tile_permutation is None or (isinstance(raw_tile_permutation, float) and pd.isna(raw_tile_permutation)):
            tile_permutation = None
        else:
            if isinstance(raw_tile_permutation, str):
                raw_tile_permutation = json.loads(raw_tile_permutation)
            tile_permutation = tile_permutation_from_jsonable(raw_tile_permutation, tiles_per_side)

        tile_compatible_size = make_tile_compatible_image_size(config.image_size, effective_tiles_per_side)
        with PILImage.open(image_path) as image:
            resized = image.convert("RGB").resize(
                (tile_compatible_size, tile_compatible_size),
                PILImage.Resampling.BILINEAR,
            )
        image_tensor = PILToFloatTensor()(resized)
        displayed_tensor = image_tensor if tile_permutation is None else apply_tile_permutation(image_tensor, tile_permutation)

        adjacency_hardness = compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side)
        global_displacement = compute_global_displacement(tile_permutation, tiles_per_side)
        spatial_entropy = compute_spatial_permutation_entropy(tile_permutation, tiles_per_side)
        combined_hardness = compute_combined_hardness(
            adjacency_destruction_hardness=adjacency_hardness,
            spatial_permutation_entropy=spatial_entropy,
            global_tile_displacement=global_displacement,
            weight_adj=config.weight_adj,
            weight_entropy=config.weight_entropy,
            weight_dist=config.weight_dist,
        )
        rows.append(
            {
                "grid": "baseline" if tiles_per_side is None else f"{tiles_per_side}x{tiles_per_side}",
                "tile_permutation_name": "no permutation" if tile_permutation is None else row.get("tile_permutation_name"),
                "image_array": displayed_tensor.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy(),
                "adjacency_destruction_hardness": adjacency_hardness,
                "global_tile_displacement": global_displacement,
                "spatial_permutation_entropy": spatial_entropy,
                "combined_hardness_score": combined_hardness,
            }
        )

    condition_order = ["easy", "medium", "hard"]
    grid_order = ["baseline", "4x4", "7x7", "10x10"]
    row_by_grid_condition = {
        (str(row["grid"]), str(row["tile_permutation_name"])): row
        for row in rows
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(condition_order),
        len(grid_order),
        figsize=(12, 8.4),
        squeeze=False,
    )
    fig.suptitle("Deterministic validation image under fixed permutations", y=0.995)
    baseline_row = next(row for row in rows if row["grid"] == "baseline")
    for row_idx, condition_name in enumerate(condition_order):
        for col_idx, grid_name in enumerate(grid_order):
            axis = axes[row_idx][col_idx]
            is_baseline_column = grid_name == "baseline"
            show_baseline = is_baseline_column and row_idx == 1
            row = baseline_row if grid_name == "baseline" else row_by_grid_condition[(grid_name, condition_name)]
            axis.axis("off")
            if not is_baseline_column or show_baseline:
                axis.imshow(row["image_array"])
            if row_idx == 0:
                axis.set_title("No permutation" if grid_name == "baseline" else f"Grid {grid_name}")
            if col_idx == 0:
                axis.text(
                    -0.05,
                    0.5,
                    condition_name.capitalize(),
                    va="center",
                    ha="right",
                    rotation=90,
                    transform=axis.transAxes,
                    fontsize=12,
                    fontweight="bold",
                )
            if is_baseline_column and not show_baseline:
                continue
            metric_text = "\n".join(
                [
                    f"A {row['adjacency_destruction_hardness']:.3f}",
                    f"D {row['global_tile_displacement']:.3f}",
                    f"E {row['spatial_permutation_entropy']:.3f}",
                    f"C {row['combined_hardness_score']:.3f}",
                ]
            )
            axis.text(
                0.02,
                0.02,
                metric_text,
                transform=axis.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.62, "pad": 2, "edgecolor": "none"},
            )
    fig.text(
        0.5,
        0.01,
        "Metric overlay: A = adjacency destruction, D = global displacement, E = spatial entropy, C = combined hardness.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.965])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
