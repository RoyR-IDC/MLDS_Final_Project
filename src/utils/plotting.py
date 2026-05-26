"""Reusable plotting helpers for notebooks and experiments."""

from __future__ import annotations

from typing import Sequence, cast

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from torch import Tensor
import torchvision

from src.preprocessing.image_transforms import PILToFloatTensor, make_tile_compatible_image_size
from src.preprocessing.samples import Sample
from src.preprocessing.tile_permutations import TilePermutationRecord
from src.preprocessing.tile_transforms import apply_tile_permutation


def _class_name(label: int) -> str:
    return "Cat" if int(label) == 0 else "Dog"


def _select_balanced_display_samples(samples: Sequence[Sample], samples_per_class: int) -> list[Sample]:
    cat_samples = [sample for sample in samples if sample[1] == 0][:samples_per_class]
    dog_samples = [sample for sample in samples if sample[1] == 1][:samples_per_class]
    return cat_samples + dog_samples


def _select_display_tile_permutation_records(
    tile_permutation_records: Sequence[TilePermutationRecord],
    max_records: int,
) -> list[TilePermutationRecord]:
    """Select non-1x1 records, because the regular image already shows that case."""

    if max_records < 0:
        raise ValueError("max_records must be non-negative")
    display_records = [
        record
        for record in tile_permutation_records
        if record.tiles_per_side is not None and record.tiles_per_side > 1 and record.tile_permutation is not None
    ]
    return display_records[:max_records]


def plot_tile_permutation_samples(
    samples: Sequence[Sample],
    tile_permutation_records: Sequence[TilePermutationRecord],
    image_size: int,
    samples_per_class: int = 1,
    max_records: int = 4,
) -> Figure:
    """Plot selected samples above tile-reordered variants.

    The first row represents the unpermuted 1x1 case, so 1x1 tile-permutation
    records are intentionally skipped to avoid duplicate rows.

    Args:
        samples: Labeled ``(path, label)`` image samples.
        tile_permutation_records: Candidate tile-permutation records to visualize.
        image_size: Base image size used by the experiment config.
        samples_per_class: Number of cat and dog samples to display. Defaults to
            one cat and one dog.
        max_records: Maximum non-1x1 tile-permutation records to display.

    Returns:
        Matplotlib figure containing the sample grid.
    """

    sample_pairs = _select_balanced_display_samples(samples, samples_per_class)
    if not sample_pairs:
        raise ValueError("No samples available to plot")

    display_records = _select_display_tile_permutation_records(tile_permutation_records, max_records)
    n_rows = 1 + len(display_records)
    n_columns = len(sample_pairs)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(4 * n_columns, 3.6 * n_rows),
        squeeze=False,
    )

    for col_index, (path, label) in enumerate(sample_pairs):
        label_name = _class_name(label)
        with PILImage.open(path) as image:
            image = image.convert("RGB")
            axes[0, col_index].imshow(image)
            axes[0, col_index].set_title(f"{label_name} regular")
            axes[0, col_index].axis("off")

            for row_index, record in enumerate(display_records, start=1):
                assert record.tiles_per_side is not None
                assert record.tile_permutation is not None
                tile_image_size = make_tile_compatible_image_size(image_size, record.tiles_per_side)
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((tile_image_size, tile_image_size)),
                        PILToFloatTensor(),
                    ]
                )
                image_tensor = cast(Tensor, transform(image))
                reordered_tensor = apply_tile_permutation(
                    image_tensor,
                    record.tile_permutation,
                )
                reordered_image = np.asarray(
                    reordered_tensor.detach().cpu().permute(1, 2, 0).numpy(force=True),
                    dtype=np.float32,
                ).clip(0.0, 1.0)
                axes[row_index, col_index].imshow(reordered_image)
                axes[row_index, col_index].set_title(
                    f"{label_name} {record.tiles_per_side}x{record.tiles_per_side} "
                    f"{record.tile_permutation_name or record.tile_permutation_id}"
                )
                axes[row_index, col_index].axis("off")

    fig.tight_layout()
    return fig
