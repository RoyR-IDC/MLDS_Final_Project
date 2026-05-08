"""Reusable plotting helpers for notebooks and experiments."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import torchvision

from src.preprocessing.dogs_cats import PILToFloatTensor, Sample, make_tile_compatible_image_size
from src.preprocessing.permutations import PermutationRecord
from src.preprocessing.tile_permutation import apply_tile_permutation


def _class_name(label: int) -> str:
    return "Cat" if int(label) == 0 else "Dog"


def _select_balanced_display_samples(samples: Sequence[Sample], samples_per_class: int) -> list[Sample]:
    cat_samples = [sample for sample in samples if sample[1] == 0][:samples_per_class]
    dog_samples = [sample for sample in samples if sample[1] == 1][:samples_per_class]
    return cat_samples + dog_samples


def _select_display_permutation_records(
    permutation_records: Sequence[PermutationRecord],
    max_records: int,
) -> list[PermutationRecord]:
    """Select non-1x1 records, because the regular image already shows that case."""

    if max_records < 0:
        raise ValueError("max_records must be non-negative")
    display_records = [record for record in permutation_records if record.grid_size > 1]
    return display_records[:max_records]


def plot_permutation_samples(
    samples: Sequence[Sample],
    permutation_records: Sequence[PermutationRecord],
    image_size: int,
    samples_per_class: int = 2,
    max_records: int = 4,
) -> plt.Figure:
    """Plot original samples next to selected tile-permuted variants.

    The original image column represents the unpermuted 1x1 case, so 1x1
    permutation records are intentionally skipped to avoid duplicate columns.

    Args:
        samples: Labeled ``(path, label)`` image samples.
        permutation_records: Candidate permutation records to visualize.
        image_size: Base image size used by the experiment config.
        samples_per_class: Number of cat and dog samples to display.
        max_records: Maximum non-1x1 permutation records to display.

    Returns:
        Matplotlib figure containing the sample grid.
    """

    sample_pairs = _select_balanced_display_samples(samples, samples_per_class)
    if not sample_pairs:
        raise ValueError("No samples available to plot")

    display_records = _select_display_permutation_records(permutation_records, max_records)
    n_columns = 1 + len(display_records)
    fig, axes = plt.subplots(
        len(sample_pairs),
        n_columns,
        figsize=(4 * n_columns, 4 * len(sample_pairs)),
        squeeze=False,
    )

    for row_index, (path, label) in enumerate(sample_pairs):
        label_name = _class_name(label)
        with PILImage.open(path) as image:
            image = image.convert("RGB")
            axes[row_index, 0].imshow(image)
            axes[row_index, 0].set_title(f"{label_name} regular")
            axes[row_index, 0].axis("off")

            for col_index, record in enumerate(display_records, start=1):
                tile_image_size = make_tile_compatible_image_size(image_size, record.grid_size)
                transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.Resize((tile_image_size, tile_image_size)),
                        PILToFloatTensor(),
                    ]
                )
                image_tensor = transform(image)
                permuted_tensor = apply_tile_permutation(image_tensor, record.grid_size, record.permutation)
                permuted_image = np.asarray(
                    permuted_tensor.detach().cpu().permute(1, 2, 0).numpy(force=True),
                    dtype=np.float32,
                ).clip(0.0, 1.0)
                axes[row_index, col_index].imshow(permuted_image)
                axes[row_index, col_index].set_title(
                    f"{label_name} {record.grid_size}x{record.grid_size} perm {record.permutation_id}"
                )
                axes[row_index, col_index].axis("off")

    fig.tight_layout()
    return fig
