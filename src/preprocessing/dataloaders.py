"""Dataset and DataLoader builders for Dogs vs Cats experiments."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.preprocessing.datasets import DogsCatsDataset
from src.preprocessing.image_transforms import build_transforms, make_tile_compatible_image_size
from src.preprocessing.samples import Sample
from src.preprocessing.tile_permutations import TilePermutation


def build_dataset(
    samples: Sequence[Sample],
    image_size: int,
    tiles_per_side: int = 1,
    tile_permutation: Optional[TilePermutation] = None,
    train: bool = False,
    standard_augmentation: bool = False,
    image_augmentation: str | None = None,
    tile_permutation_probability: float = 1.0,
    output_image_size: int | None = None,
) -> Dataset[tuple[torch.Tensor, int]]:
    """Build a Dogs/Cats image dataset with an optional tile permutation."""

    tile_image_size = make_tile_compatible_image_size(image_size, tiles_per_side)
    if output_image_size is not None and output_image_size < 1:
        raise ValueError("output_image_size must be at least 1")
    output_transform = (
        transforms.CenterCrop(output_image_size)
        if output_image_size is not None and output_image_size != tile_image_size
        else None
    )
    transform = build_transforms(
        image_size=tile_image_size,
        train=train,
        standard_augmentation=standard_augmentation,
        image_augmentation=image_augmentation,
    )
    return DogsCatsDataset(
        samples,
        transform=transform,
        tile_permutation=tile_permutation,
        tile_permutation_probability=tile_permutation_probability,
        output_transform=output_transform,
    )


def build_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    image_size: int = 224,
    tiles_per_side: int = 1,
    tile_permutation: Optional[TilePermutation] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    standard_augmentation: bool = False,
    image_augmentation: str | None = None,
    tile_permutation_probability: float = 1.0,
    output_image_size: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders."""

    train_dataset = build_dataset(
        train_samples,
        image_size=image_size,
        tiles_per_side=tiles_per_side,
        tile_permutation=tile_permutation,
        train=True,
        standard_augmentation=standard_augmentation,
        image_augmentation=image_augmentation,
        tile_permutation_probability=tile_permutation_probability,
        output_image_size=output_image_size,
    )
    val_dataset = build_dataset(
        val_samples,
        image_size=image_size,
        tiles_per_side=tiles_per_side,
        tile_permutation=tile_permutation,
        train=False,
        standard_augmentation=False,
        output_image_size=output_image_size,
    )
    pin_memory = torch.cuda.is_available()
    dataloader_options = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        dataloader_options["persistent_workers"] = True
        dataloader_options["prefetch_factor"] = 2
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_options),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **dataloader_options),
    )
