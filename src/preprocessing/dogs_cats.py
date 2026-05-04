"""Dogs vs Cats dataset discovery, splitting, and dataloader helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from src.preprocessing.tile_permutation import ImageFileDataset, TilePermutationDataset


Sample = Tuple[str, int]


def parse_label_from_filename(path: str) -> int:
    """Parse a Dogs vs Cats label from a Kaggle training filename.

    Args:
        path: File path such as ``cat.123.jpg`` or ``dog.456.jpg``.

    Returns:
        ``0`` for cat and ``1`` for dog.

    Raises:
        ValueError: If the filename does not contain a known label prefix.
    """

    filename = os.path.basename(path).lower()
    if filename.startswith("cat."):
        return 0
    if filename.startswith("dog."):
        return 1
    raise ValueError(f"Cannot parse cat/dog label from filename: {path}")


def discover_samples(data_dir: str, limit: Optional[int] = None) -> List[Sample]:
    """Discover labeled Dogs vs Cats image samples.

    Args:
        data_dir: Directory containing Kaggle training images.
        limit: Optional maximum number of samples after sorting.

    Returns:
        Sorted list of ``(path, label)`` samples.
    """

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    samples: List[Sample] = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(data_dir, filename)
        try:
            samples.append((path, parse_label_from_filename(path)))
        except ValueError:
            continue
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"No labeled cat/dog images found in {data_dir}")
    return samples


def class_counts(samples: Sequence[Sample]) -> Dict[str, int]:
    """Count cats and dogs in a sample list.

    Args:
        samples: Labeled image samples.

    Returns:
        Dictionary with cat and dog counts.
    """

    counts = Counter(label for _, label in samples)
    label_counts = {"cat": counts.get(0, 0), "dog": counts.get(1, 0)}
    return label_counts


def stratified_split(
    samples: Sequence[Sample],
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split samples into stratified train, validation, and optional test sets.

    Args:
        samples: Labeled image samples.
        val_fraction: Fraction assigned to validation.
        test_fraction: Fraction assigned to test.
        seed: Shuffle seed.

    Returns:
        ``(train_samples, val_samples, test_samples)``.
    """

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to less than 1")
    by_label: Dict[int, List[Sample]] = defaultdict(list)
    for sample in samples:
        by_label[sample[1]].append(sample)

    rng = random.Random(seed)
    train: List[Sample] = []
    val: List[Sample] = []
    test: List[Sample] = []
    for label_samples in by_label.values():
        label_samples = list(label_samples)
        rng.shuffle(label_samples)
        n_total = len(label_samples)
        n_test = int(round(n_total * test_fraction))
        n_val = int(round(n_total * val_fraction))
        test.extend(label_samples[:n_test])
        val.extend(label_samples[n_test : n_test + n_val])
        train.extend(label_samples[n_test + n_val :])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    split_samples = (train, val, test)
    return split_samples


def build_transforms(image_size: int = 224, train: bool = False, standard_augmentation: bool = False):
    """Build torchvision transforms for Dogs vs Cats experiments.

    Args:
        image_size: Output square image size.
        train: Whether the transform is for training.
        standard_augmentation: Whether to add standard image augmentations.

    Returns:
        A torchvision transform pipeline.
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train and standard_augmentation:
        transform_pipeline = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return transform_pipeline
    transform_pipeline = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize])
    return transform_pipeline


def build_dataset(
    samples: Sequence[Sample],
    image_size: int,
    grid_size: int = 1,
    permutation: Optional[Sequence[int]] = None,
    random_permutations: Optional[Sequence[Sequence[int]]] = None,
    seed: int = 0,
    train: bool = False,
    standard_augmentation: bool = False,
):
    """Build an image dataset with optional tile permutation wrapping.

    Args:
        samples: Labeled image samples.
        image_size: Output square image size.
        grid_size: Number of tiles along each image side.
        permutation: Optional fixed permutation.
        random_permutations: Optional permutation pool sampled per item.
        seed: Seed used for deterministic sampled permutations.
        train: Whether to build training transforms.
        standard_augmentation: Whether to add standard image augmentations.

    Returns:
        Image dataset, optionally wrapped with tile permutation behavior.
    """

    transform = build_transforms(image_size=image_size, train=train, standard_augmentation=standard_augmentation)
    base_dataset = ImageFileDataset(samples, transform=transform)
    if grid_size == 1 and not random_permutations:
        dataset = base_dataset
        return dataset
    dataset = TilePermutationDataset(
        base_dataset,
        grid_size=grid_size,
        permutation=permutation,
        random_permutations=random_permutations,
        seed=seed,
    )
    return dataset


def build_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    image_size: int = 224,
    grid_size: int = 1,
    permutation: Optional[Sequence[int]] = None,
    random_permutations: Optional[Sequence[Sequence[int]]] = None,
    seed: int = 0,
    batch_size: int = 32,
    num_workers: int = 2,
    standard_augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders.

    Args:
        train_samples: Training samples.
        val_samples: Validation samples.
        image_size: Output square image size.
        grid_size: Number of tiles along each image side.
        permutation: Optional fixed tile permutation.
        random_permutations: Optional permutation pool for training.
        seed: Seed used for sampled permutation behavior.
        batch_size: Dataloader batch size.
        num_workers: Number of dataloader workers.
        standard_augmentation: Whether to apply standard training augmentations.

    Returns:
        Pair of training and validation dataloaders.
    """

    train_dataset = build_dataset(
        train_samples,
        image_size=image_size,
        grid_size=grid_size,
        permutation=permutation,
        random_permutations=random_permutations,
        seed=seed,
        train=True,
        standard_augmentation=standard_augmentation,
    )
    val_dataset = build_dataset(
        val_samples,
        image_size=image_size,
        grid_size=grid_size,
        permutation=permutation,
        seed=seed,
        train=False,
        standard_augmentation=False,
    )
    dataloaders = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
    return dataloaders
