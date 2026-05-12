"""Dogs vs Cats sample discovery, counting, and splitting helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from src.preprocessing.labels import AnimalLabel, parse_label_from_filename


Sample = Tuple[str, int]


def discover_samples(data_dir: str, limit: Optional[int] = None) -> List[Sample]:
    """Discover labeled Dogs vs Cats image samples."""

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
    """Count cats and dogs in a sample list."""

    counts = Counter(label for _, label in samples)
    return {"cat": counts.get(int(AnimalLabel.CAT), 0), "dog": counts.get(int(AnimalLabel.DOG), 0)}


def stratified_split(
    samples: Sequence[Sample],
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 0,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Split samples into stratified train, validation, and optional test sets."""

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
    return train, val, test
