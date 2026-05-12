"""Label definitions and parsing for Dogs vs Cats samples."""

from __future__ import annotations

from enum import IntEnum
import os


class AnimalLabel(IntEnum):
    """Integer class labels for Dogs vs Cats training."""

    CAT = 0
    DOG = 1


def parse_label_from_filename(path: str) -> int:
    """Parse a Dogs vs Cats label from a Kaggle training filename."""

    filename = os.path.basename(path).lower()
    if filename.startswith("cat."):
        return int(AnimalLabel.CAT)
    if filename.startswith("dog."):
        return int(AnimalLabel.DOG)
    raise ValueError(f"Cannot parse cat/dog label from filename: {path}")
