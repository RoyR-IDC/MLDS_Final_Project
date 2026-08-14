"""Dataset classes for Dogs vs Cats experiments."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.preprocessing.tile_permutations import TilePermutation
from src.preprocessing.tile_transforms import TilePermutationTransform


class DogsCatsDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dogs-vs-cats image-file dataset with optional tile permutation."""

    def __init__(
        self,
        samples: Sequence[tuple[str, int]],
        transform: Callable[[Image.Image], torch.Tensor],
        tile_permutation: Optional[TilePermutation] = None,
        tile_permutation_probability: float = 1.0,
        output_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.output_transform = output_transform
        self.tile_permutation_transform = (
            TilePermutationTransform(tile_permutation) if tile_permutation is not None else None
        )
        if not 0.0 <= tile_permutation_probability <= 1.0:
            raise ValueError("tile_permutation_probability must be between 0 and 1")
        self.tile_permutation_probability = tile_permutation_probability

    def __len__(self) -> int:
        """Return the number of samples."""

        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load, transform, optionally permute, and return one image sample."""

        path, label = self.samples[idx]
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        if self.tile_permutation_transform is not None and self._should_apply_tile_permutation():
            image_tensor = self.tile_permutation_transform(image_tensor)
        if self.output_transform is not None:
            image_tensor = self.output_transform(image_tensor)
        return image_tensor, int(label)

    def _should_apply_tile_permutation(self) -> bool:
        if self.tile_permutation_probability >= 1.0:
            return True
        if self.tile_permutation_probability <= 0.0:
            return False
        return bool(torch.rand(()).item() < self.tile_permutation_probability)
