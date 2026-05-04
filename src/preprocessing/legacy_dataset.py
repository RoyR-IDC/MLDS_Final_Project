"""Backward-compatible dataset wrapper for older notebooks/tests."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

from torch.utils.data import Dataset
from torchvision import transforms

from src.preprocessing.tile_permutation import ImageFileDataset, TilePermutationDataset as _TilePermutationDataset


class TilePermutationDataset(Dataset):
    """Dataset that loads image files and applies optional tile permutations.

    This compatibility class preserves the repository's original constructor while
    delegating tensor permutation work to ``src.preprocessing.tile_permutation``.

    Args:
        samples: Sequence of ``(image_path, label)`` pairs.
        grid_size: Number of tiles along each image side.
        base_transform: Transform applied before tile permutation.
        tile_transform: Deprecated and ignored.
        permutation: Optional fixed permutation mapping output positions to sources.
        seed: Seed for deterministic random permutations.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        grid_size: int = 1,
        base_transform: Optional[Callable] = None,
        tile_transform: Optional[Callable] = None,
        permutation: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        del tile_transform
        transform = base_transform or transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        base_dataset = ImageFileDataset(samples, transform=transform)
        self.dataset = _TilePermutationDataset(
            base_dataset,
            grid_size=grid_size,
            permutation=permutation,
            seed=0 if seed is None else seed,
        )

    def __len__(self) -> int:
        """Return the number of samples."""

        sample_count = len(self.dataset)
        return sample_count

    def __getitem__(self, idx: int):
        """Return one ``(image, label)`` sample."""

        sample = self.dataset[idx]
        return sample
