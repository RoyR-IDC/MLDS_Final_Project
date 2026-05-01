from typing import Callable, List, Optional, Sequence, Tuple
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TilePermutationDataset(Dataset):
    """
    PyTorch Dataset that splits images into tiles and returns a permuted reassembly.

    This dataset reads images from provided (path, label) pairs, resizes each
    image to a canonical size (default 224x224), splits it into a GxG grid,
    optionally permutes the tiles using a provided permutation, and returns the
    reassembled image as a tensor along with the label.

    Args:
        samples: Sequence of (image_path, label) tuples.
        grid_size: int grid dimension G. Default 1 (no tiling).
        base_transform: Transform to apply to the final assembled image.
        tile_transform: Optional transform applied to each tile before reassembly.
        permutation: Optional sequence of indices length G*G specifying the
            mapping from output positions to source tile indices. If None,
            a random permutation is sampled per item (controlled by `seed`).
        seed: Optional int seed that makes per-item random permutations reproducible.
    """

    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        grid_size: int = 1,
        base_transform: Optional[Callable] = None,
        tile_transform: Optional[Callable] = None,
        permutation: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
    ):
        self.samples = list(samples)
        self.grid_size = grid_size
        self.base_transform = base_transform or transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.tile_transform = tile_transform
        self.permutation = list(permutation) if permutation is not None else None
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def _split_into_tiles(self, img: Image.Image) -> List[Image.Image]:
        w, h = img.size
        G = self.grid_size
        tile_w = w // G
        tile_h = h // G
        tiles: List[Image.Image] = []
        for r in range(G):
            for c in range(G):
                left = c * tile_w
                upper = r * tile_h
                right = left + tile_w
                lower = upper + tile_h
                tiles.append(img.crop((left, upper, right, lower)))
        return tiles

    def _reassemble(self, tiles: List[Image.Image], order: Sequence[int]) -> Image.Image:
        G = self.grid_size
        tile_w, tile_h = tiles[0].size
        out = Image.new(tiles[0].mode, (tile_w * G, tile_h * G))
        for idx, src_pos in enumerate(order):
            r = idx // G
            c = idx % G
            tile = tiles[src_pos]
            out.paste(tile, (c * tile_w, r * tile_h))
        return out

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Resize to canonical size that divides evenly by grid
        img_resized = transforms.Resize((224, 224))(img)

        if self.grid_size == 1:
            img_t = self.base_transform(img_resized)
            return img_t, int(label)

        tiles = self._split_into_tiles(img_resized)

        # Apply per-tile transform if provided (expect PIL in / out)
        if self.tile_transform is not None:
            tiles = [self.tile_transform(t) for t in tiles]

        # Determine permutation
        if self.permutation is None:
            rng = random.Random(self.seed + idx if self.seed is not None else None)
            order = list(range(len(tiles)))
            rng.shuffle(order)
        else:
            order = list(self.permutation)

        assembled = self._reassemble(tiles, order)
        img_t = self.base_transform(assembled)
        return img_t, int(label)
