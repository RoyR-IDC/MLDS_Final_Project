import os
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image
from src.preprocessing.dogs_cats import PILToFloatTensor, build_dataloaders, make_tile_compatible_image_size
from src.preprocessing.tile_permutation import ImageFileDataset, TilePermutationDataset


def _make_image(path, color):
    img = Image.new('RGB', (224, 224), color)
    img.save(path)


def test_tile_dataset_basic(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # create 4 images alternating labels
    samples = []
    for i in range(4):
        lbl = 'cat' if i % 2 == 0 else 'dog'
        p = d / f"{lbl}.{i}.jpg"
        _make_image(p, (i * 30 % 255, i * 60 % 255, i * 90 % 255))
        samples.append((str(p), 0 if 'cat' in str(p) else 1))

    base_dataset = ImageFileDataset(samples, transform=PILToFloatTensor())
    ds = TilePermutationDataset(base_dataset, grid_size=1, permutation=None, seed=0)
    assert len(ds) == 4
    x, y = ds[0]
    # tensor shape C,H,W
    assert x.shape[0] == 3
    assert isinstance(y, int)


def test_tile_dataset_permutation(tmp_path):
    d = tmp_path / "data2"
    d.mkdir()
    samples = []
    for i in range(2):
        lbl = 'cat' if i % 2 == 0 else 'dog'
        p = d / f"{lbl}.{i}.jpg"
        _make_image(p, (255, 0, 0) if i == 0 else (0, 255, 0))
        samples.append((str(p), 0 if 'cat' in str(p) else 1))

    # grid 2x2 with explicit identity permutation
    perm = [0, 1, 2, 3]
    base_dataset = ImageFileDataset(samples, transform=PILToFloatTensor())
    ds = TilePermutationDataset(base_dataset, grid_size=2, permutation=perm, seed=0)
    x, y = ds[0]
    assert x.shape[0] == 3
    assert isinstance(y, int)


def test_grid_three_dataloader_adjusts_image_size(tmp_path):
    d = tmp_path / "data3"
    d.mkdir()
    samples = []
    for i in range(4):
        lbl = 'cat' if i % 2 == 0 else 'dog'
        p = d / f"{lbl}.{i}.jpg"
        _make_image(p, (i * 40 % 255, i * 70 % 255, i * 100 % 255))
        samples.append((str(p), 0 if lbl == 'cat' else 1))

    assert make_tile_compatible_image_size(224, 3) == 225
    train_loader, _ = build_dataloaders(
        samples[:2],
        samples[2:],
        image_size=224,
        grid_size=3,
        permutation=list(range(9)),
        batch_size=2,
        num_workers=0,
    )
    images, targets = next(iter(train_loader))
    assert images.shape == (2, 3, 225, 225)
    assert targets.shape == (2,)
