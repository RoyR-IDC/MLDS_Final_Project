import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image
from src.preprocessing.image_transforms import PILToFloatTensor
from src.preprocessing.datasets import DogsCatsDataset
from src.preprocessing.tile_permutations import identity_tile_permutation


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

    ds = DogsCatsDataset(samples, transform=PILToFloatTensor())
    assert len(ds) == 4
    x, y = ds[0]
    # tensor shape C,H,W
    assert x.shape[0] == 3
    assert isinstance(y, int)


def test_tile_dataset_tile_permutation(tmp_path):
    d = tmp_path / "data2"
    d.mkdir()
    samples = []
    for i in range(2):
        lbl = 'cat' if i % 2 == 0 else 'dog'
        p = d / f"{lbl}.{i}.jpg"
        _make_image(p, (255, 0, 0) if i == 0 else (0, 255, 0))
        samples.append((str(p), 0 if 'cat' in str(p) else 1))

    ds = DogsCatsDataset(
        samples,
        transform=PILToFloatTensor(),
        tile_permutation=identity_tile_permutation(2),
    )
    x, y = ds[0]
    assert x.shape[0] == 3
    assert isinstance(y, int)
