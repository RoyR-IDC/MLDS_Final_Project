import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.utils import notebook_helpers as nh
from PIL import Image


def test_create_synthetic_and_split():
    imgs = nh.create_synthetic_rgb_images(3, size=(128, 128), seed=1)
    assert len(imgs) == 3
    tiles = nh.split_into_tiles(imgs[0], grid_side_length=2)
    assert len(tiles) == 4
    assert isinstance(tiles[0], Image.Image)


def test_visualize_tiles_returns_figure():
    imgs = nh.create_synthetic_rgb_images(1, size=(128, 128), seed=2)
    tiles = nh.split_into_tiles(imgs[0], grid_side_length=2)
    fig = nh.visualize_tiles(tiles)
    # Matplotlib Figure type check
    import matplotlib
    assert isinstance(fig, matplotlib.figure.Figure)
