import pytest

torch = pytest.importorskip("torch")

from src.preprocessing.tile_order_dataset import (
    apply_tile_order,
    reconstruct_from_tiles,
    split_into_tiles,
)
from src.preprocessing.tile_orders import identity_tile_order, random_tile_order


def test_identity_tile_order_returns_same_tensor():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    output = apply_tile_order(image, 4, identity_tile_order(4))
    assert torch.equal(output, image)


def test_random_tile_order_preserves_shape():
    image = torch.randn(3, 12, 12)
    output = apply_tile_order(image, 3, random_tile_order(3, seed=123))
    assert output.shape == image.shape


def test_invalid_grid_side_length_raises_clear_error():
    image = torch.randn(3, 10, 8)
    with pytest.raises(ValueError, match="divisible"):
        apply_tile_order(image, 3, identity_tile_order(3))


def test_output_tile_order_direction_maps_output_position_to_source_tile():
    image = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4)
    tiles = split_into_tiles(image, 2)
    output_tile_order = [3, 2, 1, 0]
    output = apply_tile_order(image, 2, output_tile_order)
    expected = reconstruct_from_tiles(tiles[torch.tensor(output_tile_order)], 2)
    assert torch.equal(output, expected)
