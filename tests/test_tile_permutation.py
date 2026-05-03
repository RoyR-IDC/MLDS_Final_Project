import pytest

torch = pytest.importorskip("torch")

from src.data.tile_permutation import (
    apply_tile_permutation,
    identity_permutation,
    random_permutation,
    reconstruct_from_tiles,
    split_into_tiles,
)


def test_identity_permutation_returns_same_tensor():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    output = apply_tile_permutation(image, 4, identity_permutation(4))
    assert torch.equal(output, image)


def test_random_permutation_preserves_shape():
    image = torch.randn(3, 12, 12)
    output = apply_tile_permutation(image, 3, random_permutation(3, seed=123))
    assert output.shape == image.shape


def test_invalid_grid_size_raises_clear_error():
    image = torch.randn(3, 10, 8)
    with pytest.raises(ValueError, match="divisible"):
        apply_tile_permutation(image, 3, identity_permutation(3))


def test_permutation_direction_maps_output_position_to_source_tile():
    image = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4)
    tiles = split_into_tiles(image, 2)
    permutation = [3, 2, 1, 0]
    output = apply_tile_permutation(image, 2, permutation)
    expected = reconstruct_from_tiles(tiles[torch.tensor(permutation)], 2)
    assert torch.equal(output, expected)
