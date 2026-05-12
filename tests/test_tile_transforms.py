import pytest

torch = pytest.importorskip("torch")

from src.preprocessing.tile_transforms import (  # noqa: E402
    TilePermutationTransform,
    apply_tile_permutation,
    reconstruct_from_tiles,
    split_into_tiles,
)
from src.preprocessing.tile_permutations import TilePermutation, identity_tile_permutation, random_tile_permutation  # noqa: E402


def test_identity_tile_permutation_returns_same_tensor():
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
    output = apply_tile_permutation(image, identity_tile_permutation(4))
    assert torch.equal(output, image)


def test_random_tile_permutation_preserves_shape():
    image = torch.randn(3, 12, 12)
    output = apply_tile_permutation(image, random_tile_permutation(3, seed=123))
    assert output.shape == image.shape


def test_invalid_non_square_image_raises_clear_error():
    image = torch.randn(3, 10, 8)
    with pytest.raises(ValueError, match="square"):
        apply_tile_permutation(image, identity_tile_permutation(2))


def test_invalid_non_divisible_side_raises_clear_error():
    image = torch.randn(3, 10, 10)
    with pytest.raises(ValueError, match="divisible"):
        apply_tile_permutation(image, identity_tile_permutation(3))


def test_tile_permutation_direction_maps_new_position_to_source_coordinate():
    image = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4)
    tiles = split_into_tiles(image, 2)
    tile_permutation = TilePermutation(
        tiles_per_side=2,
        order=[
            [(1, 1), (1, 0)],
            [(0, 1), (0, 0)],
        ],
    )

    output = TilePermutationTransform(tile_permutation)(image)
    expected = reconstruct_from_tiles(tiles[torch.tensor([3, 2, 1, 0])], 2)

    assert torch.equal(output, expected)
