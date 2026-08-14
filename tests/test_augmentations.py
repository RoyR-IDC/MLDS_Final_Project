import pytest

torch = pytest.importorskip("torch")

from src.preprocessing.augmentations import NoBatchAugmentation, RandomPatchShuffle


def test_no_batch_augmentation_preserves_batch():
    images = torch.zeros(4, 3, 12, 12)
    targets = torch.tensor([0, 0, 1, 1])

    augmented_images, augmented_targets = NoBatchAugmentation()(images, targets)

    assert torch.equal(augmented_images, images)
    assert torch.equal(augmented_targets, targets)


def test_random_patch_shuffle_preserves_shape():
    images = torch.arange(2 * 3 * 12 * 12, dtype=torch.float32).reshape(2, 3, 12, 12)
    targets = torch.tensor([0, 1])

    augmentation = RandomPatchShuffle(tiles_per_side=3, probability=1.0)
    shuffled_images, shuffled_targets = augmentation(images, targets)

    assert shuffled_images.shape == images.shape
    assert torch.equal(shuffled_targets, targets)


def test_random_patch_shuffle_handles_tile_incompatible_shape():
    images = torch.arange(2 * 3 * 224 * 224, dtype=torch.float32).reshape(2, 3, 224, 224)
    targets = torch.tensor([0, 1])

    augmentation = RandomPatchShuffle(tiles_per_side=10, probability=1.0)
    shuffled_images, shuffled_targets = augmentation(images, targets)

    assert shuffled_images.shape == images.shape
    assert torch.equal(shuffled_targets, targets)
