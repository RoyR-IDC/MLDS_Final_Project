import pytest

torch = pytest.importorskip("torch")

from src.preprocessing.augmentations import CompositeBatchAugmentation, RandomPatchShuffle, SameLabelCutMix


def test_same_label_cutmix_preserves_hard_labels():
    images = torch.zeros(4, 3, 12, 12)
    images[1] = 1.0
    images[3] = 2.0
    targets = torch.tensor([0, 0, 1, 1])

    augmentation = SameLabelCutMix(alpha=1.0, probability=1.0)
    mixed_images, mixed_targets = augmentation(images, targets)

    assert mixed_images.shape == images.shape
    assert torch.equal(mixed_targets, targets)


def test_random_patch_shuffle_preserves_shape():
    images = torch.arange(2 * 3 * 12 * 12, dtype=torch.float32).reshape(2, 3, 12, 12)
    targets = torch.tensor([0, 1])

    augmentation = RandomPatchShuffle(tiles_per_side=3, probability=1.0)
    shuffled_images, shuffled_targets = augmentation(images, targets)

    assert shuffled_images.shape == images.shape
    assert torch.equal(shuffled_targets, targets)


def test_composite_batch_augmentation_preserves_shapes_and_targets():
    images = torch.zeros(4, 3, 12, 12)
    targets = torch.tensor([0, 0, 1, 1])
    augmentation = CompositeBatchAugmentation(
        [
            SameLabelCutMix(alpha=1.0, probability=1.0),
            RandomPatchShuffle(tiles_per_side=3, probability=1.0),
        ]
    )

    augmented_images, augmented_targets = augmentation(images, targets)

    assert augmented_images.shape == images.shape
    assert torch.equal(augmented_targets, targets)
