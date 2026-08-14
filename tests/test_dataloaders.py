import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image
from torchvision import transforms

from src.preprocessing.dataloaders import build_dataloaders
import src.preprocessing.dataloaders as dataloaders_module
from src.preprocessing.image_transforms import make_tile_compatible_image_size
from src.preprocessing.tile_permutations import identity_tile_permutation


def _make_image(path, color):
    Image.new("RGB", (224, 224), color).save(path)


def test_grid_three_dataloader_adjusts_image_size(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = []
    for index in range(4):
        label_name = "cat" if index % 2 == 0 else "dog"
        path = data_dir / f"{label_name}.{index}.jpg"
        _make_image(path, (index * 40 % 255, index * 70 % 255, index * 100 % 255))
        samples.append((str(path), 0 if label_name == "cat" else 1))

    assert make_tile_compatible_image_size(224, 3) == 225
    train_loader, _ = build_dataloaders(
        samples[:2],
        samples[2:],
        image_size=224,
        tiles_per_side=3,
        tile_permutation=identity_tile_permutation(3),
        batch_size=2,
        num_workers=0,
    )
    images, targets = next(iter(train_loader))

    assert images.shape == (2, 3, 225, 225)
    assert targets.shape == (2,)


def test_dataloader_can_crop_tile_compatible_image_back_to_model_size(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = []
    for index in range(4):
        label_name = "cat" if index % 2 == 0 else "dog"
        path = data_dir / f"{label_name}.{index}.jpg"
        _make_image(path, (index * 40 % 255, index * 70 % 255, index * 100 % 255))
        samples.append((str(path), 0 if label_name == "cat" else 1))

    assert make_tile_compatible_image_size(224, 10) == 230
    train_loader, _ = build_dataloaders(
        samples[:2],
        samples[2:],
        image_size=224,
        tiles_per_side=10,
        tile_permutation=identity_tile_permutation(10),
        batch_size=2,
        num_workers=0,
        output_image_size=224,
    )
    images, targets = next(iter(train_loader))

    assert images.shape == (2, 3, 224, 224)
    assert targets.shape == (2,)


def test_regular_augmentations_are_train_only(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = []
    for index in range(4):
        label_name = "cat" if index % 2 == 0 else "dog"
        path = data_dir / f"{label_name}.{index}.jpg"
        _make_image(path, (index * 40 % 255, index * 70 % 255, index * 100 % 255))
        samples.append((str(path), 0 if label_name == "cat" else 1))

    train_loader, val_loader = build_dataloaders(
        samples[:2],
        samples[2:],
        image_size=224,
        batch_size=2,
        num_workers=0,
        image_augmentation="regular_augmentations",
    )

    train_transform_types = [type(transform) for transform in train_loader.dataset.transform.transforms]
    val_transform_types = [type(transform) for transform in val_loader.dataset.transform.transforms]

    assert transforms.RandomRotation in train_transform_types
    assert transforms.ColorJitter in train_transform_types
    assert transforms.RandomRotation not in val_transform_types
    assert transforms.ColorJitter not in val_transform_types


def test_dataloader_uses_cuda_loader_options_when_workers_enabled(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = []
    for index in range(4):
        label_name = "cat" if index % 2 == 0 else "dog"
        path = data_dir / f"{label_name}.{index}.jpg"
        _make_image(path, (index * 20 % 255, index * 30 % 255, index * 40 % 255))
        samples.append((str(path), 0 if label_name == "cat" else 1))
    monkeypatch.setattr(dataloaders_module.torch.cuda, "is_available", lambda: True)

    train_loader, _ = build_dataloaders(
        samples[:2],
        samples[2:],
        image_size=224,
        batch_size=2,
        num_workers=2,
    )

    assert train_loader.pin_memory is True
    assert train_loader.num_workers == 2
    assert train_loader.persistent_workers is True
    assert train_loader.prefetch_factor == 2


def test_dataloader_omits_worker_only_options_when_workers_disabled(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    samples = []
    for index in range(4):
        label_name = "cat" if index % 2 == 0 else "dog"
        path = data_dir / f"{label_name}.{index}.jpg"
        _make_image(path, (index * 20 % 255, index * 30 % 255, index * 40 % 255))
        samples.append((str(path), 0 if label_name == "cat" else 1))
    monkeypatch.setattr(dataloaders_module.torch.cuda, "is_available", lambda: False)

    train_loader, _ = build_dataloaders(
        samples[:2],
        samples[2:],
        image_size=224,
        batch_size=2,
        num_workers=0,
    )

    assert train_loader.pin_memory is False
    assert train_loader.num_workers == 0
    assert train_loader.persistent_workers is False
    assert train_loader.prefetch_factor is None
