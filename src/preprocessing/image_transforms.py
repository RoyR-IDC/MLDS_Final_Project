"""Image preprocessing transforms for Dogs vs Cats experiments."""

from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms


def make_tile_compatible_image_size(image_size: int, tiles_per_side: int) -> int:
    """Return the smallest square size that can be split by ``tiles_per_side``."""

    if image_size < 1:
        raise ValueError("image_size must be at least 1")
    if tiles_per_side < 1:
        raise ValueError("tiles_per_side must be at least 1")
    remainder = image_size % tiles_per_side
    if remainder == 0:
        return image_size
    return image_size + tiles_per_side - remainder


class PILToFloatTensor:
    """Convert a PIL image to a float tensor without going through NumPy."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        data = torch.ByteTensor(bytearray(image.tobytes()))
        return data.reshape(height, width, 3).permute(2, 0, 1).float().div(255.0)


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    standard_augmentation: bool = False,
    image_augmentation: str | None = None,
) -> transforms.Compose:
    """Build torchvision transforms for Dogs vs Cats experiments."""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train and image_augmentation == "random_erasing":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                PILToFloatTensor(),
                transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                normalize,
            ]
        )
    if train and standard_augmentation:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                PILToFloatTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.Resize((image_size, image_size)), PILToFloatTensor(), normalize])
