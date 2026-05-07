"""Model factory for Dogs vs Cats experiments."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
import torchvision.models as tv_models


class ConvMixerBlock(nn.Module):
    """One ConvMixer block with depthwise and pointwise convolutions."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pointwise = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the ConvMixer block."""

        x = x + self.depthwise(x)
        return self.pointwise(x)


class ConvMixer(nn.Module):
    """Compact ConvMixer classifier.

    Args:
        num_classes: Number of output classes.
        dim: Hidden channel width.
        depth: Number of ConvMixer blocks.
        patch_size: Patch embedding stride and kernel size.
        kernel_size: Depthwise convolution kernel size.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dim: int = 256,
        depth: int = 8,
        patch_size: int = 7,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(*[ConvMixerBlock(dim, kernel_size) for _ in range(depth)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits."""

        return self.head(self.blocks(self.stem(x)))


def _weights(enum_class: Any, pretrained: bool) -> Optional[Any]:
    return enum_class.DEFAULT if pretrained else None


def _replace_classifier(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "swin_t":
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported classifier replacement for {name}")


def freeze_feature_extractor(model: nn.Module) -> None:
    """Freeze all parameters except common classifier heads."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("fc", "head", "classifier"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True


def get_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = False,
    device: Optional[torch.device] = None,
    freeze_backbone: bool = False,
    convmixer_dim: int = 256,
    convmixer_depth: int = 8,
) -> nn.Module:
    """Create a supported image classifier.

    Args:
        name: Model name: ``resnet18``, ``resnet34``, ``swin_t``, or ``convmixer``.
        num_classes: Number of output classes.
        pretrained: Whether to use torchvision pretrained weights where available.
        device: Optional device to move the model to.
        freeze_backbone: Whether to freeze feature extractor parameters.
        convmixer_dim: Hidden width for local ConvMixer.
        convmixer_depth: Number of ConvMixer blocks.

    Returns:
        A PyTorch model.
    """

    key = name.lower()
    if key == "resnet18":
        model = tv_models.resnet18(weights=_weights(tv_models.ResNet18_Weights, pretrained))
        model = _replace_classifier(model, key, num_classes)
    elif key == "resnet34":
        model = tv_models.resnet34(weights=_weights(tv_models.ResNet34_Weights, pretrained))
        model = _replace_classifier(model, key, num_classes)
    elif key == "swin_t":
        model = tv_models.swin_t(weights=_weights(tv_models.Swin_T_Weights, pretrained))
        model = _replace_classifier(model, key, num_classes)
    elif key == "convmixer":
        if pretrained:
            raise ValueError("Local ConvMixer does not provide pretrained weights; set pretrained=false.")
        model = ConvMixer(num_classes=num_classes, dim=convmixer_dim, depth=convmixer_depth)
    else:
        raise ValueError(f"Unsupported model: {name}")

    if freeze_backbone:
        freeze_feature_extractor(model)
    if device is not None:
        model = model.to(device)
    return model
