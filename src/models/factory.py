"""Model factory for Dogs vs Cats experiments."""

from __future__ import annotations

from typing import Any, Optional

import timm
import torch
from torch import nn
import torchvision.models as tv_models

from src.models.registry import TIMM_MODEL_IDS, format_supported_model_names, validate_model_name


def _weights(enum_class: Any, pretrained: bool) -> Optional[Any]:
    return enum_class.DEFAULT if pretrained else None


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
) -> nn.Module:
    """Create one of the three supported lightweight image classifiers.

    Args:
        name: Model name: ``resnet18``, ``deit_tiny``, or ``mlp_mixer_small``.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        device: Optional device to move the model to.
        freeze_backbone: Whether to freeze feature extractor parameters.

    Returns:
        A PyTorch model.
    """

    key = validate_model_name(name)
    if key == "resnet18":
        model = tv_models.resnet18(weights=_weights(tv_models.ResNet18_Weights, pretrained))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key in TIMM_MODEL_IDS:
        model_id = TIMM_MODEL_IDS[key]
        if pretrained and not timm.is_model_pretrained(model_id):
            raise ValueError(
                f"Requested pretrained weights for {name} via timm model '{model_id}', "
                "but this timm installation does not provide pretrained weights for that model. "
                "Update or pin timm to a version that provides them; no fallback model is used."
            )
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}. Supported models: {format_supported_model_names()}")

    if freeze_backbone:
        freeze_feature_extractor(model)
    if device is not None:
        model = model.to(device)
    return model
