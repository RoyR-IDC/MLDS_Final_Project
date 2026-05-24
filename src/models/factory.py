"""Model factory for Dogs vs Cats experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import warnings

import timm
from torch import nn
from torch._C import device as TorchDevice
import torchvision.models as tv_models

from src.models.registry import TIMM_MODEL_IDS, format_supported_model_names, validate_model_name


def _weights(enum_class: Any, pretrained: bool) -> Optional[Any]:
    return enum_class.DEFAULT if pretrained else None


@dataclass(frozen=True)
class ModelTrainingOptions:
    """Resolved model initialization options."""

    pretrained: bool
    freeze_backbone: bool


def resolve_model_training_options(
    name: str,
    *,
    pretrained: bool,
    freeze_backbone: bool,
) -> ModelTrainingOptions:
    """Return feasible pretrained/freeze options for the local model registry."""

    key = validate_model_name(name)
    resolved_pretrained = bool(pretrained)
    resolved_freeze_backbone = bool(freeze_backbone)

    if key in TIMM_MODEL_IDS and resolved_pretrained:
        model_id = TIMM_MODEL_IDS[key]
        if not timm.is_model_pretrained(model_id):
            warnings.warn(
                f"Requested pretrained weights for {name} via timm model '{model_id}', "
                "but this timm installation does not provide them. The same model "
                "architecture will be initialized from scratch and fully trained.",
                RuntimeWarning,
                stacklevel=2,
            )
            resolved_pretrained = False
            resolved_freeze_backbone = False

    return ModelTrainingOptions(
        pretrained=resolved_pretrained,
        freeze_backbone=resolved_freeze_backbone,
    )


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
    device: Optional[TorchDevice] = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create one of the three supported lightweight image classifiers.

    Args:
        name: Model name: ``resnet18``, ``deit_tiny``,
            ``mlp_mixer_base``, or ``mlp_mixer_small``.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        device: Optional device to move the model to.
        freeze_backbone: Whether to freeze feature extractor parameters. Defaults
            to ``True`` so experiment runs train only the classifier head unless
            explicitly opted out.

    Returns:
        A PyTorch model.
    """

    key = validate_model_name(name)
    options = resolve_model_training_options(
        key,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    if key == "resnet18":
        model = tv_models.resnet18(weights=_weights(tv_models.ResNet18_Weights, options.pretrained))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif key in TIMM_MODEL_IDS:
        model_id = TIMM_MODEL_IDS[key]
        model = timm.create_model(model_id, pretrained=options.pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}. Supported models: {format_supported_model_names()}")

    if options.freeze_backbone:
        freeze_feature_extractor(model)
    if device is not None:
        model = model.to(device)
    return model
