"""Model factory for Dogs vs Cats experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import timm
from torch import nn
from torch._C import device as TorchDevice

from src.models.registry import (
    ACTIVE_MODEL_NAMES,
    TIMM_MODEL_IDS,
    format_supported_model_names,
    validate_model_name,
)


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

    if resolved_pretrained:
        validate_imagenet1k_pretrained_model_id(key)

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


def validate_imagenet1k_pretrained_model_id(name: str) -> str:
    """Return the timm id after verifying an ImageNet-1K pretrained checkpoint exists."""

    key = validate_model_name(name)
    model_id = TIMM_MODEL_IDS[key]
    lowered_id = model_id.lower()
    banned_tokens = ("in21k", "21k", "jft", "hybrid")
    if any(token in lowered_id for token in banned_tokens):
        raise ValueError(f"timm model '{model_id}' is not an ImageNet-1K-only checkpoint")
    if not timm.is_model_pretrained(model_id):
        raise ValueError(f"timm model '{model_id}' does not provide pretrained ImageNet-1K weights")
    return model_id


def _validate_loaded_pretrained_cfg(model: nn.Module, *, name: str, model_id: str) -> None:
    """Fail fast if timm metadata indicates a non-ImageNet-1K source."""

    cfg = getattr(model, "pretrained_cfg", None) or {}
    cfg_text = " ".join(str(value).lower() for value in cfg.values())
    banned_tokens = ("in21k", "21k", "jft", "hybrid")
    if any(token in cfg_text for token in banned_tokens):
        raise ValueError(f"{name} resolved to non-ImageNet-1K pretrained metadata for '{model_id}'")
    num_classes = cfg.get("num_classes")
    if num_classes is not None and int(num_classes) != 1000:
        raise ValueError(
            f"{name} pretrained metadata for '{model_id}' reports num_classes={num_classes}, expected 1000"
        )


def _classifier_attr_name(model: nn.Module) -> str:
    for attr_name in ("fc", "head", "classifier"):
        if hasattr(model, attr_name):
            return attr_name
    raise ValueError("Model does not expose a supported classifier attribute")


def _replace_classifier_with_mlp(model: nn.Module, num_classes: int) -> None:
    classifier_name = _classifier_attr_name(model)
    classifier = getattr(model, classifier_name)
    if not isinstance(classifier, nn.Linear):
        raise ValueError("MLP classification_head requires a linear classifier")
    setattr(
        model,
        classifier_name,
        nn.Sequential(
            nn.Linear(classifier.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        ),
    )


def _format_pretrained_source(model: nn.Module) -> str:
    cfg = getattr(model, "pretrained_cfg", None) or {}
    source_fields = {
        key: cfg.get(key)
        for key in ("architecture", "tag", "url", "file", "hf_hub_id", "source")
        if cfg.get(key)
    }
    return ", ".join(f"{key}={value}" for key, value in source_fields.items()) or "unavailable"


def _log_model_init(
    *,
    requested_name: str,
    model_id: str,
    model: nn.Module,
    pretrained: bool,
) -> None:
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"Model init: name={requested_name}, timm_model_id={model_id}, pretrained={pretrained}, "
        f"total_params={total_params:,}, trainable_params={trainable_params:,}, "
        f"pretrained_source={_format_pretrained_source(model)}"
    )


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create one active ImageNet-1K pretrained model via timm."""

    key = validate_model_name(name)
    if key not in ACTIVE_MODEL_NAMES:
        raise ValueError(
            f"create_model supports active models only: {', '.join(ACTIVE_MODEL_NAMES)}; got '{name}'"
        )
    model_id = validate_imagenet1k_pretrained_model_id(key) if pretrained else TIMM_MODEL_IDS[key]
    model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
    if pretrained:
        _validate_loaded_pretrained_cfg(model, name=key, model_id=model_id)
    _log_model_init(requested_name=key, model_id=model_id, model=model, pretrained=pretrained)
    return model


def get_model(
    name: str,
    num_classes: int = 2,
    pretrained: bool = False,
    device: Optional[TorchDevice] = None,
    freeze_backbone: bool = True,
    classification_head: str = "linear",
) -> nn.Module:
    """Create one supported lightweight image classifier.

    Args:
        name: Model name. Active choices are ``mobilenetv3_small``,
            ``deit_tiny``, and ``gmlp_s16``. ``resnet18`` is retained as a
            configurable legacy CNN option.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.
        device: Optional device to move the model to.
        freeze_backbone: Whether to freeze feature extractor parameters. Defaults
            to ``True`` so experiment runs train only the classifier head unless
            explicitly opted out.
        classification_head: Classifier head variant, either ``"linear"`` or
            ``"mlp"``.

    Returns:
        A PyTorch model.
    """

    key = validate_model_name(name)
    options = resolve_model_training_options(
        key,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    head_name = str(classification_head or "linear").lower()
    if key in ACTIVE_MODEL_NAMES:
        model = create_model(key, num_classes=num_classes, pretrained=options.pretrained)
    elif key in TIMM_MODEL_IDS:
        model_id = validate_imagenet1k_pretrained_model_id(key) if options.pretrained else TIMM_MODEL_IDS[key]
        model = timm.create_model(model_id, pretrained=options.pretrained, num_classes=num_classes)
        if options.pretrained:
            _validate_loaded_pretrained_cfg(model, name=key, model_id=model_id)
        _log_model_init(requested_name=key, model_id=model_id, model=model, pretrained=options.pretrained)
    else:
        raise ValueError(f"Unsupported model: {name}. Supported models: {format_supported_model_names()}")
    if head_name == "mlp":
        _replace_classifier_with_mlp(model, num_classes)
    elif head_name != "linear":
        raise ValueError(f"Unsupported classification_head: {classification_head}")

    if options.freeze_backbone:
        freeze_feature_extractor(model)
    if device is not None:
        model = model.to(device)
    return model


def get_imagenet_pretrained_model(
    name: str,
    *,
    device: Optional[TorchDevice] = None,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a pretrained ImageNet-1k classifier with its native output head."""

    key = validate_model_name(name)
    model_id = validate_imagenet1k_pretrained_model_id(key)
    model = timm.create_model(model_id, pretrained=True)
    _validate_loaded_pretrained_cfg(model, name=key, model_id=model_id)
    _log_model_init(requested_name=key, model_id=model_id, model=model, pretrained=True)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    if device is not None:
        model = model.to(device)
    return model
