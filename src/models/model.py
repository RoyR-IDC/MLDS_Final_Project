from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
except Exception:
    timm = None


def get_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Return a model by name. Supports lightweight wrappers for common backbones.

    Supported names (case-insensitive): 'resnet18', 'vit_base_patch16_224', 'mixer_*.'.

    Args:
        name: Model name string.
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained weights (ImageNet).
    """
    n = name.lower()
    if 'resnet18' in n:
        model = tv_models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if 'vit' in n or 'deit' in n or 'mixer' in n:
        if timm is None:
            raise RuntimeError('timm is required for ViT/Mixer models; add timm to requirements')
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        return model

    raise ValueError(f'Unsupported model: {name}')
