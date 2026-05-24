"""Shared model-name registry and validation helpers."""

from __future__ import annotations

from collections.abc import Sequence


TIMM_MODEL_IDS = {
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
    "mlp_mixer_base": "mixer_b16_224.goog_in21k_ft_in1k",
    "mlp_mixer_small": "mixer_s16_224",
}

SUPPORTED_MODEL_NAMES = ("resnet18", *TIMM_MODEL_IDS.keys())


def format_supported_model_names() -> str:
    """Return supported model names as a stable display string."""

    return ", ".join(SUPPORTED_MODEL_NAMES)


def validate_model_name(name: str) -> str:
    """Return a canonical supported model name or raise a clear error."""

    if not isinstance(name, str):
        raise TypeError(f"Model name must be a string, got {type(name).__name__}")

    key = name.strip().lower()
    if key not in SUPPORTED_MODEL_NAMES:
        raise ValueError(
            f"Unsupported model_name='{name}'. Supported models: {format_supported_model_names()}"
        )
    return key


def validate_model_names(model_names: Sequence[str]) -> list[str]:
    """Validate and canonicalize a sequence of model names."""

    if isinstance(model_names, str) or not isinstance(model_names, Sequence):
        raise TypeError("model_names must be a sequence of model-name strings")
    if not model_names:
        raise ValueError("model_names must contain at least one model")

    canonical_names = [validate_model_name(model_name) for model_name in model_names]
    duplicates = sorted({name for name in canonical_names if canonical_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"model_names contains duplicate entries: {', '.join(duplicates)}")
    return canonical_names
