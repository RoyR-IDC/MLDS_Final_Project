"""Compatibility wrapper for the model factory."""

from src.models.factory import ConvMixer, ConvMixerBlock, freeze_feature_extractor, get_model

__all__ = ["ConvMixer", "ConvMixerBlock", "freeze_feature_extractor", "get_model"]

