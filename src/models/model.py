"""Compatibility wrapper for the model factory."""

from src.models.factory import TIMM_MODEL_IDS, freeze_feature_extractor, get_model

__all__ = ["TIMM_MODEL_IDS", "freeze_feature_extractor", "get_model"]
