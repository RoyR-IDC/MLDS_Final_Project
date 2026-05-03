"""Configuration loading and normalization helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from src.utils.io import load_yaml


GROUPED_CONFIG_KEYS = {"general", "input_output", "data", "models", "experiment", "ablations"}


def normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert grouped notebook configs into the flat internal runner format.

    Args:
        config: Raw YAML configuration, either grouped or already flat.

    Returns:
        Flat dictionary consumed by the experiment runners.
    """

    if not GROUPED_CONFIG_KEYS.intersection(config.keys()):
        return dict(config)

    general = dict(config.get("general", {}))
    input_output = dict(config.get("input_output", {}))
    data = dict(config.get("data", {}))
    models = dict(config.get("models", {}))
    experiment = dict(config.get("experiment", {}))

    normalized: Dict[str, Any] = {}
    normalized.update(general)
    normalized.update(input_output)
    normalized.update(data)
    normalized.update(models)
    normalized.update(experiment)

    if "ablations" in config:
        normalized["ablations"] = config["ablations"]

    # Keep Part 2's single model name ergonomic in YAML while preserving the
    # internal key used by the runner.
    if "model_name" not in normalized and "model_names" in normalized:
        model_names = normalized["model_names"]
        if isinstance(model_names, list) and len(model_names) == 1:
            normalized["model_name"] = model_names[0]

    return normalized


def load_experiment_config(path: str) -> Dict[str, Any]:
    """Load a YAML experiment config and normalize it for internal runners."""

    return normalize_config(load_yaml(path))

