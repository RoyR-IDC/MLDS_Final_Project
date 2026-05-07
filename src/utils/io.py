"""Small file I/O helpers for experiments."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Mapping

import pandas as pd
import yaml


def ensure_dir(path: str) -> str:
    """Create a directory if needed and return its path."""

    os.makedirs(path, exist_ok=True)
    return path


def load_yaml(path: str) -> dict:
    """Load a YAML file."""

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(data: Mapping[str, Any], path: str) -> None:
    """Save mapping data as formatted JSON."""

    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(data, path: str) -> None:
    """Save rows or a DataFrame to CSV."""

    ensure_dir(os.path.dirname(path) or ".")
    if isinstance(data, pd.DataFrame):
        data.to_csv(path, index=False)
        return

    rows = list(data)
    if not rows:
        with open(path, "w", encoding="utf-8", newline=""):
            pass
        return

    if not all(isinstance(row, Mapping) for row in rows):
        frame = pd.DataFrame(rows)
        frame.to_csv(path, index=False)
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            key = str(key)
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({str(key): value for key, value in row.items()})
