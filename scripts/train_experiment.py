"""Dispatch scriptable experiment configs.

Part 1 and Part 2 are notebook-only workflows. This script supports Part 3
analysis because it remains a reusable evaluation entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_experiment_config


def main(config_path: str) -> None:
    """Run the scriptable experiment described by a YAML config.

    Args:
        config_path: YAML config path.

    Raises:
        ValueError: If the config requests a notebook-only part.
    """

    config = load_experiment_config(config_path)
    part = str(config.get("part", "part1")).lower()
    if part in {"part1", "part2"}:
        raise ValueError(f"{part} is notebook-only. Run the matching notebook under src/notebooks instead.")
    if part == "part3":
        from src.evaluation.part3_difficulty import run_part3

        run_part3(config)
    else:
        raise ValueError(f"Unknown experiment part: {part}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
