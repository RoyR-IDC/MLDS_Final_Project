"""Dispatch experiment configs to the Part 1, Part 2, or Part 3 runners."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_experiment_config


def main(config_path: str) -> None:
    """Run the experiment described by a YAML config."""

    config = load_experiment_config(config_path)
    part = str(config.get("part", "part1")).lower()
    if part == "part1":
        from src.training.part1_baselines import run_part1

        run_part1(config)
    elif part == "part2":
        from src.training.part2_improvement import run_part2

        run_part2(config)
    elif part == "part3":
        from src.evaluation.part3_difficulty import run_part3

        run_part3(config)
    else:
        raise ValueError(f"Unknown experiment part: {part}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    main(args.config)
