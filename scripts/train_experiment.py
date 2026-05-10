"""Legacy experiment launcher.

The official project workflow is notebook-only. Keep this script as a clear
failure mode for older commands that still try to launch YAML experiments.
"""

from __future__ import annotations

import argparse


def main(config_path: str | None = None) -> None:
    """Reject scriptable experiment execution in favor of notebooks.

    Args:
        config_path: Ignored legacy YAML config path.

    Raises:
        ValueError: Always, because all parts are notebook-owned.
    """

    _ = config_path
    raise ValueError("All experiment parts are notebook-owned. Run the matching notebook under src/notebooks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", type=str)
    args = parser.parse_args()
    main(args.config)
