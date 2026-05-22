#!/usr/bin/env python3
"""Create a flat ZIP archive for the Dogs vs Cats training images."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import zipfile


DEFAULT_SOURCE_DIR = Path("data/dogs-vs-cats/train")
DEFAULT_OUTPUT_PATH = Path("data/dogs-vs-cats/train.zip")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def is_labeled_image_path(path: Path) -> bool:
    """Return whether ``path`` looks like a labeled Dogs vs Cats image."""

    name = path.name.lower()
    return path.suffix.lower() in IMAGE_SUFFIXES and ("cat" in name or "dog" in name)


def format_file_size(size_bytes: int) -> str:
    """Return a compact human-readable file size."""

    units = ("B", "KiB", "MiB", "GiB")
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def zip_train_images(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    *,
    compression: int = zipfile.ZIP_STORED,
) -> int:
    """Zip labeled images from ``source_dir`` into ``output_path`` and return the count."""

    start_time = perf_counter()
    source_dir = source_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    print("Creating Dogs vs Cats training image ZIP.")
    print(f"  Source directory: {source_dir}")
    print(f"  Output ZIP: {output_path}")
    print("  ZIP layout: flat image filenames at archive root")

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    image_paths = sorted(path for path in source_dir.iterdir() if path.is_file() and is_labeled_image_path(path))
    print(f"  Labeled image files found: {len(image_paths)}")
    if not image_paths:
        raise FileNotFoundError(f"No labeled cat/dog images found in: {source_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="w", compression=compression) as archive:
        for image_path in image_paths:
            archive.write(image_path, arcname=image_path.name)

    elapsed_seconds = perf_counter() - start_time
    output_size = output_path.stat().st_size
    print(f"Finished ZIP creation in {elapsed_seconds:.2f}s.")
    print(f"  Images written: {len(image_paths)}")
    print(f"  ZIP size: {format_file_size(output_size)}")
    return len(image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing train images. Defaults to {DEFAULT_SOURCE_DIR}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"ZIP file to create. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    parser.add_argument(
        "--deflate",
        action="store_true",
        help="Use ZIP_DEFLATED compression instead of fast storage.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = zipfile.ZIP_DEFLATED if args.deflate else zipfile.ZIP_STORED
    zip_train_images(args.source, args.output, compression=compression)


if __name__ == "__main__":
    main()
