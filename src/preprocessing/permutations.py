"""Dependency-light tile permutation generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class PermutationRecord:
    """Metadata for one reusable tile permutation.

    Attributes:
        grid_size: Number of tiles along each image side.
        permutation_id: Stable ID within the grid.
        permutation_seed: Seed used to generate the permutation, or ``None`` for identity.
        permutation: Mapping from output tile position to source tile index.
    """

    grid_size: int
    permutation_id: int
    permutation_seed: Optional[int]
    permutation: List[int]


def identity_permutation(grid_size: int) -> List[int]:
    """Return the identity permutation for a ``grid_size x grid_size`` grid.

    Args:
        grid_size: Number of tiles along each image side.

    Returns:
        Row-major identity permutation.
    """

    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    permutation = list(range(grid_size * grid_size))
    return permutation


def random_permutation(grid_size: int, seed: int) -> List[int]:
    """Generate one seeded random tile permutation.

    Args:
        grid_size: Number of tiles along each image side.
        seed: Random seed.

    Returns:
        A permutation mapping output tile position to source tile index.
    """

    permutation = identity_permutation(grid_size)
    random.Random(seed).shuffle(permutation)
    shuffled_permutation = permutation
    return shuffled_permutation


def generate_permutations(grid_size: int, n: int, seed: int = 0) -> List[List[int]]:
    """Generate reusable seeded random permutations for a grid.

    Args:
        grid_size: Number of tiles along each image side.
        n: Number of random permutations to generate.
        seed: Seed for the permutation generator.

    Returns:
        List of permutations, each mapping output position to source tile index.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    permutations: List[List[int]] = []
    for _ in range(n):
        permutation = identity_permutation(grid_size)
        rng.shuffle(permutation)
        permutations.append(permutation)
    generated_permutations = permutations
    return generated_permutations


def build_permutation_records(
    grid_sizes: Iterable[int],
    num_permutations: int,
    permutation_seed: int = 42,
    include_identity: bool = True,
) -> List[PermutationRecord]:
    """Build stable permutation records for experiment reuse.

    Args:
        grid_sizes: Grid side lengths to include.
        num_permutations: Number of random permutations per non-identity grid.
        permutation_seed: Seed used to generate random permutations.
        include_identity: Whether to include identity at ``permutation_id=0``.

    Returns:
        List of permutation records.
    """

    records: List[PermutationRecord] = []
    for grid_size in grid_sizes:
        next_id = 0
        if include_identity:
            records.append(
                PermutationRecord(
                    grid_size=grid_size,
                    permutation_id=next_id,
                    permutation_seed=None,
                    permutation=identity_permutation(grid_size),
                )
            )
            next_id += 1
        for offset, permutation in enumerate(generate_permutations(grid_size, num_permutations, permutation_seed)):
            records.append(
                PermutationRecord(
                    grid_size=grid_size,
                    permutation_id=next_id + offset,
                    permutation_seed=permutation_seed,
                    permutation=permutation,
                )
            )
    permutation_records = records
    return permutation_records
