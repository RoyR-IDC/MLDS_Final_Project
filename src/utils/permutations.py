"""Small dependency-light permutation helpers."""

from __future__ import annotations

import random
from typing import List


def identity_permutation(grid_size: int) -> List[int]:
    """Return the identity permutation for a square tile grid."""

    if grid_size < 1:
        raise ValueError("grid_size must be at least 1")
    return list(range(grid_size * grid_size))


def random_permutation(grid_size: int, seed: int) -> List[int]:
    """Return one seeded random permutation."""

    permutation = identity_permutation(grid_size)
    random.Random(seed).shuffle(permutation)
    return permutation


def generate_permutations(grid_size: int, n: int, seed: int = 0) -> List[List[int]]:
    """Generate ``n`` seeded random permutations."""

    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    permutations = []
    for _ in range(n):
        permutation = identity_permutation(grid_size)
        rng.shuffle(permutation)
        permutations.append(permutation)
    return permutations
