from typing import List, Sequence
import random


def generate_permutations(grid_size: int, n: int, seed: int = 0) -> List[List[int]]:
    """
    Generate `n` random permutations for a grid of size GxG.

    Args:
        grid_size: Grid dimension G (int).
        n: Number of permutations to generate.
        seed: RNG seed for reproducibility.

    Returns:
        List of permutations, each a list of length G*G containing indices 0..G*G-1.
    """
    rng = random.Random(seed)
    size = grid_size * grid_size
    perms = []
    base = list(range(size))
    for i in range(n):
        p = base.copy()
        rng.shuffle(p)
        perms.append(p)
    return perms


def identity_permutation(grid_size: int) -> List[int]:
    """Return the identity permutation for a GxG grid."""
    size = grid_size * grid_size
    return list(range(size))
