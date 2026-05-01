from typing import List, Tuple
import math
import numpy as np


def average_displacement(permutation: List[int], grid_size: int) -> float:
    """
    Compute the average normalized displacement of tiles under a permutation.

    The displacement of a tile is the Euclidean distance between its original
    grid center and its permuted position, normalized by the grid diagonal.
    """
    size = grid_size * grid_size
    coords = [(i // grid_size, i % grid_size) for i in range(size)]
    diag = math.sqrt((grid_size - 1) ** 2 + (grid_size - 1) ** 2)
    if diag == 0:
        return 0.0
    dists = []
    for src_idx, dst_idx in enumerate(permutation):
        src = coords[src_idx]
        dst = coords[dst_idx]
        dist = math.sqrt((src[0] - dst[0]) ** 2 + (src[1] - dst[1]) ** 2) / diag
        dists.append(dist)
    return float(np.mean(dists))


def adjacency_preservation(permutation: List[int], grid_size: int) -> float:
    """
    Fraction of originally adjacent tile pairs that remain adjacent after permutation.
    Returns 1.0 for perfect adjacency preservation, 0.0 for none.
    """
    size = grid_size * grid_size
    coords = [(i // grid_size, i % grid_size) for i in range(size)]
    # Build set of original adjacency pairs (undirected)
    adjacent = set()
    for i in range(size):
        r, c = coords[i]
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                j = nr * grid_size + nc
                adjacent.add(tuple(sorted((i, j))))
    kept = 0
    for a, b in adjacent:
        # After permutation, compute new positions
        pa = permutation[a]
        pb = permutation[b]
        ra, ca = coords[pa]
        rb, cb = coords[pb]
        if abs(ra - rb) + abs(ca - cb) == 1:
            kept += 1
    return float(kept / max(1, len(adjacent)))


def displacement_entropy(permutation: List[int], grid_size: int) -> float:
    """
    Shannon entropy of displacement distances (binned) as a proxy for disorder.
    """
    size = grid_size * grid_size
    coords = [(i // grid_size, i % grid_size) for i in range(size)]
    dists = []
    for i, dst in enumerate(permutation):
        r1, c1 = coords[i]
        r2, c2 = coords[dst]
        dists.append(math.hypot(r1 - r2, c1 - c2))
    hist, _ = np.histogram(dists, bins=min(10, size))
    probs = hist.astype(float) / hist.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())
