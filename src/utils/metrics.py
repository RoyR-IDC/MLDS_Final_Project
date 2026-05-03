"""Compatibility wrappers for permutation difficulty metrics."""

from src.metrics.permutation_difficulty import (
    adjacency_preservation,
    average_displacement,
    combined_difficulty_score,
    displacement_entropy,
    locality_disruption,
    normalized_average_displacement,
)

__all__ = [
    "average_displacement",
    "normalized_average_displacement",
    "adjacency_preservation",
    "locality_disruption",
    "displacement_entropy",
    "combined_difficulty_score",
]

