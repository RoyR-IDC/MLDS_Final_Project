from src.metrics.permutation_difficulty import (
    adjacency_preservation,
    combined_difficulty_score,
    locality_disruption,
    normalized_average_displacement,
)


def test_identity_metrics_are_easy():
    permutation = list(range(9))
    assert normalized_average_displacement(permutation, 3) == 0.0
    assert adjacency_preservation(permutation, 3) == 1.0
    assert locality_disruption(permutation, 3) == 0.0
    assert combined_difficulty_score(permutation, 3) == 0.0


def test_nontrivial_metrics_are_bounded_and_harder_than_identity():
    permutation = [8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert 0.0 < normalized_average_displacement(permutation, 3) <= 1.0
    assert 0.0 <= adjacency_preservation(permutation, 3) <= 1.0
    assert 0.0 <= locality_disruption(permutation, 3) <= 1.0
    assert combined_difficulty_score(permutation, 3) > 0.0

