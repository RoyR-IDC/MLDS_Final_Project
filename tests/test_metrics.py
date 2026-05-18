import torch
import torch.nn.functional as F

from src.evaluation.tile_permutation_difficulty import (
    compute_adjacency_destruction_hardness,
    compute_combined_hardness,
    compute_global_displacement,
    compute_spatial_permutation_entropy,
)
from src.preprocessing.tile_permutations import TilePermutation, identity_tile_permutation
from src.training.losses import FocalLoss


def test_focal_loss_is_finite_for_confident_and_incorrect_logits():
    logits = torch.tensor([[12.0, -12.0], [12.0, -12.0]])
    targets = torch.tensor([0, 1])

    loss = FocalLoss()(logits, targets)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_loss_uses_mean_reduction():
    logits = torch.tensor([[0.3, -0.1], [-0.2, 0.5], [1.1, -0.7]])
    targets = torch.tensor([0, 1, 1])
    loss_fn = FocalLoss(gamma=2.0, alpha=0.75)

    actual = loss_fn(logits, targets)
    log_probabilities = F.log_softmax(logits, dim=1)
    target_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_probabilities = target_log_probabilities.exp()
    expected = (-0.75 * (1.0 - target_probabilities).pow(2.0) * target_log_probabilities).mean()

    assert torch.allclose(actual, expected)


def test_focal_loss_matches_cross_entropy_when_gamma_zero_and_alpha_one():
    logits = torch.tensor([[0.3, -0.1], [-0.2, 0.5], [1.1, -0.7]])
    targets = torch.tensor([0, 1, 1])

    focal_loss = FocalLoss(gamma=0.0, alpha=1.0)(logits, targets)
    cross_entropy = F.cross_entropy(logits, targets)

    assert torch.allclose(focal_loss, cross_entropy)


def test_identity_tile_permutation_has_zero_hardness_metrics():
    tiles_per_side = 3
    tile_permutation = identity_tile_permutation(tiles_per_side)

    assert compute_global_displacement(tile_permutation, tiles_per_side) == 0.0
    assert compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side) == 0.0
    assert compute_spatial_permutation_entropy(tile_permutation, tiles_per_side) == 0.0
    assert compute_combined_hardness(
        adjacency_destruction_hardness=0.0,
        spatial_permutation_entropy=0.0,
        global_tile_displacement=0.0,
    ) == 0.0


def test_nontrivial_tile_permutation_has_bounded_hardness_metrics():
    tiles_per_side = 3
    tile_permutation = TilePermutation(
        tiles_per_side=tiles_per_side,
        order=[
            [(2, 2), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (0, 0)],
        ],
    )

    assert 0.0 < compute_global_displacement(tile_permutation, tiles_per_side) <= 1.0
    assert 0.0 < compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side) <= 1.0
    assert 0.0 < compute_spatial_permutation_entropy(tile_permutation, tiles_per_side) <= 1.0
    assert 0.0 < compute_combined_hardness(
        adjacency_destruction_hardness=compute_adjacency_destruction_hardness(tile_permutation, tiles_per_side),
        spatial_permutation_entropy=compute_spatial_permutation_entropy(tile_permutation, tiles_per_side),
        global_tile_displacement=compute_global_displacement(tile_permutation, tiles_per_side),
    ) <= 1.0
