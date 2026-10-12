import numpy as np
import torch
from pathlib import Path

from src.train.training import TrainingConfig, adjacency_from_positions, chamfer_loss, collision_loss, coverage_loss, local_target_features, train_targets


def test_losses_are_zero_or_small_on_simple_expected_cases() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    assert chamfer_loss(positions, positions).item() == 0.0
    assert collision_loss(positions, safety_distance=0.2).item() == 0.0


def test_adjacency_excludes_self_and_respects_radius() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]])
    adjacency = adjacency_from_positions(positions, radius=1.5)
    assert adjacency.tolist() == [[[False, True, False], [True, False, False], [False, False, False]]]


def test_target_features_and_coverage_match_an_identical_target() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    features = local_target_features(positions, positions, scale=1.0)
    assert torch.allclose(features, torch.zeros_like(features))
    assert coverage_loss(positions, positions).item() == 0.0


def test_training_accepts_multiple_target_conditions() -> None:
    config = TrainingConfig(2, 2, 1, 1, 0.001, 2, (2.0, 2.0), 2.0, 1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.0, 8, 1)
    targets = [np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([[0.0, -1.0], [0.0, 1.0]])]
    model, losses, metrics = train_targets(config, targets, Path("results/test-target-conditions"))
    assert len(losses) == 1
    assert set(metrics) == {0, 1}
    assert model.target_embedding.num_embeddings == 2
