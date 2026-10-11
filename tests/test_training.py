import torch

from src.train.training import TrainingConfig, adjacency_from_positions, chamfer_loss, collision_loss


def test_losses_are_zero_or_small_on_simple_expected_cases() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    assert chamfer_loss(positions, positions).item() == 0.0
    assert collision_loss(positions, safety_distance=0.2).item() == 0.0


def test_adjacency_excludes_self_and_respects_radius() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]])
    adjacency = adjacency_from_positions(positions, radius=1.5)
    assert adjacency.tolist() == [[[False, True, False], [True, False, False], [False, False, False]]]
