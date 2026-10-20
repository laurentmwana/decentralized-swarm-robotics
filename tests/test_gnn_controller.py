import pytest
import torch

from src.models import GNNController, NoCommunicationController


def sample_inputs(robot_count: int):
    torch.manual_seed(5)
    positions = torch.rand(1, robot_count, 2)
    velocities = torch.rand(1, robot_count, 2)
    adjacency = torch.ones(1, robot_count, robot_count, dtype=torch.bool)
    adjacency[:, torch.arange(robot_count), torch.arange(robot_count)] = False
    return positions, velocities, adjacency, torch.tensor([1])


@pytest.mark.parametrize("robot_count", [5, 10, 50])
def test_gnn_accepts_variable_swarm_sizes(robot_count: int) -> None:
    model = GNNController(hidden_dim=16, message_passing_steps=2, target_count=4)
    output = model(*sample_inputs(robot_count))
    assert output.shape == (1, robot_count, 2)
    assert torch.all(torch.linalg.vector_norm(output, dim=-1) <= 1.0 + 1e-6)


def test_robot_permutation_only_permutates_commands() -> None:
    model = GNNController(hidden_dim=16, message_passing_steps=2, target_count=4).eval()
    positions, velocities, adjacency, target = sample_inputs(7)
    permutation = torch.tensor([4, 2, 6, 0, 5, 1, 3])
    expected = model(positions, velocities, adjacency, target)[:, permutation]
    actual = model(positions[:, permutation], velocities[:, permutation], adjacency[:, permutation][:, :, permutation], target)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_global_translation_does_not_change_commands() -> None:
    model = GNNController(hidden_dim=16, message_passing_steps=1, target_count=4).eval()
    positions, velocities, adjacency, target = sample_inputs(6)
    assert torch.allclose(model(positions, velocities, adjacency, target), model(positions + 20, velocities, adjacency, target), atol=1e-6)


def test_goal_relative_position_can_condition_the_controller() -> None:
    model = GNNController(hidden_dim=16, message_passing_steps=1, target_count=4)
    positions, velocities, adjacency, target = sample_inputs(4)
    assert model(positions, velocities, adjacency, target, positions - 0.5).shape == (1, 4, 2)


def test_target_geometry_features_can_condition_the_controller() -> None:
    model = GNNController(hidden_dim=16, message_passing_steps=1, target_count=4)
    positions, velocities, adjacency, target = sample_inputs(4)
    geometry = torch.rand(1, 4, 3)
    assert model(positions, velocities, adjacency, target, positions - 0.5, geometry).shape == (1, 4, 2)


def test_isolated_robots_and_no_communication_baseline_are_valid() -> None:
    model = GNNController(hidden_dim=16, message_passing_steps=1, target_count=4)
    positions, velocities, adjacency, target = sample_inputs(4)
    isolated = torch.zeros_like(adjacency)
    assert torch.isfinite(model(positions, velocities, isolated, target)).all()
    assert NoCommunicationController(hidden_dim=16, target_count=4)(velocities, target).shape == (1, 4, 2)
