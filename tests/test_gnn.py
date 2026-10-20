"""Tests for GNN Controller (Phase 3)."""

import pytest
import torch

from src.swarm_gnn.models import GNNController, NoCommunicationController


class TestGNNController:
    """Tests for GNNController."""

    @pytest.fixture
    def controller(self):
        return GNNController(
            hidden_dim=32,
            message_passing_steps=2,
            target_count=4,
            max_speed=1.0,
        )

    @pytest.fixture
    def batch_data(self):
        """Batch of 2 episodes, 5 robots each."""
        batch_size = 2
        robot_count = 5
        return {
            "positions": torch.randn(batch_size, robot_count, 2),
            "velocities": torch.randn(batch_size, robot_count, 2),
            "adjacency": torch.randint(0, 2, (batch_size, robot_count, robot_count)).bool(),
            "target_ids": torch.randint(0, 4, (batch_size,)),
        }

    def test_init_valid_params(self):
        """Valid parameters create controller."""
        ctrl = GNNController(
            hidden_dim=32,
            message_passing_steps=2,
            target_count=4,
            max_speed=1.0,
        )
        assert ctrl.max_speed == 1.0
        assert ctrl.target_embedding.weight.shape == (4, 32)

    def test_init_invalid_params_raises(self):
        """Invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            GNNController(hidden_dim=32, message_passing_steps=0, target_count=4, max_speed=1.0)
        with pytest.raises(ValueError):
            GNNController(hidden_dim=32, message_passing_steps=2, target_count=0, max_speed=1.0)
        with pytest.raises(ValueError):
            GNNController(hidden_dim=32, message_passing_steps=2, target_count=4, max_speed=0.0)

    def test_forward_shape(self, controller, batch_data):
        """Forward returns [batch, robots, 2]."""
        out = controller(
            positions=batch_data["positions"],
            velocities=batch_data["velocities"],
            adjacency=batch_data["adjacency"],
            target_ids=batch_data["target_ids"],
        )
        assert out.shape == (2, 5, 2)

    def test_forward_speed_bounded(self, controller, batch_data):
        """Output commands bounded by max_speed."""
        out = controller(
            positions=batch_data["positions"],
            velocities=batch_data["velocities"],
            adjacency=batch_data["adjacency"],
            target_ids=batch_data["target_ids"],
        )
        speeds = torch.linalg.vector_norm(out, dim=-1)
        assert torch.all(speeds <= controller.max_speed + 1e-6)

    def test_forward_goal_relative_positions(self, controller, batch_data):
        """Forward accepts goal_relative_positions."""
        goal_rel = torch.randn_like(batch_data["positions"])
        out = controller(
            positions=batch_data["positions"],
            velocities=batch_data["velocities"],
            adjacency=batch_data["adjacency"],
            target_ids=batch_data["target_ids"],
            goal_relative_positions=goal_rel,
        )
        assert out.shape == (2, 5, 2)

    def test_forward_target_geometry(self, controller, batch_data):
        """Forward accepts target_geometry_features."""
        geometry = torch.randn(2, 5, 3)  # [batch, robots, 3]
        out = controller(
            positions=batch_data["positions"],
            velocities=batch_data["velocities"],
            adjacency=batch_data["adjacency"],
            target_ids=batch_data["target_ids"],
            target_geometry_features=geometry,
        )
        assert out.shape == (2, 5, 2)

    def test_forward_invalid_positions_raises(self, controller, batch_data):
        """Invalid positions shape raises ValueError."""
        with pytest.raises(ValueError, match="positions et velocities doivent être"):
            controller(
                positions=batch_data["positions"][0],  # missing batch dim
                velocities=batch_data["velocities"],
                adjacency=batch_data["adjacency"],
                target_ids=batch_data["target_ids"],
            )

    def test_forward_invalid_adjacency_raises(self, controller, batch_data):
        """Invalid adjacency shape raises ValueError."""
        with pytest.raises(ValueError, match="adjacency doit être"):
            controller(
                positions=batch_data["positions"],
                velocities=batch_data["velocities"],
                adjacency=batch_data["adjacency"][0],  # missing batch dim
                target_ids=batch_data["target_ids"],
            )

    def test_forward_invalid_target_ids_raises(self, controller, batch_data):
        """Invalid target_ids shape raises ValueError."""
        with pytest.raises(ValueError, match="target_ids doit être"):
            controller(
                positions=batch_data["positions"],
                velocities=batch_data["velocities"],
                adjacency=batch_data["adjacency"],
                target_ids=batch_data["target_ids"][0].unsqueeze(0).expand(3),  # wrong batch size
            )

    def test_permutation_equivariance(self, controller):
        """Output equivariant to robot permutation."""
        # Create symmetric situation: 3 robots in line
        positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]], dtype=torch.float32)
        velocities = torch.zeros(1, 3, 2)
        adjacency = torch.ones(1, 3, 3, dtype=torch.bool)
        # Remove self-loops properly for batched tensor
        adjacency = adjacency & ~torch.eye(3, dtype=torch.bool).unsqueeze(0)
        target_ids = torch.tensor([0])

        out1 = controller(positions, velocities, adjacency, target_ids)

        # Permute robots 0 and 2
        perm = torch.tensor([2, 1, 0])
        positions_perm = positions[:, perm, :]
        velocities_perm = velocities[:, perm, :]
        adjacency_perm = adjacency[:, perm, :][:, :, perm]

        out2 = controller(positions_perm, velocities_perm, adjacency_perm, target_ids)

        # Output should be permuted same way
        out1_perm = out1[:, perm, :]
        assert torch.allclose(out1_perm, out2, atol=1e-6)

    def test_variable_robot_count(self):
        """Controller handles variable robot counts."""
        ctrl = GNNController(hidden_dim=16, message_passing_steps=1, target_count=2, max_speed=1.0)

        for n_robots in [2, 5, 10, 20]:
            positions = torch.randn(1, n_robots, 2)
            velocities = torch.randn(1, n_robots, 2)
            adjacency = torch.ones(1, n_robots, n_robots, dtype=torch.bool)
            # Remove self-loops
            adjacency = adjacency & ~torch.eye(n_robots, dtype=torch.bool).unsqueeze(0)
            target_ids = torch.tensor([0])

            out = ctrl(positions, velocities, adjacency, target_ids)
            assert out.shape == (1, n_robots, 2)

    def test_deterministic_same_inputs(self, controller, batch_data):
        """Same inputs produce same outputs."""
        out1 = controller(**batch_data)
        out2 = controller(**batch_data)
        assert torch.allclose(out1, out2)


class TestNoCommunicationController:
    """Tests for baseline NoCommunicationController."""

    @pytest.fixture
    def controller(self):
        return NoCommunicationController(
            hidden_dim=32,
            target_count=4,
            max_speed=1.0,
        )

    def test_forward_no_adjacency(self, controller):
        """Forward works without adjacency matrix."""
        batch_size = 2
        robot_count = 5
        velocities = torch.randn(batch_size, robot_count, 2)
        target_ids = torch.randint(0, 4, (batch_size,))

        out = controller(
            velocities=velocities,
            target_ids=target_ids,
        )
        assert out.shape == (batch_size, robot_count, 2)

    def test_speed_bounded(self, controller):
        """Output speed bounded by max_speed."""
        velocities = torch.randn(1, 5, 2) * 10  # large velocities
        target_ids = torch.tensor([0])
        out = controller(velocities=velocities, target_ids=target_ids)
        speeds = torch.linalg.vector_norm(out, dim=-1)
        assert torch.all(speeds <= controller.max_speed + 1e-6)

    def test_variable_robot_count(self):
        """Handles variable robot counts."""
        ctrl = NoCommunicationController(hidden_dim=16, target_count=2, max_speed=1.0)
        for n_robots in [2, 5, 10]:
            velocities = torch.randn(1, n_robots, 2)
            target_ids = torch.tensor([0])
            out = ctrl(velocities=velocities, target_ids=target_ids)
            assert out.shape == (1, n_robots, 2)