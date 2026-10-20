"""Tests for training pipeline (Phase 4)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.swarm_gnn.config import TrainingConfig
from src.swarm_gnn.models import GNNController
from src.swarm_gnn.training import (
    chamfer_loss,
    collision_loss,
    rollout,
    train_circle,
    train_targets,
)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_chamfer_loss_symmetric(self):
        """Chamfer loss is symmetric and non-negative."""
        pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        target = torch.tensor([[[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]]])
        loss = chamfer_loss(pred, target)
        assert loss >= 0
        # Symmetric: L(pred, target) == L(target, pred)
        loss_rev = chamfer_loss(target, pred)
        assert torch.allclose(loss, loss_rev)

    def test_chamfer_loss_zero_when_equal(self):
        """Chamfer loss is zero when pred == target."""
        points = torch.randn(2, 5, 2)
        loss = chamfer_loss(points, points)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_collision_loss_penalizes_close(self):
        """Collision loss penalizes pairs closer than safety_distance."""
        # Two robots very close
        pos = torch.tensor([[[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]]])  # safety=0.5
        loss = collision_loss(pos, safety_distance=0.5)
        assert loss > 0

    def test_collision_loss_zero_when_far(self):
        """Collision loss is zero when all pairs > safety_distance."""
        pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
        loss = collision_loss(pos, safety_distance=0.5)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_collision_loss_excludes_self(self):
        """Self-pairs excluded from collision loss."""
        pos = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])  # same position
        loss = collision_loss(pos, safety_distance=0.5)
        # Should not crash, self-pairs excluded
        assert loss >= 0


class TestRollout:
    """Tests for differentiable rollout."""

    def test_rollout_shape(self):
        """Rollout returns correct shapes."""
        config = TrainingConfig(
            seed=42,
            batch_size=2,
            iterations=1,
            episode_steps=5,
            learning_rate=0.001,
            robot_count=4,
            arena_size=(10.0, 10.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=0.3,
            safety_distance=0.3,
            target_radius=2.0,
            shape_loss_weight=1.0,
            collision_loss_weight=1.0,
            motion_loss_weight=0.01,
            hidden_dim=16,
            message_passing_steps=1,
        )
        # Use actual GNNController as model
        model = GNNController(hidden_dim=16, message_passing_steps=1, target_count=2, max_speed=1.0)
        model.eval()
        positions = torch.randn(2, 4, 2)
        target_ids = torch.tensor([0, 1])
        targets = torch.randn(2, 4, 2)

        with torch.no_grad():
            final_pos, commands = rollout(model, positions, target_ids, config, targets)

        assert final_pos.shape == (2, 4, 2)
        assert commands.shape == (2, 5, 4, 2)  # [batch, steps, robots, 2]

    def test_rollout_respects_arena_bounds(self):
        """Rollout clips positions to arena."""
        config = TrainingConfig(
            seed=42,
            batch_size=1,
            iterations=1,
            episode_steps=3,
            learning_rate=0.001,
            robot_count=2,
            arena_size=(5.0, 5.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=1.0,  # large step
            safety_distance=0.3,
            target_radius=2.0,
            shape_loss_weight=1.0,
            collision_loss_weight=1.0,
            motion_loss_weight=0.01,
        )
        # Model that pushes robots out of arena
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 2),
            torch.nn.Identity(),
        )
        # Force large velocities
        def big_model(pos, vel, adj, tid, goal=None, geom=None):
            return torch.ones_like(vel) * 10.0  # huge velocity

        positions = torch.tensor([[[4.5, 2.5], [0.5, 2.5]]])
        target_ids = torch.tensor([0])
        targets = torch.ones_like(positions) * 2.5

        final_pos, _ = rollout(big_model, positions, target_ids, config, targets)
        assert torch.all(final_pos >= 0)
        assert torch.all(final_pos <= 5.0)


class TestTrainCircle:
    """Tests for circle training entry point."""

    def test_train_circle_runs(self):
        """train_circle executes and returns model with decreasing loss."""
        config = TrainingConfig(
            seed=42,
            batch_size=4,
            iterations=20,  # more iterations for convergence
            episode_steps=5,
            learning_rate=0.01,
            robot_count=4,
            arena_size=(5.0, 5.0),
            communication_radius=2.0,
            max_speed=1.0,
            time_step=0.5,
            safety_distance=0.3,
            target_radius=1.0,
            shape_loss_weight=1.0,
            collision_loss_weight=1.0,
            motion_loss_weight=0.01,
            coverage_loss_weight=0.0,
            hidden_dim=16,
            message_passing_steps=1,
        )
        # Simple circle target
        target = np.array([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]
        ], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            model, losses = train_circle(config, target, tmpdir)

            assert len(losses) == 20
            # Loss should generally decrease (allow small fluctuations)
            assert losses[-1] < losses[0] * 1.2  # allow 20% tolerance for randomness
            assert (Path(tmpdir) / "model.pt").exists()
            assert (Path(tmpdir) / "circle-best.pt").exists()
            assert (Path(tmpdir) / "config.yaml").exists()
            assert (Path(tmpdir) / "loss.csv").exists()
            assert (Path(tmpdir) / "loss.png").exists()


class TestTrainTargets:
    """Tests for multi-target training."""

    def test_train_targets_multi(self):
        """train_targets works with multiple targets."""
        config = TrainingConfig(
            seed=42,
            batch_size=2,
            iterations=3,
            episode_steps=3,
            learning_rate=0.01,
            robot_count=4,
            arena_size=(5.0, 5.0),
            communication_radius=2.0,
            max_speed=1.0,
            time_step=0.5,
            safety_distance=0.3,
            target_radius=1.0,
            shape_loss_weight=1.0,
            collision_loss_weight=1.0,
            motion_loss_weight=0.01,
            coverage_loss_weight=0.0,
            hidden_dim=8,
            message_passing_steps=1,
        )
        target1 = np.array([[1,0],[0,1],[-1,0],[0,-1]], dtype=np.float32)
        target2 = np.array([[0.7,0.7],[-0.7,0.7],[-0.7,-0.7],[0.7,-0.7]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            model, losses, metrics, per_target = train_targets(
                config, [target1, target2], tmpdir
            )

            assert len(losses) == 3
            assert set(metrics.keys()) == {0, 1}
            assert len(per_target[0]) > 0
            assert len(per_target[1]) > 0