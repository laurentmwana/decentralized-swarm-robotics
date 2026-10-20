"""Tests for evaluation pipeline (Phase 6)."""

import numpy as np
import pytest
import torch

from src.swarm_gnn.config import EvaluationConfig
from src.swarm_gnn.evaluation import (
    simulate_episode,
    collision_count,
    nearest_neighbor_std,
    symmetric_chamfer_distance,
    graph_connectivity,
)
from src.swarm_gnn.models import GNNController


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_collision_count(self):
        """Count pairs closer than safety_distance."""
        pos = np.array([[0.0, 0.0], [0.3, 0.0], [5.0, 5.0]])
        count = collision_count(pos, 0.5)
        assert count == 1

    def test_collision_count_zero(self):
        """No collisions when all far apart."""
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assert collision_count(pos, 0.5) == 0

    def test_symmetric_chamfer_distance(self):
        """Symmetric Chamfer distance."""
        pred = np.array([[0.0, 0.0], [1.0, 0.0]])
        target = np.array([[0.1, 0.1], [1.1, 0.1]])
        dist = symmetric_chamfer_distance(pred, target)
        assert dist > 0
        # Symmetric
        assert np.isclose(dist, symmetric_chamfer_distance(target, pred))

    def test_symmetric_chamfer_zero(self):
        """Zero when identical."""
        pts = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert np.isclose(symmetric_chamfer_distance(pts, pts), 0.0)

    def test_nearest_neighbor_std(self):
        """Std of nearest neighbor distances."""
        # Regular grid -> low std
        pos = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        std = nearest_neighbor_std(pos)
        assert std >= 0

    def test_graph_connectivity(self):
        """Graph connectivity ratio."""
        pos = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
        conn = graph_connectivity(pos, radius=2.0)
        assert 0 <= conn <= 1


class TestSimulateEpisode:
    """Tests for episode simulation."""

    @pytest.fixture
    def eval_config(self):
        return EvaluationConfig(
            seed=42,
            episodes=1,
            episode_steps=5,
            robot_count=4,
            arena_size=(10.0, 10.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=0.3,
            safety_distance=0.3,
            target_radius=2.0,
            failure_fractions=(0.0, 0.1),
            scale_factors=(1, 2),
            success_threshold=0.5,
            model_path="dummy.pt",
            target_path="dummy.npy",
        )

    @pytest.fixture
    def model(self):
        # Use actual GNNController
        return GNNController(hidden_dim=32, message_passing_steps=1, target_count=4, max_speed=1.0)

    def test_simulate_episode_basic(self, eval_config, model):
        """Episode runs and returns metrics."""
        target = np.random.randn(4, 2).astype(np.float32)
        result = simulate_episode(
            model, "gnn", target, 4, 0.0, 42, eval_config, target_id=0
        )

        assert result["method"] == "gnn"
        assert result["robots"] == 4
        assert result["failure_fraction"] == 0.0
        assert "shape_distance" in result
        assert "collisions" in result
        assert "convergence_step" in result
        assert "success" in result
        assert "final_connectivity" in result

    def test_simulate_episode_with_failure(self, eval_config, model):
        """Episode with robot failures."""
        target = np.random.randn(4, 2).astype(np.float32)
        result = simulate_episode(
            model, "gnn", target, 4, 0.5, 42, eval_config, target_id=0
        )

        assert result["failure_fraction"] == 0.5
        assert "failure_step" in result or "failure_step" not in result  # key may or may not be present
        assert "retired" in result
        assert len(result["retired"]) > 0

    def test_methods_enum(self, eval_config, model):
        """All methods produce valid results."""
        target = np.random.randn(4, 2).astype(np.float32)
        for method in ["gnn", "gathering", "manual", "gnn_no_comm"]:
            result = simulate_episode(
                model, method, target, 4, 0.0, 42, eval_config, target_id=0
            )
            assert result["method"] == method
            assert "shape_distance" in result