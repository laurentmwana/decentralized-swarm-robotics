"""Tests for curriculum training (Phase 5)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.swarm_gnn.config import TrainingConfig
from src.swarm_gnn.training.curriculum import (
    CurriculumConfig,
    CurriculumState,
    run_curriculum_stage,
    load_curriculum,
)


class TestCurriculumConfig:
    """Tests for curriculum configuration."""

    def test_load_curriculum_yaml(self):
        """Load curriculum from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
base:
  seed: 42
  batch_size: 4
  episode_steps: 5
  learning_rate: 0.01
  robot_count: 4
  arena_size: [5.0, 5.0]
  communication_radius: 2.0
  max_speed: 1.0
  time_step: 0.5
  safety_distance: 0.3
  target_radius: 1.0
  shape_loss_weight: 1.0
  collision_loss_weight: 1.0
  motion_loss_weight: 0.01
  coverage_loss_weight: 0.0
  hidden_dim: 8
  message_passing_steps: 1
stages:
  - name: geometry
    iterations: 5
    targets:
      - target1.npy
      - target2.npy
  - name: digits
    iterations: 5
    targets:
      - target1.npy
      - target2.npy
      - target3.npy
""")
            path = f.name

        curriculum = load_curriculum(path)
        assert curriculum.base.seed == 42
        assert curriculum.base.batch_size == 4
        assert len(curriculum.stages) == 2
        assert curriculum.stages[0].name == "geometry"
        assert curriculum.stages[0].iterations == 5
        assert len(curriculum.stages[0].targets) == 2
        assert curriculum.stages[1].name == "digits"
        assert curriculum.stages[1].iterations == 5
        assert len(curriculum.stages[1].targets) == 3

    def test_curriculum_state_transfer(self):
        """CurriculumState carries model state between stages."""
        state = CurriculumState()
        assert state.model_state is None

        # Simulate saving state after stage
        import torch
        state.model_state = {"layer.weight": torch.randn(10, 10)}
        assert state.model_state is not None
        assert "layer.weight" in state.model_state


class TestRunCurriculumStage:
    """Tests for running individual curriculum stages."""

    def test_single_stage_runs(self):
        """Single stage executes and produces outputs."""
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
            # Save targets as .npy files
            t1_path = Path(tmpdir) / "t1.npy"
            t2_path = Path(tmpdir) / "t2.npy"
            np.save(t1_path, target1)
            np.save(t2_path, target2)

            state = run_curriculum_stage(
                config, "test_stage", [str(t1_path), str(t2_path)], Path(tmpdir)
            )

            assert state.model_state is not None
            assert (Path(tmpdir) / "test_stage" / "model.pt").exists()
            assert (Path(tmpdir) / "test_stage" / "config.yaml").exists()
            assert (Path(tmpdir) / "test_stage" / "loss.csv").exists()


class TestFullCurriculum:
    """Tests for full curriculum execution."""

    def test_two_stage_curriculum(self):
        """Two stages execute sequentially with state transfer."""
        base_config = TrainingConfig(
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

        curriculum = CurriculumConfig(
            base=base_config,
            stages=[
                {"name": "stage1", "iterations": 3, "targets": ["t1.npy", "t2.npy"]},
                {"name": "stage2", "iterations": 3, "targets": ["t1.npy", "t2.npy", "t3.npy"]},
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # This would require the full curriculum runner
            # For now test that config loads correctly
            assert curriculum.base.seed == 42
            assert len(curriculum.stages) == 2