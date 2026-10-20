"""Tests for configuration system (Phase 0)."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.swarm_gnn.config import (
    SimulatorConfig,
    TrainingConfig,
    EvaluationConfig,
    load_config,
)


class TestSimulatorConfig:
    """Tests for SimulatorConfig."""

    def test_default_values(self):
        """SimulatorConfig has sensible defaults."""
        config = SimulatorConfig(
            seed=42,
            robot_count=20,
            arena_size=(10.0, 10.0),
            communication_radius=2.5,
            max_speed=1.0,
            time_step=0.3,
            episode_steps=20,
            safety_distance=0.3,
        )
        assert config.seed == 42
        assert config.robot_count == 20
        assert config.arena_size == (10.0, 10.0)
        assert config.communication_radius == 2.5
        assert config.max_speed == 1.0
        assert config.time_step == 0.3
        assert config.episode_steps == 20
        assert config.safety_distance == 0.3
        assert config.measurement_noise_std == 0.0  # default

    def test_validation_robot_count_positive(self):
        """robot_count must be positive."""
        with pytest.raises(ValueError):
            SimulatorConfig(
                seed=42,
                robot_count=0,
                arena_size=(10.0, 10.0),
                communication_radius=2.5,
                max_speed=1.0,
                time_step=0.3,
                episode_steps=20,
                safety_distance=0.3,
            )

    def test_validation_arena_size(self):
        """arena_size must have two positive dimensions."""
        with pytest.raises(ValueError):
            SimulatorConfig(
                seed=42,
                robot_count=20,
                arena_size=(10.0,),  # wrong length
                communication_radius=2.5,
                max_speed=1.0,
                time_step=0.3,
                episode_steps=20,
                safety_distance=0.3,
            )
        with pytest.raises(ValueError):
            SimulatorConfig(
                seed=42,
                robot_count=20,
                arena_size=(-10.0, 10.0),  # negative
                communication_radius=2.5,
                max_speed=1.0,
                time_step=0.3,
                episode_steps=20,
                safety_distance=0.3,
            )

    def test_validation_positive_params(self):
        """All radius/speed/time params must be positive."""
        for param in ["communication_radius", "max_speed", "time_step", "safety_distance"]:
            kwargs = dict(
                seed=42,
                robot_count=20,
                arena_size=(10.0, 10.0),
                communication_radius=2.5,
                max_speed=1.0,
                time_step=0.3,
                episode_steps=20,
                safety_distance=0.3,
            )
            kwargs[param] = -1.0
            with pytest.raises(ValueError, match=param):
                SimulatorConfig(**kwargs)

    def test_load_from_yaml(self):
        """Can load config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "seed": 123,
                "robot_count": 30,
                "arena_size": [15.0, 15.0],
                "communication_radius": 3.0,
                "max_speed": 1.5,
                "time_step": 0.5,
                "episode_steps": 30,
                "safety_distance": 0.5,
            }, f)
            path = Path(f.name)

        try:
            config = SimulatorConfig.from_yaml(path)
            assert config.seed == 123
            assert config.robot_count == 30
            assert config.arena_size == (15.0, 15.0)
            assert config.communication_radius == 3.0
        finally:
            path.unlink()

    def test_immutable(self):
        """Config is frozen/immutable."""
        config = SimulatorConfig(
            seed=42,
            robot_count=20,
            arena_size=(10.0, 10.0),
            communication_radius=2.5,
            max_speed=1.0,
            time_step=0.3,
            episode_steps=20,
            safety_distance=0.3,
        )
        with pytest.raises(Exception):
            config.seed = 999


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """TrainingConfig has sensible defaults extending SimulatorConfig."""
        config = TrainingConfig(
            seed=42,
            robot_count=20,
            arena_size=(10.0, 10.0),
            communication_radius=2.5,
            max_speed=1.0,
            time_step=0.3,
            episode_steps=20,
            safety_distance=0.3,
            batch_size=8,
            iterations=100,
            learning_rate=0.001,
            target_radius=2.0,
            shape_loss_weight=1.0,
            collision_loss_weight=1.0,
            motion_loss_weight=0.01,
            coverage_loss_weight=0.0,
            hidden_dim=32,
            message_passing_steps=2,
            use_target_geometry=False,
        )
        assert config.batch_size == 8
        assert config.iterations == 100
        assert config.learning_rate == 0.001
        assert config.hidden_dim == 32
        assert config.message_passing_steps == 2

    def test_load_from_yaml(self):
        """Can load training config from YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "seed": 42,
                "batch_size": 16,
                "iterations": 200,
                "episode_steps": 20,
                "learning_rate": 0.003,
                "robot_count": 20,
                "arena_size": [10.0, 10.0],
                "communication_radius": 2.5,
                "max_speed": 1.0,
                "time_step": 0.3,
                "safety_distance": 0.3,
                "target_radius": 2.0,
                "shape_loss_weight": 1.0,
                "collision_loss_weight": 1.0,
                "motion_loss_weight": 0.01,
                "coverage_loss_weight": 0.0,
                "hidden_dim": 32,
                "message_passing_steps": 2,
                "use_target_geometry": False,
            }, f)
            path = Path(f.name)

        try:
            config = TrainingConfig.from_yaml(path)
            assert config.batch_size == 16
            assert config.iterations == 200
            assert config.learning_rate == 0.003
        finally:
            path.unlink()


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_values(self):
        """EvaluationConfig has sensible defaults."""
        config = EvaluationConfig(
            seed=101,
            episodes=50,
            episode_steps=20,
            robot_count=20,
            arena_size=(10.0, 10.0),
            communication_radius=2.5,
            max_speed=1.0,
            time_step=0.3,
            safety_distance=0.3,
            target_radius=2.0,
            failure_fractions=(0.0, 0.05, 0.10, 0.20),
            scale_factors=(1, 2, 5),
            success_threshold=0.5,
            model_path="model.pt",
            target_path="target.npy",
        )
        assert config.episodes == 50
        assert config.failure_fractions == (0.0, 0.05, 0.10, 0.20)
        assert config.scale_factors == (1, 2, 5)
        assert config.success_threshold == 0.5


class TestLoadConfig:
    """Tests for load_config utility."""

    def test_load_config_returns_correct_type(self):
        """load_config returns correct config type based on path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "seed": 42,
                "robot_count": 20,
                "arena_size": [10.0, 10.0],
                "communication_radius": 2.5,
                "max_speed": 1.0,
                "time_step": 0.3,
                "episode_steps": 20,
                "safety_distance": 0.3,
            }, f)
            path = Path(f.name)

        try:
            config = load_config(path, "simulator")
            assert isinstance(config, SimulatorConfig)
        finally:
            path.unlink()

    def test_load_config_unknown_type_raises(self):
        """Unknown config type raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({}, f)
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Unknown config type"):
                load_config(path, "unknown")
        finally:
            path.unlink()