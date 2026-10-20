"""Configuration system for swarm-gnn.

All configs are frozen dataclasses with validation, loaded from YAML.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple, Union

import yaml


@dataclass(frozen=True)
class SimulatorConfig:
    """Reproducible simulation episode parameters."""

    seed: int
    robot_count: int
    arena_size: Tuple[float, float]
    communication_radius: float
    max_speed: float
    time_step: float
    episode_steps: int
    safety_distance: float
    measurement_noise_std: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.robot_count <= 0:
            raise ValueError("robot_count doit être strictement positif.")
        if len(self.arena_size) != 2 or min(self.arena_size) <= 0:
            raise ValueError("arena_size doit contenir deux dimensions positives.")
        for name in ("communication_radius", "max_speed", "time_step", "safety_distance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} doit être strictement positif.")
        if self.episode_steps <= 0 or self.measurement_noise_std < 0:
            raise ValueError("Paramètres d'épisode ou de bruit invalides.")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SimulatorConfig":
        """Load config from YAML file."""
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        values["arena_size"] = tuple(values["arena_size"])
        return cls(**values)


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration (includes all simulator params + training-specific)."""

    # Simulator params
    seed: int
    robot_count: int
    arena_size: Tuple[float, float]
    communication_radius: float
    max_speed: float
    time_step: float
    episode_steps: int
    safety_distance: float
    measurement_noise_std: float = field(default=0.0)

    # Training params
    batch_size: int = field(default=8)
    iterations: int = field(default=100)
    learning_rate: float = field(default=0.001)
    target_radius: float = field(default=2.0)
    shape_loss_weight: float = field(default=1.0)
    collision_loss_weight: float = field(default=1.0)
    motion_loss_weight: float = field(default=0.01)
    coverage_loss_weight: float = field(default=0.0)
    hidden_dim: int = field(default=32)
    message_passing_steps: int = field(default=2)
    use_target_geometry: bool = field(default=False)
    transport_loss_weight: float = field(default=0.0)
    transport_epsilon: float = field(default=0.2)
    transport_iterations: int = field(default=30)
    use_assigned_targets: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.robot_count <= 0:
            raise ValueError("robot_count doit être strictement positif.")
        if len(self.arena_size) != 2 or min(self.arena_size) <= 0:
            raise ValueError("arena_size doit contenir deux dimensions positives.")
        for name in ("communication_radius", "max_speed", "time_step", "safety_distance"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} doit être strictement positif.")
        if self.episode_steps <= 0 or self.measurement_noise_std < 0:
            raise ValueError("Paramètres d'épisode ou de bruit invalides.")
        if self.batch_size <= 0:
            raise ValueError("batch_size doit être positif.")
        if self.iterations <= 0:
            raise ValueError("iterations doit être positif.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate doit être positif.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim doit être positif.")
        if self.message_passing_steps <= 0:
            raise ValueError("message_passing_steps doit être positif.")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainingConfig":
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        values["arena_size"] = tuple(values["arena_size"])
        return cls(**values)

    def to_simulator_config(self) -> "SimulatorConfig":
        """Extract simulator config for use in simulation."""
        return SimulatorConfig(
            seed=self.seed,
            robot_count=self.robot_count,
            arena_size=self.arena_size,
            communication_radius=self.communication_radius,
            max_speed=self.max_speed,
            time_step=self.time_step,
            episode_steps=self.episode_steps,
            safety_distance=self.safety_distance,
            measurement_noise_std=self.measurement_noise_std,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation protocol configuration."""

    seed: int
    episodes: int
    episode_steps: int
    robot_count: int
    arena_size: Tuple[float, float]
    communication_radius: float
    max_speed: float
    time_step: float
    safety_distance: float
    target_radius: float
    failure_fractions: Tuple[float, ...]
    scale_factors: Tuple[int, ...]
    success_threshold: float
    model_path: str
    target_path: str
    use_target_geometry: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes doit être positif.")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvaluationConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        for key in ("arena_size", "failure_fractions", "scale_factors"):
            data[key] = tuple(data[key])
        return cls(**data)

    def to_simulator_config(self) -> "SimulatorConfig":
        """Extract simulator config for use in simulation."""
        return SimulatorConfig(
            seed=self.seed,
            robot_count=self.robot_count,
            arena_size=self.arena_size,
            communication_radius=self.communication_radius,
            max_speed=self.max_speed,
            time_step=self.time_step,
            episode_steps=self.episode_steps,
            safety_distance=self.safety_distance,
            measurement_noise_std=0.0,
        )


ConfigType = Literal["simulator", "training", "evaluation"]


def load_config(path: Union[str, Path], config_type: ConfigType) -> Union[SimulatorConfig, TrainingConfig, EvaluationConfig]:
    """Load config of specified type from YAML file."""
    if config_type == "simulator":
        return SimulatorConfig.from_yaml(path)
    elif config_type == "training":
        return TrainingConfig.from_yaml(path)
    elif config_type == "evaluation":
        return EvaluationConfig.from_yaml(path)
    else:
        raise ValueError(f"Unknown config type: {config_type}. Choose from: simulator, training, evaluation")