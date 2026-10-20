"""Curriculum training: progressive shape learning with state transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import yaml

from src.swarm_gnn.config import TrainingConfig
from src.swarm_gnn.models import GNNController
from src.swarm_gnn.training import train_targets, TrainingConfig as TrainConfig


@dataclass(frozen=True)
class StageConfig:
    """Single curriculum stage configuration."""
    name: str
    iterations: int
    targets: List[str]  # paths to .npy files


@dataclass(frozen=True)
class CurriculumConfig:
    """Full curriculum configuration."""
    base: TrainingConfig
    stages: List[StageConfig]

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CurriculumConfig":
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        base_data = data["base"]
        base_data["arena_size"] = tuple(base_data["arena_size"])
        base_config = TrainingConfig(**base_data)

        stages = []
        for s in data["stages"]:
            stages.append(StageConfig(
                name=s["name"],
                iterations=s["iterations"],
                targets=s["targets"],
            ))
        return cls(base=base_config, stages=stages)


def load_curriculum(path: Union[str, Path]) -> CurriculumConfig:
    """Load curriculum configuration from YAML file."""
    return CurriculumConfig.from_yaml(path)


@dataclass
class CurriculumState:
    """Carries model state between curriculum stages."""
    model_state: Optional[dict] = None

    def create_model(self, hidden_dim: int, message_passing_steps: int,
                     target_count: int, max_speed: float) -> GNNController:
        """Create model and load previous state if available."""
        model = GNNController(hidden_dim, message_passing_steps, target_count, max_speed)
        if self.model_state is not None:
            # Load compatible weights
            current = model.state_dict()
            compatible = {k: v for k, v in self.model_state.items()
                          if k in current and current[k].shape == v.shape}
            model.load_state_dict(compatible, strict=False)
            # Handle target embedding specially
            if "target_embedding.weight" in self.model_state:
                prev_emb = self.model_state["target_embedding.weight"]
                if prev_emb.shape[1] == model.target_embedding.weight.shape[1]:
                    model.target_embedding.weight.data[:min(len(prev_emb), target_count)] = \
                        prev_emb[:target_count]
        return model

    def update(self, model: GNNController) -> None:
        """Save current model state."""
        self.model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}


def load_targets(paths: List[str], target_radius: float, arena_size: Tuple[float, float]) -> List[np.ndarray]:
    """Load and scale target points from .npy files."""
    targets = []
    arena = np.asarray(arena_size)
    for p in paths:
        pts = np.load(p) * target_radius + arena / 2
        targets.append(pts)
    return targets


def run_curriculum_stage(
    base_config: TrainingConfig,
    stage_name: str,
    target_paths: List[str],
    output_dir: Path,
    state: Optional[CurriculumState] = None,
) -> CurriculumState:
    """
    Run a single curriculum stage.

    Args:
        base_config: Base training configuration
        stage_name: Name for output subdirectory
        target_paths: List of paths to target .npy files
        output_dir: Root output directory
        state: Previous curriculum state (for weight transfer)

    Returns:
        Updated CurriculumState with new model weights
    """
    if state is None:
        state = CurriculumState()

    stage_output = output_dir / stage_name
    stage_output.mkdir(parents=True, exist_ok=True)

    # Load and scale targets
    targets = load_targets(target_paths, base_config.target_radius, base_config.arena_size)

    # Create model with state transfer
    model = state.create_model(
        base_config.hidden_dim,
        base_config.message_passing_steps,
        len(targets),
        base_config.max_speed,
    )

    # Create stage-specific config
    stage_config = TrainConfig(
        seed=base_config.seed,
        batch_size=base_config.batch_size,
        iterations=base_config.iterations,
        episode_steps=base_config.episode_steps,
        learning_rate=base_config.learning_rate,
        robot_count=base_config.robot_count,
        arena_size=base_config.arena_size,
        communication_radius=base_config.communication_radius,
        max_speed=base_config.max_speed,
        time_step=base_config.time_step,
        safety_distance=base_config.safety_distance,
        target_radius=base_config.target_radius,
        shape_loss_weight=base_config.shape_loss_weight,
        collision_loss_weight=base_config.collision_loss_weight,
        motion_loss_weight=base_config.motion_loss_weight,
        coverage_loss_weight=base_config.coverage_loss_weight,
        hidden_dim=base_config.hidden_dim,
        message_passing_steps=base_config.message_passing_steps,
        use_target_geometry=base_config.use_target_geometry,
    )

    # Train
    model, losses, metrics, per_target = train_targets(
        stage_config, targets, stage_output, model.state_dict()
    )

    # Save stage info
    (stage_output / "targets.yaml").write_text(
        yaml.safe_dump({"targets": target_paths}, sort_keys=False), encoding="utf-8"
    )

    # Update state
    state.update(model)

    return state