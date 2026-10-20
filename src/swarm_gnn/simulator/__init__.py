"""Discrete-time 2D swarm simulator with direct velocity commands.

Features:
- Direct velocity control with speed clipping
- Communication radius-based neighbor graph
- Robot failures (inactive robots frozen)
- Collision detection
- Measurement noise
- Deterministic RNG per seed
"""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Reproducible simulation episode parameters."""

    seed: int
    robot_count: int
    arena_size: Tuple[float, float]
    communication_radius: float
    max_speed: float
    time_step: float
    episode_steps: int
    safety_distance: float
    measurement_noise_std: float = 0.0

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


class SwarmSimulator:
    """Discrete-time simulator with direct velocity commands."""

    def __init__(
        self,
        config: SimulationConfig,
        positions: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.active = np.ones(config.robot_count, dtype=bool)
        self.velocities = np.zeros((config.robot_count, 2), dtype=float)
        self.step_count = 0
        self.positions = self._initial_positions() if positions is None else self._validate_positions(positions)

    def _initial_positions(self) -> np.ndarray:
        """Uniform random initial positions in arena."""
        return self.rng.uniform(
            (0.0, 0.0),
            self.config.arena_size,
            size=(self.config.robot_count, 2),
        )

    def _validate_positions(self, positions: np.ndarray) -> np.ndarray:
        """Validate and clip positions to arena."""
        values = np.asarray(positions, dtype=float).copy()
        if values.shape != (self.config.robot_count, 2):
            raise ValueError("positions doit être de forme (robot_count, 2).")
        return np.clip(values, (0.0, 0.0), self.config.arena_size)

    def reset(self, positions: np.ndarray | None = None) -> None:
        """Reset to initial state with same seed."""
        self.rng = np.random.default_rng(self.config.seed)
        self.active[:] = True
        self.velocities.fill(0.0)
        self.step_count = 0
        self.positions = self._initial_positions() if positions is None else self._validate_positions(positions)

    def neighbor_matrix(self) -> np.ndarray:
        """Boolean matrix of active neighbors within communication radius."""
        offsets = self.positions[:, None, :] - self.positions[None, :, :]
        distances = np.linalg.norm(offsets, axis=-1)
        neighbors = distances <= self.config.communication_radius
        np.fill_diagonal(neighbors, False)
        return neighbors & self.active[:, None] & self.active[None, :]

    def observed_positions(self) -> np.ndarray:
        """Positions with optional measurement noise."""
        if self.config.measurement_noise_std == 0:
            return self.positions.copy()
        noise = self.rng.normal(0.0, self.config.measurement_noise_std, self.positions.shape)
        return self.positions + noise

    def fail(self, robot_ids: Iterable[int]) -> None:
        """Mark robots as inactive: they stop moving and communicating."""
        indices = np.asarray(list(robot_ids), dtype=int)
        if indices.size == 0:
            return
        if np.any(indices < 0) or np.any(indices >= self.config.robot_count):
            raise IndexError("Identifiant de robot hors limites.")
        self.active[indices] = False
        self.velocities[indices] = 0.0

    def collision_pairs(self) -> List[Tuple[int, int]]:
        """List active pairs closer than safety_distance."""
        offsets = self.positions[:, None, :] - self.positions[None, :, :]
        distances = np.linalg.norm(offsets, axis=-1)
        mask = np.triu(distances < self.config.safety_distance, k=1)
        mask &= self.active[:, None] & self.active[None, :]
        return [tuple(pair) for pair in np.argwhere(mask)]

    def step(self, commands: np.ndarray) -> None:
        """Apply velocity command for one time step."""
        values = np.asarray(commands, dtype=float)
        if values.shape != (self.config.robot_count, 2):
            raise ValueError("commands doit être de forme (robot_count, 2).")

        # Clip speed to max_speed
        speeds = np.linalg.norm(values, axis=1, keepdims=True)
        clipped = values * np.minimum(1.0, self.config.max_speed / np.maximum(speeds, 1e-12))
        clipped[~self.active] = 0.0

        self.velocities = clipped
        self.positions = np.clip(
            self.positions + self.velocities * self.config.time_step,
            (0.0, 0.0),
            self.config.arena_size,
        )
        self.step_count += 1