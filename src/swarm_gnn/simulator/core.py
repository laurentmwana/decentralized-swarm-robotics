"""Dynamique, voisinage, pannes et collisions d'un essaim 2D."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class SimulationConfig:
    """Paramètres reproductibles d'un épisode de simulation."""

    seed: int
    robot_count: int
    arena_size: tuple[float, float]
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

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimulationConfig":
        """Charge une configuration de simulation depuis un fichier YAML."""
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        values["arena_size"] = tuple(values["arena_size"])
        return cls(**values)


class SwarmSimulator:
    """Simulateur à temps discret avec commandes de vitesse directe."""

    def __init__(self, config: SimulationConfig, positions: np.ndarray | None = None) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.active = np.ones(config.robot_count, dtype=bool)
        self.velocities = np.zeros((config.robot_count, 2), dtype=float)
        self.step_count = 0
        self.positions = self._initial_positions() if positions is None else self._validate_positions(positions)

    def _initial_positions(self) -> np.ndarray:
        return self.rng.uniform((0.0, 0.0), self.config.arena_size, size=(self.config.robot_count, 2))

    def _validate_positions(self, positions: np.ndarray) -> np.ndarray:
        values = np.asarray(positions, dtype=float).copy()
        if values.shape != (self.config.robot_count, 2):
            raise ValueError("positions doit être de forme (robot_count, 2).")
        return np.clip(values, (0.0, 0.0), self.config.arena_size)

    def reset(self, positions: np.ndarray | None = None) -> None:
        """Réinitialise l'état avec la même graine configurée."""
        self.rng = np.random.default_rng(self.config.seed)
        self.active[:] = True
        self.velocities.fill(0.0)
        self.step_count = 0
        self.positions = self._initial_positions() if positions is None else self._validate_positions(positions)

    def neighbor_matrix(self) -> np.ndarray:
        """Retourne la matrice booléenne des voisins actifs dans la portée."""
        offsets = self.positions[:, None, :] - self.positions[None, :, :]
        distances = np.linalg.norm(offsets, axis=-1)
        neighbors = distances <= self.config.communication_radius
        np.fill_diagonal(neighbors, False)
        return neighbors & self.active[:, None] & self.active[None, :]

    def observed_positions(self) -> np.ndarray:
        """Retourne les positions bruitées observables par les règles de contrôle."""
        if self.config.measurement_noise_std == 0:
            return self.positions.copy()
        noise = self.rng.normal(0.0, self.config.measurement_noise_std, self.positions.shape)
        return self.positions + noise

    def fail(self, robot_ids: Iterable[int]) -> None:
        """Retire les robots indiqués : ils ne bougent et ne communiquent plus."""
        indices = np.asarray(list(robot_ids), dtype=int)
        if indices.size == 0:
            return
        if np.any(indices < 0) or np.any(indices >= self.config.robot_count):
            raise IndexError("Identifiant de robot hors limites.")
        self.active[indices] = False
        self.velocities[indices] = 0.0

    def collision_pairs(self) -> list[tuple[int, int]]:
        """Liste les paires actives plus proches que la distance de sécurité."""
        offsets = self.positions[:, None, :] - self.positions[None, :, :]
        distances = np.linalg.norm(offsets, axis=-1)
        mask = np.triu(distances < self.config.safety_distance, k=1)
        mask &= self.active[:, None] & self.active[None, :]
        return [tuple(pair) for pair in np.argwhere(mask)]

    def step(self, commands: np.ndarray) -> None:
        """Applique une commande de vitesse par robot pendant un pas de temps."""
        values = np.asarray(commands, dtype=float)
        if values.shape != (self.config.robot_count, 2):
            raise ValueError("commands doit être de forme (robot_count, 2).")
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

    def apply_random_failures_at_midpoint(self, failure_fraction: float) -> None:
        """
        Applique des pannes aléatoires à mi-épisode selon la fraction spécifiée.

        Args:
            failure_fraction: Fraction de robots actifs à mettre en panne (0.0 à 1.0)
        """
        if not 0.0 <= failure_fraction <= 1.0:
            raise ValueError("failure_fraction doit être entre 0.0 et 1.0")

        # Compter les robots actifs actuellement
        active_count = np.sum(self.active)
        if active_count == 0:
            return

        # Calculer le nombre de robots à mettre en panne
        n_to_fail = max(1, int(active_count * failure_fraction))

        # Sélectionner aléatoirement parmi les robots actifs
        active_indices = np.where(self.active)[0]
        if len(active_indices) > 0:
            fail_indices = self.rng.choice(
                active_indices,
                size=min(n_to_fail, len(active_indices)),
                replace=False
            )
            self.fail(fail_indices)