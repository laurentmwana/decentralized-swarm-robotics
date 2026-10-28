"""Épisodes du nouveau plan : commandes locales d'accélération et traces SI."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from src.common.seeds import SeedManager
from .dynamics import AccelerationDynamics, DynamicsConfig, _limit, _vectors


def _snapshot(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LocalObservation:
    position: np.ndarray
    velocity: np.ndarray
    relative_positions: np.ndarray
    relative_velocities: np.ndarray
    time: float


@dataclass(frozen=True)
class AccelerationEpisodeConfig:
    physics: DynamicsConfig
    horizon: int
    communication_radius: float
    spawn_radius: float

    def __post_init__(self):
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("L'horizon doit être un entier positif")
        if not np.isfinite(self.communication_radius) or self.communication_radius <= 0:
            raise ValueError("La portée doit être finie et positive")
        half = self.physics.arena_size / 2 - self.physics.robot_radius
        if not np.isfinite(self.spawn_radius) or not 0 < self.spawn_radius <= half:
            raise ValueError("Le disque initial doit tenir dans l'arène")

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or not isinstance(data.get('physics'), dict):
            raise ValueError("Configuration attendue : objet avec section physics")
        return cls(**{**data, 'physics': DynamicsConfig(**data['physics'])})


@dataclass(frozen=True)
class AccelerationEpisodeResult:
    positions: np.ndarray
    velocities: np.ndarray
    requested_accelerations: np.ndarray
    accelerations: np.ndarray
    adjacency: np.ndarray
    config: AccelerationEpisodeConfig
    seed: int

    @property
    def times(self):
        return np.arange(len(self.positions)) * self.config.physics.dt

    def save(self, path):
        """Archive autonome, sans pickle, contenant configuration et unités."""
        metadata = dict(config=asdict(self.config), seed=self.seed,
                        units=dict(position='m', velocity='m/s', acceleration='m/s^2', time='s'))
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, positions=self.positions, velocities=self.velocities,
                                requested_accelerations=self.requested_accelerations,
                                accelerations=self.accelerations, adjacency=self.adjacency,
                                times=self.times, metadata=json.dumps(metadata))


class AccelerationEpisode:
    def __init__(self, config: AccelerationEpisodeConfig):
        self.config = config
        self.engine = AccelerationDynamics(config.physics)

    def adjacency(self, positions):
        distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        graph = distances <= self.config.communication_radius
        np.fill_diagonal(graph, False)
        return graph

    def execute(self, controller: Callable[[LocalObservation], np.ndarray],
                n_robots: int, seed: int, initial_positions=None, initial_velocities=None):
        if isinstance(n_robots, bool) or not isinstance(n_robots, int) or n_robots < 1:
            raise ValueError("Le nombre de robots doit être un entier positif")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("La graine doit être un entier positif ou nul")
        rng = SeedManager(seed).rng('initial_positions')
        if initial_positions is None:
            angles = rng.uniform(0, 2 * np.pi, n_robots)
            radii = self.config.spawn_radius * np.sqrt(rng.uniform(0, 1, n_robots))
            initial_positions = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        positions = _vectors(initial_positions, 'positions').copy()
        velocities = np.zeros((n_robots, 2)) if initial_velocities is None else _vectors(initial_velocities, 'velocities').copy()
        if positions.shape != (n_robots, 2) or velocities.shape != positions.shape:
            raise ValueError("État initial incompatible avec le nombre de robots")
        # Validation physique avant tout appel au contrôleur.
        self.engine.step(positions, velocities, np.zeros_like(positions))
        history_p, history_v = [positions.copy()], [velocities.copy()]
        requests, bounded, graphs = [], [], []
        for step in range(self.config.horizon):
            graph = self.adjacency(positions)
            commands = []
            for i in range(n_robots):
                neighbors = graph[i]
                observation = LocalObservation(
                    _snapshot(positions[i]), _snapshot(velocities[i]),
                    _snapshot(positions[neighbors] - positions[i]),
                    _snapshot(velocities[neighbors] - velocities[i]),
                    step * self.config.physics.dt)
                command = np.asarray(controller(observation), dtype=float)
                if command.shape != (2,) or not np.isfinite(command).all():
                    raise ValueError("Le contrôleur doit retourner deux accélérations finies")
                commands.append(command.copy())
            requested = np.stack(commands)
            acceleration = _limit(requested, self.config.physics.max_acceleration)
            positions, velocities = self.engine.step(positions, velocities, requested)
            requests.append(requested)
            bounded.append(acceleration)
            graphs.append(graph)
            history_p.append(positions.copy())
            history_v.append(velocities.copy())
        graphs.append(self.adjacency(positions))
        return AccelerationEpisodeResult(np.array(history_p), np.array(history_v),
                                         np.array(requests), np.array(bounded),
                                         np.array(graphs), self.config, seed)
