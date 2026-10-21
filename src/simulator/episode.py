"""Interface d'épisode unique, utilisable par tous les contrôleurs (docs/decisions/02-simulator.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from src.common.config import SimulatorConfig
from src.common.seeds import SeedManager
from src.common.mission import Mission

if TYPE_CHECKING:
    from src.controllers.base import Controller

from .arena import Arena
from .state import SwarmState
from .graph import CommunicationGraph
from .engine import Simulator
from .failures import FailureModel
from .perturbations import IdentityPerturbation, Perturbation


@dataclass(frozen=True)
class EpisodeResult:
    positions: np.ndarray = field(hash=False)  # (T+1, N, 2), vérité physique
    active: np.ndarray = field(hash=False)  # (T+1, N) bool
    commands: np.ndarray = field(hash=False)  # (T, N, 2) bornées, appliquées
    episode_seed: int = 0
    observed_positions: np.ndarray | None = field(default=None, hash=False)  # (T, N, 2)
    adjacency: np.ndarray | None = field(default=None, hash=False)  # (T+1, N, N)
    communication_radius: float | None = None
    failure_step: int | None = None
    failure_indices: tuple[int, ...] = ()

    @property
    def step(self) -> int:
        return len(self.positions) - 1


class Episode:
    """Orchestration d'un épisode : aléa par graine, pannes, contrôleur injecté, enregistrement."""

    def __init__(self, config: SimulatorConfig, perturbation: Perturbation | None = None) -> None:
        self.config = config
        self.arena = Arena(config.arena_size)
        self.graph = CommunicationGraph(config.r_c)
        self.engine = Simulator(self.arena, config)
        self.failures = FailureModel(config.failure_fraction)
        self.perturbation = perturbation if perturbation is not None else IdentityPerturbation()

    def initial_positions(self, n_robots: int, seeds: SeedManager) -> np.ndarray:
        rng = seeds.rng("initial_positions")
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_robots)
        radii = self.config.spawn_radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_robots))
        return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    def execute(self, mission: Mission, controller: Controller, n_robots: int, episode_seed: int) -> EpisodeResult:
        if n_robots <= 0:
            raise ValueError(f"taille d'essaim invalide : {n_robots}")
        seeds = SeedManager(seed=episode_seed)
        positions = self.initial_positions(n_robots, seeds)
        active = np.ones(n_robots, dtype=bool)
        failure = self.failures.plan(n_robots, self.config.horizon, seeds.rng("failures"))
        parts = self.perturbation.components()
        # Les clés ne contiennent ni niveau ni contrôleur : épisodes appariés.
        streams = {
            (part.stream_key, hook): seeds.rng(f"perturbations/{part.stream_key}/{hook}")
            for part in parts for hook in ("observe", "graph", "commands", "failures")
        }
        events = [failure] if failure is not None else []
        radius = self.config.r_c
        for part in parts:
            event = part.plan_failures(n_robots, self.config.horizon, streams[(part.stream_key, "failures")])
            if event is not None:
                if not 0 <= event.step < self.config.horizon or any(not 0 <= i < n_robots for i in event.indices):
                    raise ValueError("événement de panne invalide")
                events.append(event)
            radius = part.graph_radius(radius)
        if not np.isfinite(radius) or not 0 <= radius <= self.config.r_c:
            raise ValueError("portée perturbée invalide")

        def build_graph(state: SwarmState) -> np.ndarray:
            # Le bruit d'observation ne peut ni créer ni déplacer une arête physique.
            base = CommunicationGraph(radius).adjacency(state) if radius > 0 else np.zeros((n_robots, n_robots), dtype=bool)
            adjacency = base.copy()
            for part in parts:
                adjacency = np.asarray(part.perturb_graph(adjacency, state, streams[(part.stream_key, "graph")]), dtype=bool)
                if adjacency.shape != base.shape:
                    raise ValueError("forme de graphe perturbé invalide")
                adjacency = adjacency & adjacency.T & base
            np.fill_diagonal(adjacency, False)
            return adjacency

        position_history: list[np.ndarray] = []
        active_history: list[np.ndarray] = []
        command_history: list[np.ndarray] = []
        observation_history: list[np.ndarray] = []
        graph_history: list[np.ndarray] = []

        for t in range(self.config.horizon):
            for event in events:
                if t == event.step:
                    active[list(event.indices)] = False
            position_history.append(positions.copy())
            active_history.append(active.copy())
            # Une copie sépare toujours l'observation de la vérité du moteur.
            state = SwarmState(positions=positions.copy(), active=active.copy(), step=t)
            state.positions.setflags(write=False)
            state.active.setflags(write=False)
            adjacency = build_graph(state)
            graph_history.append(adjacency.copy())
            observed = state
            for part in parts:
                observed = part.observe(observed, streams[(part.stream_key, "observe")])
                if observed.positions.shape != positions.shape or not np.isfinite(observed.positions).all():
                    raise ValueError("observation perturbée invalide")
                if observed.step != t or not np.array_equal(observed.active, active):
                    raise ValueError("une observation ne peut modifier l'activité ni le pas")
            observation_history.append(observed.positions.copy())
            # Le contrôleur recalcule notamment sa direction de cible depuis observed.
            commands = controller.command(observed, adjacency.copy(), mission)
            for part in parts:
                commands = part.perturb_commands(commands, state, streams[(part.stream_key, "commands")])
            commands = np.asarray(commands, dtype=np.float64)
            if commands.shape != positions.shape or not np.isfinite(commands).all():
                raise ValueError("commandes perturbées invalides")
            bounded = self.engine.bound_commands(commands, active)
            positions = self.engine.step(positions, bounded, active)
            command_history.append(bounded)

        position_history.append(positions.copy())
        active_history.append(active.copy())
        # Graphe final diagnostique : aucun appel supplémentaire au contrôleur.
        graph_history.append(build_graph(SwarmState(positions.copy(), active.copy(), self.config.horizon)))

        return EpisodeResult(
            positions=np.array(position_history),
            active=np.array(active_history),
            commands=np.array(command_history),
            episode_seed=episode_seed,
            observed_positions=np.array(observation_history),
            adjacency=np.array(graph_history),
            communication_radius=radius,
            failure_step=min((event.step for event in events), default=None),
            failure_indices=tuple(sorted({i for event in events for i in event.indices})),
        )


def simulate_episode(
    target: np.ndarray,
    controller: Controller,
    config: SimulatorConfig,
    n_robots: int,
    episode_seed: int,
    shape_id: int = 0,
    perturbation: Perturbation | None = None,
) -> EpisodeResult:
    """Façade conforme au contrat des décisions 02 : positions initiales tirées par la graine."""
    episode = Episode(config, perturbation)
    mission = Mission(shape_id=shape_id, target=np.asarray(target, dtype=np.float64))
    return episode.execute(mission, controller, n_robots, episode_seed)
