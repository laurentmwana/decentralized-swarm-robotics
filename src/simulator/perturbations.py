"""Perturbations sans politique : observation, portée, arêtes, commandes et retraits.

Le bruit de communication est une suppression d'arêtes, PAS un bruit de messages.
Les positions observées ne définissent jamais la portée physique du graphe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .failures import FailureEvent
from .state import SwarmState


class Perturbation(ABC):
    """Crochets composables ; chaque feuille possède ses propres flux nommés."""

    @property
    @abstractmethod
    def stream_key(self) -> str:
        """Nom anglais stable, indépendant du niveau de perturbation."""

    def components(self) -> tuple[Perturbation, ...]:
        return (self,)

    def observe(self, state: SwarmState, rng: np.random.Generator) -> SwarmState:
        return state

    def graph_radius(self, radius: float) -> float:
        return radius

    def perturb_graph(self, adjacency: np.ndarray, state: SwarmState, rng: np.random.Generator) -> np.ndarray:
        return adjacency

    def perturb_commands(self, commands: np.ndarray, state: SwarmState, rng: np.random.Generator) -> np.ndarray:
        return commands

    def plan_failures(self, n_robots: int, horizon: int, rng: np.random.Generator) -> FailureEvent | None:
        return None


@dataclass(frozen=True)
class IdentityPerturbation(Perturbation):
    stream_key = "identity"

    def components(self) -> tuple[Perturbation, ...]:
        # Aucun générateur n'est même créé dans le chemin ordinaire.
        return ()


@dataclass(frozen=True)
class PositionNoise(Perturbation):
    """Bruit gaussien indépendant par coordonnée active, dans le repère de la cible.

    Chaque robot utilise sa position observée pour sa consigne locale ; ses voisins
    ne reçoivent cette observation que via les arêtes autorisées du contrôleur.
    """

    std: float
    stream_key = "position_noise"

    def __post_init__(self) -> None:
        if not np.isfinite(self.std) or self.std < 0:
            raise ValueError("écart-type de position invalide")

    def observe(self, state: SwarmState, rng: np.random.Generator) -> SwarmState:
        if self.std == 0:
            return state
        positions = state.positions.copy()
        positions[state.active] += rng.normal(0.0, self.std, size=(int(state.active.sum()), 2))
        return SwarmState(positions, state.active.copy(), state.step)


@dataclass(frozen=True)
class CommunicationNoise(Perturbation):
    """Suppression Bernoulli indépendante par arête non orientée et par pas."""

    probability: float
    stream_key = "communication_noise"

    def __post_init__(self) -> None:
        if not np.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("probabilité de suppression hors [0, 1]")

    def perturb_graph(self, adjacency: np.ndarray, state: SwarmState, rng: np.random.Generator) -> np.ndarray:
        if self.probability == 0:
            return adjacency
        result = adjacency.copy()
        rows, columns = np.nonzero(np.triu(result, k=1))
        dropped = rng.random(len(rows)) < self.probability
        result[rows[dropped], columns[dropped]] = False
        result[columns[dropped], rows[dropped]] = False
        return result


@dataclass(frozen=True)
class RangeReduction(Perturbation):
    """Portée vraie multipliée par scale ; scale=1 est l'identité, 0 coupe tout."""

    scale: float
    stream_key = "range_reduction"

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or not 0 <= self.scale <= 1:
            raise ValueError("facteur de portée hors [0, 1]")

    def graph_radius(self, radius: float) -> float:
        return radius * self.scale


@dataclass(frozen=True)
class FaultyRobot(Perturbation):
    """Retrait immobile de round(fraction*N) robots à mi-épisode.

    Tirage indépendant des pannes ordinaires : les ensembles sont réunis, sans
    doubler un retrait. La fraction effectivement retirée est enregistrée. Aucune
    fausse coordonnée globale ni transmission aberrante n'est revendiquée ici.
    """

    fraction: float
    stream_key = "faulty_robot"

    def __post_init__(self) -> None:
        if not np.isfinite(self.fraction) or not 0 <= self.fraction <= 1:
            raise ValueError("fraction de robots défaillants hors [0, 1]")

    def plan_failures(self, n_robots: int, horizon: int, rng: np.random.Generator) -> FailureEvent | None:
        count = int(round(self.fraction * n_robots))
        if count == 0:
            return None
        indices = tuple(sorted(int(i) for i in rng.choice(n_robots, count, replace=False)))
        return FailureEvent(horizon // 2, indices)


@dataclass(frozen=True)
class CompositePerturbation(Perturbation):
    perturbations: tuple[Perturbation, ...]
    stream_key = "composite"

    def __post_init__(self) -> None:
        keys = [part.stream_key for part in self.components()]
        if len(keys) != len(set(keys)):
            raise ValueError("flux de perturbations dupliqués dans la composition")

    def components(self) -> tuple[Perturbation, ...]:
        return tuple(part for perturbation in self.perturbations for part in perturbation.components())
