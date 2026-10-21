"""Scénarios et campagne : produit cartésien tailles × retraits × formes × répétitions.

Les positions initiales ne sont PAS un paramètre du scénario : elles sont tirées par la
graine d'épisode via `Episode.initial_positions`, donc identiques pour tous les
contrôleurs (comparaison appariée valide par construction).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    n_robots: int
    failure_fraction: float
    shape_key: str
    episode_seed: int

    @property
    def identifier(self) -> str:
        shape = self.shape_key.replace("@", "_at_")
        return f"n{self.n_robots}_failure{int(round(self.failure_fraction * 100)):02d}_{shape}_seed{self.episode_seed}"


@dataclass(frozen=True)
class ScenarioGrid:
    """Grille complète d'une campagne (positions figees par graine d'episode)."""

    swarm_sizes: tuple[int, ...]
    failure_fractions: tuple[float, ...]
    shapes: tuple[str, ...]
    repetitions: int
    evaluation_seed: int
    horizon: int
    seed_stride: int

    def __post_init__(self) -> None:
        if self.repetitions < 1 or self.horizon < 1 or self.seed_stride < self.repetitions:
            raise ValueError("répétitions, horizon ou espacement des graines invalides")
        if not self.swarm_sizes or not self.shapes or not self.failure_fractions:
            raise ValueError("la grille de scénarios ne peut pas être vide")
        if any(size < 1 for size in self.swarm_sizes) or any(not 0 <= fraction < 1 for fraction in self.failure_fractions):
            raise ValueError("taille d'essaim ou fraction de retraits invalide")

    @staticmethod
    def from_config(cfg: dict) -> "ScenarioGrid":
        return ScenarioGrid(
            swarm_sizes=tuple(int(t) for t in cfg["swarm_sizes"]),
            failure_fractions=tuple(float(f) for f in cfg["failure_fractions"]),
            shapes=tuple(str(f) for f in cfg["shapes"]),
            repetitions=int(cfg["repetitions"]),
            evaluation_seed=int(cfg["evaluation_seed"]),
            horizon=int(cfg["evaluation_horizon"]),
            seed_stride=int(cfg["seed_stride"]),
        )

    def scenarios(self) -> list[Scenario]:
        return [
            Scenario(n_robots=n, failure_fraction=f, shape_key=shape, episode_seed=self.episode_seed(rep))
            for n in self.swarm_sizes
            for f in self.failure_fractions
            for shape in self.shapes
            for rep in range(self.repetitions)
        ]

    def episode_seed(self, repetition: int) -> int:
        return self.evaluation_seed * self.seed_stride + repetition

    def simulator_config(self, cfg_sim):
        return dataclasses.replace(cfg_sim, horizon=self.horizon)
