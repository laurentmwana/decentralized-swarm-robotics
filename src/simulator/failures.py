"""Modèle de panne : retrait aléatoire à mi-épisode (règle figée dans docs/decisions/02-simulator.md).

Un robot retiré est supprimé du graphe, exclu des métriques, immobile et sans commande ;
il reste présent dans l'état (masque `active`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FailureEvent:
    step: int
    indices: tuple[int, ...]


@dataclass(frozen=True)
class FailureModel:
    fraction: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError(f"fraction de panne hors [0, 1) : {self.fraction}")

    def plan(self, n_robots: int, horizon: int, rng: np.random.Generator) -> Optional[FailureEvent]:
        if self.fraction == 0.0:
            return None
        failure_count = int(round(self.fraction * n_robots))
        if failure_count == 0:
            return None
        indices = rng.choice(n_robots, size=failure_count, replace=False)
        return FailureEvent(step=horizon // 2, indices=tuple(sorted(int(i) for i in indices)))
