"""Taux de collisions (définition figée : docs/decisions/06-evaluation.md).

Fraction des paires (i<j) **actives** ayant été à distance <= collision_radius au moins un
pas, normalisée par (nombre de paires actives x durée), mesurée sur les robots actifs.
"""

from __future__ import annotations

import numpy as np

from .base import Metric


class CollisionRate(Metric):
    name = "collisions"

    def __init__(self, collision_radius: float) -> None:
        self.collision_radius = float(collision_radius)

    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> dict[str, object]:
        step_count = len(positions) - 1
        events = 0
        total_pair_steps = 0.0
        for t in range(step_count):
            active_indices = np.flatnonzero(active[t])
            if len(active_indices) < 2:
                continue
            p = positions[t][active_indices]
            d = np.linalg.norm(p[:, None] - p[None, :], axis=2)
            iu = np.triu_indices(len(active_indices), k=1)
            total_pair_steps += len(iu[0])
            events += int((d[iu] <= self.collision_radius).sum())
        rate = events / total_pair_steps if total_pair_steps > 0 else 0.0
        return {"collision_rate": float(rate)}
