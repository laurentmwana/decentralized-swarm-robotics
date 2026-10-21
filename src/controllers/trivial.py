"""Contrôleur trivial de démonstration (critère d'acceptation phase 1 uniquement).

Attraction vers le point de cible le plus proche (calcul local, pas d'affectation
globale) + répulsion à courte portée entre voisins. Les vraies références (regroupement,
règle manuelle réglée) arrivent en phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.mission import Mission
from src.controllers.base import Controller
from src.simulator.state import SwarmState


@dataclass(frozen=True)
class TrivialController(Controller):
    attraction_gain: float
    repulsion_gain: float
    repulsion_radius: float

    def command(self, state: SwarmState, adjacency: np.ndarray, mission: Mission) -> np.ndarray:
        positions = state.positions
        delta = mission.target[None, :, :] - positions[:, None, :]
        distances = np.linalg.norm(delta, axis=2)
        nearest = np.argmin(distances, axis=1)
        toward_target = delta[np.arange(len(positions)), nearest]
        cmd = self.attraction_gain * toward_target

        if self.repulsion_radius > 0 and len(positions) > 1:
            d = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
            nearby = adjacency & (d < self.repulsion_radius) & (d > 0.0)
            directions = positions[:, None] - positions[None, :]
            weighting = np.where(nearby, (self.repulsion_radius - d) / self.repulsion_radius, 0.0)
            cmd += self.repulsion_gain * (directions * weighting[:, :, None]).sum(axis=1)
        return cmd
