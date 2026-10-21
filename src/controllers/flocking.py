"""Règle de regroupement (cohésion–répulsion), sans consigne de forme (baseline phase 3).

Plancher honnête : les robots se collent aux voisins proches et s'évitent à courte
portée, sans jamais regarder la cible. Décentralisé : seules les positions des
voisins du graphe de communication interviennent ; aucun paramètre codé (gains en
configuration).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.mission import Mission
from src.controllers.base import Controller
from src.simulator.state import SwarmState


@dataclass(frozen=True)
class FlockingController(Controller):
    cohesion_gain: float
    repulsion_gain: float
    repulsion_radius: float

    def __post_init__(self) -> None:
        for name in ("cohesion_gain", "repulsion_gain", "repulsion_radius"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"parametre de regroupement negatif : {name} = {value}")

    def command(self, state: SwarmState, adjacency: np.ndarray, mission: Mission) -> np.ndarray:
        positions = state.positions
        degrees = adjacency.sum(axis=1)
        neighbors = adjacency @ positions
        # cohesion nulle pour un robot isole (sinon il serait attire vers l'origine : info globale)
        cohesion = np.divide(neighbors, degrees[:, None], out=np.zeros_like(positions), where=degrees[:, None] > 0)
        cmd = self.cohesion_gain * np.where(degrees[:, None] > 0, cohesion - positions, 0.0)

        if self.repulsion_radius > 0 and len(positions) > 1:
            d = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
            nearby = adjacency & (d < self.repulsion_radius) & (d > 0.0)
            directions = positions[:, None] - positions[None, :]
            weighting = np.where(nearby, (self.repulsion_radius - d) / self.repulsion_radius, 0.0)
            cmd = cmd + self.repulsion_gain * (directions * weighting[:, :, None]).sum(axis=1)
        return cmd
