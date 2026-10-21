"""Contrat commun de tous les contrôleurs (phase 3 le réutilise tel quel).

Un contrôleur ne voit que : l'état local (via l'adjacence à portée), la consigne de
mission partagée. Aucune affectation globale robot->point (AGENT.md règle 1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from src.common.mission import Mission

if TYPE_CHECKING:
    from src.simulator.state import SwarmState


class Controller(ABC):
    def reset(self) -> None:
        pass

    @abstractmethod
    def command(self, state: SwarmState, adjacency: np.ndarray, mission: Mission) -> np.ndarray:
        """Vitesse demandée par robot, forme (N, 2). Le simulateur borne à v_max ensuite."""
