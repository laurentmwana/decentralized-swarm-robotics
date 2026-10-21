"""Contrat commun des métriques (phase 2).

Toute métrique est une fonction **pure** des enregistrements de l'épisode : positions
(T+1, N, 2), masque d'activité (T+1, N), cible (P, 2). Garantie : recalculer depuis les
fichiers enregistrés redonne exactement les mêmes nombres (test obligatoire).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    name: str = "metric"

    @abstractmethod
    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> dict[str, object]:
        """Retourne un dictionnaire de valeurs scalaires ou tableaux (une entrée par clé)."""


def active_robots(positions: np.ndarray, active: np.ndarray) -> np.ndarray:
    return positions[active]
