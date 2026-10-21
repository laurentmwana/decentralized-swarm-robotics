"""Régularité de répartition (valeur absolue, docs/decisions/06-evaluation.md).

Ecart-type des distances au plus proche voisin parmi les robots actifs, rapporté à
l'espacement ideal sur la cible (perimetre effectif / nombre de robots actifs).
"""

from __future__ import annotations

import numpy as np

from .base import Metric


def nearest_neighbor(p: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(p[:, None] - p[None, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


class SpreadRegularity(Metric):
    name = "regularity"

    def __init__(self, reference: str = "auto") -> None:
        self.reference = reference

    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> dict[str, object]:
        t_final = len(positions) - 1
        active_indices = np.flatnonzero(active[t_final])
        if len(active_indices) < 2:
            return {"regularity_value": 0.0, "normalized_regularity": float("nan")}
        nearest_distances = nearest_neighbor(positions[t_final][active_indices])
        deviation = float(np.std(nearest_distances))
        ideal = self._ideal_spacing(len(active_indices))
        return {"regularity_value": deviation, "normalized_regularity": float(deviation / ideal) if ideal > 0 else float("nan")}

    def _ideal_spacing(self, n: int) -> float:
        if self.reference != "auto":
            return float(self.reference)
        return 2.0 * np.pi / n  # perimetre du cercle unite / n
