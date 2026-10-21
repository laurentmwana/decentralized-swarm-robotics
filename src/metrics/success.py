"""Critère de réussite unique (définition figée : docs/decisions/06-evaluation.md).

convergence_step : premier pas k tel que la distance de Chamfer normalisée est <= seuil à
tous les pas de k jusqu'à la fin (franchissement **stabilisé**, pas un simple passage
instantané). reussite : convergence_step existe. Un épisode dont la distance initiale
depasse deja le seuil ne peut pas etre valide ; la consigne « distance initiale > seuil »
est verifiee par le harnais d'evaluation, pas par la métrique elle-meme.
"""

from __future__ import annotations

import numpy as np

from .base import Metric
from .chamfer import chamfer_step


def convergence_step(curve: np.ndarray, threshold: float) -> int | None:
    """Premier k tel que courbe[t] <= seuil pour tout t >= k ; None si aucun franchissement stable."""
    ok = curve <= threshold
    if not ok[-1]:
        return None
    # balayer de la fin vers le debut : k est le dernier indice avant une rupture de stabilite
    for k in range(len(curve) - 1, -1, -1):
        if not ok[k]:
            return k + 1
    return 0


class SuccessCriterion(Metric):
    name = "success"

    def __init__(self, success_threshold: float, target_radius: float = 1.0) -> None:
        self.success_threshold = float(success_threshold)
        self.target_radius = float(target_radius)

    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> dict[str, object]:
        curve = np.array(
            [chamfer_step(positions[t][active[t]], target) / self.target_radius for t in range(len(positions))]
        )
        k = convergence_step(curve, self.success_threshold)
        return {"convergence_step": k, "success": k is not None}
