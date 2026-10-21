"""Distance de Chamfer symétrique (définition figée : docs/decisions/06-evaluation.md).

valeur(t) = moyenne sur les robots actifs de la distance au point de cible le plus proche
          + moyenne sur les points de cible de la distance au robot actif le plus proche.
Moyenne des distances (non au carré), divisée par le rayon de la cible (= 1 par
construction), donc comparable entre 20/40/100 robots et entre formes.
"""

from __future__ import annotations

import numpy as np

from .base import Metric


def chamfer_step(active_positions: np.ndarray, target: np.ndarray) -> float:
    if len(active_positions) == 0:
        return float("nan")
    d = np.linalg.norm(active_positions[:, None, :] - target[None, :, :], axis=2)
    outward = d.min(axis=1).mean()
    result = d.min(axis=0).mean()
    return float(outward + result)


class ChamferDistance(Metric):
    name = "chamfer"

    def __init__(self, target_radius: float = 1.0) -> None:
        self.target_radius = float(target_radius)

    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> dict[str, object]:
        curve = np.array(
            [chamfer_step(positions[t][active[t]], target) / self.target_radius for t in range(len(positions))]
        )
        return {"distance_curve": curve, "final_distance": float(curve[-1])}
