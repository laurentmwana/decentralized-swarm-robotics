"""Connectivite du graphe de communication sur toute la trajectoire.

Le graphe n'est pas enregistre par le simulateur : la metrique le **recalcule** a chaque
pas a partir des positions et du masque d'activite, avec la portee r_c de la
configuration. Pure fonction des enregistrements, donc reproducible a l'identique.
"""

from __future__ import annotations

import numpy as np

from .base import Metric


def _adjacency(p: np.ndarray, active: np.ndarray, r_c: float) -> np.ndarray:
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    adj = (d <= r_c) & (d > 0.0)
    adj &= active[:, None] & active[None, :]
    np.fill_diagonal(adj, False)
    return adj


def _components(adjacency: np.ndarray, active: np.ndarray) -> int:
    remaining = set(np.flatnonzero(active).tolist())
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            n = stack.pop()
            for m in np.flatnonzero(adjacency[n]):
                if int(m) in remaining:
                    remaining.remove(int(m))
                    stack.append(int(m))
    return components


class GraphConnectivity(Metric):
    name = "connectivity"

    def __init__(self, r_c: float) -> None:
        self.r_c = float(r_c)

    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> dict[str, object]:
        step_count = len(positions)
        components = np.array(
            [_components(_adjacency(positions[t], active[t], self.r_c), active[t]) for t in range(step_count)],
            dtype=int,
        )
        connected = components <= 1
        return {
            "components_by_step": components,
            "final_components": int(components[-1]),
            "initially_connected": bool(connected[0]),
            "connected_steps": int(connected.sum()),
        }
