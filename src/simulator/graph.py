"""Graphe de communication par portée, recalculé à chaque pas, sans auto-boucle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import SwarmState


@dataclass(frozen=True)
class CommunicationGraph:
    r_c: float

    def __post_init__(self) -> None:
        if self.r_c <= 0:
            raise ValueError(f"portée de communication invalide : {self.r_c}")

    def adjacency(self, state: SwarmState) -> np.ndarray:
        """Matrice booléenne (N, N) : arête si distance <= r_c, les deux robots actifs.

        Pas d'auto-boucle ; un robot retiré n'est ni émetteur ni récepteur.
        """
        positions = state.positions
        d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        adj = (d <= self.r_c) & (d > 0.0)
        active = state.active
        adj &= active[:, None] & active[None, :]
        np.fill_diagonal(adj, False)
        return adj

    @staticmethod
    def connected_components(adjacency: np.ndarray, active: np.ndarray) -> int:
        """Nombre de composantes connexes du sous-graphe des robots actifs.

        Un robot actif isolé compte comme une composante. Zéro robot actif -> zéro.
        """
        indices = np.flatnonzero(active)
        remaining = set(indices.tolist())
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
