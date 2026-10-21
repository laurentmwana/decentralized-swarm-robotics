"""Moteur physique : bornage de la commande puis mise à jour de position (docs/decisions/02-simulator.md).

Le moteur ne connaît aucune politique ; il applique des commandes déjà calculées.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import SimulatorConfig

from .arena import Arena


@dataclass(frozen=True)
class Simulator:
    arena: Arena
    config: SimulatorConfig

    def bound_commands(self, commands: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Remet à zéro la commande des robots retirés, puis borne la norme à v_max.

        Le bornage s'applique AVANT la mise à jour de position.
        """
        commands = np.asarray(commands, dtype=np.float64)
        if commands.ndim != 2 or commands.shape[1] != 2:
            raise ValueError(f"commandes attendues en (N, 2) : {commands.shape}")
        bounded = np.where(active[:, None], commands, 0.0)
        norms = np.linalg.norm(bounded, axis=1, keepdims=True)
        factor = np.minimum(1.0, self.config.v_max / np.maximum(norms, 1e-12))
        return bounded * factor

    def step(self, positions: np.ndarray, bounded_commands: np.ndarray, active: np.ndarray) -> np.ndarray:
        updated = positions.copy()
        updated[active] += bounded_commands[active] * self.config.dt
        return self.arena.clip(updated)
