"""Objet de valeur immuable : l'état complet de l'essaim à un pas donné."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SwarmState:
    positions: np.ndarray = field(hash=False)  # (N, 2)
    active: np.ndarray = field(hash=False)  # (N,) bool — False = robot retiré (panne)
    step: int

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions, dtype=np.float64)
        active = np.asarray(self.active, dtype=bool)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions attendues en (N, 2) : {positions.shape}")
        if active.shape != (len(positions),):
            raise ValueError(f"masque d'activité attendu en (N,) : {active.shape}")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "active", active)

    @property
    def n_robots(self) -> int:
        return len(self.positions)

    @property
    def active_indices(self) -> np.ndarray:
        return np.flatnonzero(self.active)
