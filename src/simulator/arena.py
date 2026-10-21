"""Arène carrée centrée sur l'origine (docs/decisions/02-simulator.md)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Arena:
    size: float

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"taille d'arène invalide : {self.size}")

    @property
    def half_size(self) -> float:
        return self.size / 2.0

    def clip(self, positions: np.ndarray) -> np.ndarray:
        half = self.half_size
        return np.clip(positions, -half, half)

    def contains(self, positions: np.ndarray) -> np.ndarray:
        return (np.abs(positions) <= self.half_size + 1e-9).all(axis=1)
