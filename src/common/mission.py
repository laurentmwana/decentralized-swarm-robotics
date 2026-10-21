"""La consigne de mission partagée (docs/decisions/04-shapes.md)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Mission:
    shape_id: int
    target: np.ndarray = field(hash=False)

    def __post_init__(self) -> None:
        target = np.asarray(self.target, dtype=np.float64)
        if target.ndim != 2 or target.shape[1] != 2 or len(target) == 0:
            raise ValueError(f"cible invalide, attendu (N_points, 2) : {target.shape}")
        object.__setattr__(self, "target", target)
