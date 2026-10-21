"""Graines : une seule source d'aléa par épisode (AGENT.md règle 5, docs/decisions/02-simulator.md).

Tous les tirages (positions initiales, pannes, échantillonnage, épisodes d'évaluation)
passent par `SeedManager`, qui dérive des sous-graines déterministes depuis une graine
mère unique. Même graine mère = mêmes suites, quel que soit l'ordre d'appel des clefs.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np


def _hash(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))


@dataclass(frozen=True)
class SeedManager:
    seed: int

    def rng(self, key: str) -> np.random.Generator:
        sequence = np.random.SeedSequence(entropy=self.seed, spawn_key=(_hash(key),))
        return np.random.default_rng(sequence)

    def episode_seeds(self, n: int) -> list[int]:
        rng = self.rng("episodes")
        return [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]

    def log(self) -> dict[str, object]:
        return {"root_seed": self.seed}
