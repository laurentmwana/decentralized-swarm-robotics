"""Échantillonnage des formes : point le plus éloigné, normalisation dans le cadre commun.

Contrainte absolue (anomalie plan §0.6) : le nombre de points distincts rendus doit être
égal à la résolution demandée ; sinon `ShapeSampler` lève une erreur. Aucun remplissage
par duplication, jamais.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.seeds import SeedManager


@dataclass(frozen=True)
class ShapeSampler:
    seed_manager: SeedManager

    def sample(self, mask: np.ndarray, n_points: int) -> np.ndarray:
        """Sélectionne `n_points` pixels distincts du masque ( coordonnées image ), par farthest-point."""
        coords = np.argwhere(mask).astype(np.float64)  # (P, 2) en (ligne, colonne)
        if len(coords) == 0:
            raise ValueError("masque vide : aucune encre à échantillonner")
        if len(coords) < n_points:
            raise ValueError(
                f"le rendu ne contient que {len(coords)} pixels distincts < {n_points} points demandés ; "
                "augmenter la résolution de rendu ou épaissir le trait"
            )
        rng = self.seed_manager.rng(f"fps:{hash_pixels(coords)}:{n_points}")
        chosen = [int(rng.integers(len(coords)))]
        dist_min = np.linalg.norm(coords - coords[chosen[0]], axis=1)
        for _ in range(n_points - 1):
            nxt = int(np.argmax(dist_min))
            chosen.append(nxt)
            dist_min = np.minimum(dist_min, np.linalg.norm(coords - coords[nxt], axis=1))
        points = coords[chosen]
        distinct = np.unique(points, axis=0)
        if len(distinct) != n_points:
            raise AssertionError(
                f"anti-doublon : {len(distinct)} points distincts pour {n_points} demandés"
            )
        # (ligne, colonne) -> (x, y) avec y vers le haut
        return np.stack([points[:, 1], mask.shape[0] - 1.0 - points[:, 0]], axis=1)

    @staticmethod
    def normalize(points: np.ndarray, target_radius: float = 1.0) -> np.ndarray:
        """Recentre sur le centroïde puis met à l'échelle : rayon de cible = target_radius."""
        center = points.mean(axis=0)
        centered = points - center
        radius = np.linalg.norm(centered, axis=1).max()
        if radius <= 0:
            raise ValueError("forme dégénérée : tous les points confondus")
        return centered * (target_radius / radius)


def hash_pixels(coords: np.ndarray) -> int:
    """Identifiant stable de l'ensemble de pixels (pour une graine FPS reproductible)."""
    quant = np.round(coords).astype(np.int64)
    return int(np.abs(quant).sum() * 1_000_003 + len(quant))
