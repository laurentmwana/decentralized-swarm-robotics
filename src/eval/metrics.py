"""Métriques de qualité d'une formation d'essaim."""

from __future__ import annotations

import numpy as np


def symmetric_chamfer_distance(robots: np.ndarray, target: np.ndarray) -> float:
    """Calcule la distance de Chamfer symétrique moyenne en coordonnées 2D."""
    if len(robots) == 0 or len(target) == 0:
        return float("inf")
    distances = np.linalg.norm(robots[:, None] - target[None, :], axis=-1)
    return float(distances.min(axis=1).mean() + distances.min(axis=0).mean())


def nearest_neighbor_std(positions: np.ndarray) -> float:
    """Mesure l'irrégularité des espacements par l'écart-type du plus proche voisin."""
    if len(positions) < 2:
        return float("nan")
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min(axis=1).std())


def collision_count(positions: np.ndarray, safety_distance: float) -> int:
    """Compte les paires de robots sous la distance de sécurité."""
    distances = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    return int(np.triu(distances < safety_distance, k=1).sum())
