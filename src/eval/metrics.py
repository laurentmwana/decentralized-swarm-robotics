"""Métriques de qualité d'une formation d'essaim."""

from __future__ import annotations

import numpy as np


def symmetric_chamfer_distance(robots: np.ndarray, target: np.ndarray) -> float:
    """Calcule la distance de Chamfer symétrique moyenne en coordonnées 2D, normalisée par racine du nombre de points cible."""
    if len(robots) == 0 or len(target) == 0:
        return float("inf")
    distances = np.linalg.norm(robots[:, None] - target[None, :], axis=-1)
    raw_distance = float(distances.min(axis=1).mean() + distances.min(axis=0).mean())
    # Normalisation par racine du nombre de points cible pour comparabilité entre tailles d'essaim
    # Comme décidé en phase 0 pour que le seuil ait le même sens indépendamment du nombre de robots
    normalization_factor = np.sqrt(len(target))
    return raw_distance / normalization_factor if normalization_factor > 0 else float("inf")


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


def graph_connectivity(positions: np.ndarray, communication_radius: float) -> int:
    """Calcule le nombre de composantes connexes dans le graphe de communication.

    Deux robots sont connectés s'ils sont dans la portée de communication
    et tous les deux sont actifs (non supprimés).
    """
    if len(positions) == 0:
        return 0

    # Calculer la matrice d'adjacence basée sur la portée de communication
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    adjacency = distances <= communication_radius
    np.fill_diagonal(adjacency, False)  # Pas d'auto-boucle

    # Compter les composantes connexes en utilisant DFS/BFS simplifié
    visited = np.zeros(len(positions), dtype=bool)
    components = 0

    for i in range(len(positions)):
        if not visited[i]:
            # Nouvelle composante trouvée
            components += 1
            # Parcours en profondeur pour marquer tous les nœuds connectés
            stack = [i]
            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    # Ajouter tous les voisins non visités
                    neighbors = np.where(adjacency[node])[0]
                    stack.extend(neighbors[~visited[neighbors]])

    return components
