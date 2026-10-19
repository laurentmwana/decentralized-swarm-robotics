"""Références comportementales locales pour le simulateur."""

import numpy as np

from .core import SwarmSimulator


def local_gathering_and_repulsion(
    simulator: SwarmSimulator,
    gathering_gain: float = 1.0,
    repulsion_gain: float = 1.0,
) -> np.ndarray:
    """Calcule une commande locale de cohésion et de répulsion.

    Chaque robot ne dépend que des positions de ses voisins dans le rayon de
    communication. Un robot isolé reçoit une commande nulle.
    """
    if gathering_gain < 0 or repulsion_gain < 0:
        raise ValueError("Les gains doivent être positifs ou nuls.")
    positions = simulator.observed_positions()
    neighbors = simulator.neighbor_matrix()
    commands = np.zeros_like(positions)
    for robot_id in np.flatnonzero(simulator.active):
        neighbor_ids = np.flatnonzero(neighbors[robot_id])
        if neighbor_ids.size == 0:
            continue
        relative = positions[neighbor_ids] - positions[robot_id]
        gathering = relative.mean(axis=0)
        distances = np.linalg.norm(relative, axis=1)
        close = distances < simulator.config.safety_distance
        repulsion = np.zeros(2)
        if np.any(close):
            safe_relative = relative[close]
            safe_distances = np.maximum(distances[close, None], 1e-12)
            repulsion = (-safe_relative / safe_distances**2).sum(axis=0)
        commands[robot_id] = gathering_gain * gathering + repulsion_gain * repulsion
    return commands


def local_manual_with_shape_consensus(
    simulator: SwarmSimulator,
    target_points: np.ndarray,
    attraction_gain: float = 1.0,
    repulsion_gain: float = 1.0,
) -> np.ndarray:
    """Calcule une commande manuelle avec consigne de forme : attraction vers le contour + répulsion.

    Chaque robot est attiré vers le point de la cible le plus proche et repoussé
    par les voisins trop proches pour éviter les collisions.
    Un robot isolé reçoit uniquement la commande d'attraction vers la cible.
    """
    if attraction_gain < 0 or repulsion_gain < 0:
        raise ValueError("Les gains doivent être positifs ou nuls.")

    positions = simulator.observed_positions()
    # S'assurer que target_points a la bonne forme pour le calcul des distances
    if target_points.ndim == 2:
        # target_points est de forme [n_points, 2], on veut calculer la distance
        # de chaque robot à chaque point cible
        pass  # Déjà dans la bonne forme
    elif target_points.ndim == 3 and target_points.shape[0] == len(positions):
        # target_points est déjà de forme [n_robots, n_points, 2]
        pass
    else:
        # Essayer de broadcast si nécessaire
        if target_points.ndim == 2 and len(positions) > 0:
            target_points = np.broadcast_to(target_points, (len(positions), *target_points.shape))

    commands = np.zeros_like(positions)

    for robot_id in np.flatnonzero(simulator.active):
        robot_position = positions[robot_id]

        # Trouver le point cible le plus proche
        distances_to_target = np.linalg.norm(target_points[robot_id] - robot_position, axis=-1)
        closest_target_idx = np.argmin(distances_to_target)
        closest_target_point = target_points[robot_id][closest_target_idx]

        # Attraction vers le point cible le plus proche
        attraction = closest_target_point - robot_position

        # Répulsion des voisins trop proches (éviter les collisions)
        neighbor_ids = np.flatnonzero(simulator.neighbor_matrix()[robot_id])
        repulsion = np.zeros(2)
        if len(neighbor_ids) > 0:
            neighbor_positions = positions[neighbor_ids]
            relative_positions = neighbor_positions - robot_position
            distances = np.linalg.norm(relative_positions, axis=1)
            # Seuils de répulsion : plus proches que la distance de sécurité
            too_close = distances < simulator.config.safety_distance
            if np.any(too_close):
                too_close_relative = relative_positions[too_close]
                too_close_distances = np.maximum(distances[too_close, None], 1e-12)
                # Répulsion inverse du carré de la distance (comme dans la règle de regroupement)
                repulsion = (-too_close_relative / too_close_distances**2).sum(axis=0)

        commands[robot_id] = attraction_gain * attraction + repulsion_gain * repulsion

    return commands
