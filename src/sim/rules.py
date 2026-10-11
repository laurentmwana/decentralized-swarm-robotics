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
