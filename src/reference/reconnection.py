"""Rendez-vous partagé sans connaissance des composantes ou positions distantes."""
from dataclasses import dataclass
import math

import numpy as np

from .allocation import _integer


@dataclass(frozen=True)
class ReconnectionConfig:
    gather_seconds: float = 8.
    braking_seconds: float = 1.
    spacing: float = .6
    center: tuple[float, float] = (0., 0.)

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.gather_seconds, self.braking_seconds, self.spacing)):
            raise ValueError('Durées et espacement de rendez-vous finis et positifs requis')
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError('Centre de rendez-vous 2D fini requis')
        object.__setattr__(self, 'center', tuple(float(v) for v in center))

    def steps(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Pas de temps positif requis')
        return math.ceil(self.gather_seconds/dt-1e-9), math.ceil(self.braking_seconds/dt-1e-9)


def rendezvous_targets(count, config, reconnection, safety_distance):
    _integer(count, 'count', 2)
    if not math.isfinite(safety_distance) or safety_distance <= 0:
        raise ValueError('Distance de sécurité finie et positive requise')
    if reconnection.spacing <= safety_distance:
        raise ValueError('Espacement de rendez-vous supérieur à la sécurité requis')
    if reconnection.spacing > config.communication_radius:
        raise ValueError('La portée doit couvrir les voisins du rendez-vous')
    radius = reconnection.spacing/(2*np.sin(np.pi/count))
    angles = np.arange(count)*2*np.pi/count
    points = np.asarray(reconnection.center)+radius*np.column_stack((np.cos(angles), np.sin(angles)))
    half = config.physics.arena_size/2-config.physics.robot_radius
    if (np.abs(points) > half).any():
        raise ValueError('Le rendez-vous doit tenir dans l’arène')
    return points
