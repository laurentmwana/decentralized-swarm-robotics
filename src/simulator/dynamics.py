"""Dynamique à accélérations du nouveau PLAN.md, en unités SI."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicsConfig:
    arena_size: float
    dt: float
    max_speed: float
    max_acceleration: float
    robot_radius: float

    def __post_init__(self):
        values = (self.arena_size, self.dt, self.max_speed,
                  self.max_acceleration, self.robot_radius)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("Les paramètres physiques doivent être finis et positifs")
        if 2 * self.robot_radius >= self.arena_size:
            raise ValueError("Le diamètre du robot doit être inférieur au côté de l'arène")


def _vectors(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} doit être un tableau fini (N, 2)")
    return array


def _limit(vectors, maximum):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1.0, maximum / np.maximum(norms, 1e-12))


@dataclass(frozen=True)
class AccelerationDynamics:
    config: DynamicsConfig

    def step(self, positions, velocities, accelerations):
        """Retourne de nouveaux tableaux (positions, vitesses), sans modifier les entrées.

        Euler semi-implicite : a bornée, v' = borne(v + a dt), x' = x + v' dt.
        Le contact avec un mur annule seulement la composante normale de vitesse.
        Les contacts entre robots sont laissés au détecteur de collisions.
        """
        positions = _vectors(positions, "positions")
        velocities = _vectors(velocities, "velocities")
        accelerations = _vectors(accelerations, "accelerations")
        if positions.shape != velocities.shape or positions.shape != accelerations.shape:
            raise ValueError("Les tableaux doivent avoir la même forme")
        half = self.config.arena_size / 2 - self.config.robot_radius
        if (np.abs(positions) > half).any():
            raise ValueError("Les disques des robots doivent être dans l'arène")
        if (np.linalg.norm(velocities, axis=1) > self.config.max_speed + 1e-12).any():
            raise ValueError("La vitesse initiale dépasse la limite physique")
        acceleration = _limit(accelerations, self.config.max_acceleration)
        velocity = _limit(velocities + acceleration * self.config.dt, self.config.max_speed)
        proposed = positions + velocity * self.config.dt
        contact = ((proposed >= half) & (velocity > 0)) | ((proposed <= -half) & (velocity < 0))
        return np.clip(proposed, -half, half), np.where(contact, 0.0, velocity)
