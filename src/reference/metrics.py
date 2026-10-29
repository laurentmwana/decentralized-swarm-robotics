"""Quatre métriques du cercle en unités SI, sans dépendance à l'entraînement."""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import yaml

from .circle import CircleMission


@dataclass(frozen=True)
class CircleMetricConfig:
    error_threshold: float
    speed_threshold: float
    hold_seconds: float
    deadline_seconds: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (
            self.error_threshold, self.speed_threshold, self.hold_seconds, self.deadline_seconds)):
            raise ValueError('Seuils et durées finis et positifs requis')
        if self.hold_seconds > self.deadline_seconds:
            raise ValueError('Le maintien doit tenir dans le délai maximal')

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            return cls(**yaml.safe_load(stream))


def formation_error(positions, mission: CircleMission):
    """Max de l'écart radial et de l'écart de répartition angulaire exprimé en mètres."""
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (mission.robot_count, 2) or not np.isfinite(positions).all():
        raise ValueError('Positions incompatibles avec la consigne')
    offsets = positions - np.asarray(mission.center)
    radial = np.max(np.abs(np.linalg.norm(offsets, axis=1) - mission.radius))
    if mission.robot_count == 1:
        return float(radial)
    angles = np.sort(np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    spacing = mission.radius * np.max(np.abs(gaps - 2 * np.pi / mission.robot_count))
    return float(max(radial, spacing))


def collision_timeline(positions, robot_radius):
    """Contacts de disques : initiaux + entrées, y compris traversées entre deux états.

    Chaque segment de trajectoire est interpolé linéairement. Un contact prolongé
    compte une fois ; deux contacts séparés d'une même paire comptent deux fois.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2 or len(positions) < 1 or not np.isfinite(positions).all():
        raise ValueError('Trajectoires finies (T+1, N, 2) requises')
    if not np.isfinite(robot_radius) or robot_radius <= 0:
        raise ValueError('Rayon physique fini et positif requis')
    i, j = np.triu_indices(positions.shape[1], 1)
    relative = positions[:, i] - positions[:, j]
    threshold = (2 * robot_radius) ** 2
    initial = np.sum(relative[0] ** 2, axis=-1) <= threshold + 1e-12
    initial_count = int(initial.sum())
    counts = []
    intervals = []
    for start, end in zip(relative[:-1], relative[1:]):
        movement = end - start
        norm2 = np.sum(movement ** 2, axis=-1)
        fraction = np.clip(-np.sum(start * movement, axis=-1) / np.maximum(norm2, 1e-30), 0, 1)
        nearest = start + fraction[:, None] * movement
        contact = np.sum(nearest ** 2, axis=-1) <= threshold + 1e-12
        at_start = np.sum(start ** 2, axis=-1) <= threshold + 1e-12
        counts.append(int((contact & ~at_start).sum()))
        intervals.append(bool(contact.any()))
    return np.array(counts, dtype=int), np.array(intervals, dtype=bool), initial_count


def collision_events(positions, robot_radius):
    counts, intervals, initial = collision_timeline(positions, robot_radius)
    return int(counts.sum())+initial, intervals, initial > 0


def evaluate_circle(result, mission, config: CircleMetricConfig):
    dt = result.config.physics.dt
    if result.positions.shape != result.velocities.shape or not np.isfinite(result.velocities).all():
        raise ValueError('Historique de vitesses invalide')
    if result.times[-1] + 1e-12 < config.deadline_seconds:
        raise ValueError('Épisode trop court pour le délai du protocole')
    end = int(math.floor(config.deadline_seconds / dt + 1e-9))
    positions, velocities = result.positions[:end+1], result.velocities[:end+1]
    errors = np.array([formation_error(p, mission) for p in positions])
    collisions, contacts, initial_contact = collision_events(positions, result.config.physics.robot_radius)
    speeds = np.linalg.norm(velocities, axis=-1).max(axis=1)
    stable = (errors <= config.error_threshold) & (speeds <= config.speed_threshold)
    hold_steps = int(math.ceil(config.hold_seconds / dt - 1e-9))
    stabilization = None
    # Tout contact pendant le scénario constitue un échec, même si le cercle se rétablit.
    if collisions == 0:
        for start in range(len(positions) - hold_steps):
            stop = start + hold_steps
            if stable[start:stop+1].all() and not contacts[start:stop].any():
                stabilization = float(start * dt)
                break
    return dict(formation_error_m=float(errors[-1]), stabilization_time_s=stabilization,
                collisions=collisions, success=stabilization is not None,
                initial_collision=initial_contact)


def summarize(reports):
    reports = list(reports)
    if not reports:
        raise ValueError('Au moins un épisode requis')
    times = [r['stabilization_time_s'] for r in reports if r['success']]
    return dict(episodes=len(reports), success_rate_percent=100 * sum(r['success'] for r in reports)/len(reports),
                formation_error_mean_m=float(np.mean([r['formation_error_m'] for r in reports])),
                formation_error_std_m=float(np.std([r['formation_error_m'] for r in reports])),
                collisions_total=sum(r['collisions'] for r in reports),
                stabilization_time_mean_s=float(np.mean(times)) if times else None,
                stabilization_successful_episodes=len(times))
