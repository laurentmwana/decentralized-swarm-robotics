"""Professeur circulaire local : rappel radial, espacement et amortissement."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.simulator.acceleration_episode import LocalObservation, AccelerationEpisodeConfig


@dataclass(frozen=True)
class CircleMission:
    center: tuple[float, float]
    radius: float
    robot_count: int

    def __post_init__(self):
        center = np.asarray(self.center, dtype=float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Centre attendu : deux coordonnées finies")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Rayon fini et positif requis")
        if isinstance(self.robot_count, bool) or not isinstance(self.robot_count, int) or self.robot_count < 1:
            raise ValueError("Nombre de robots entier positif requis")
        object.__setattr__(self, 'center', tuple(center))

    @property
    def spacing(self):
        """Corde entre voisins d'un polygone régulier ; n=1 sans espacement."""
        return 0.0 if self.robot_count == 1 else 2 * self.radius * np.sin(np.pi / self.robot_count)


@dataclass(frozen=True)
class CircleGains:
    radial: float
    damping: float
    spacing: float
    avoidance: float
    safety_distance: float

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.radial, self.damping, self.spacing, self.avoidance, self.safety_distance)):
            raise ValueError("Les gains et la distance de sécurité doivent être finis et positifs")


@dataclass(frozen=True)
class CircleTeacher:
    mission: CircleMission
    gains: CircleGains

    @classmethod
    def from_yaml(cls, path, episode_config: AccelerationEpisodeConfig):
        with Path(path).open(encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict) or set(raw) != {'mission', 'gains'}:
            raise ValueError("Sections mission et gains requises")
        teacher = cls(CircleMission(**raw['mission']), CircleGains(**raw['gains']))
        teacher.validate_environment(episode_config)
        return teacher

    def validate_environment(self, config):
        half = config.physics.arena_size / 2 - config.physics.robot_radius
        if np.any(np.abs(self.mission.center) + self.mission.radius > half):
            raise ValueError("La formation doit tenir dans l'arène avec le rayon physique")
        if self.gains.safety_distance <= 2 * config.physics.robot_radius:
            raise ValueError("La distance de sécurité doit dépasser le diamètre physique")
        if self.mission.robot_count > 1:
            if self.mission.spacing < self.gains.safety_distance:
                raise ValueError("Formation trop dense pour la distance de sécurité")
            if config.communication_radius < max(self.mission.spacing, self.gains.safety_distance):
                raise ValueError("La portée doit couvrir l'espacement et la sécurité")

    def __call__(self, observation: LocalObservation):
        offset = observation.position - np.asarray(self.mission.center)
        distance = np.linalg.norm(offset)
        # Au centre exact, une direction commune déterministe évite la division par zéro.
        radial = offset / distance if distance > 1e-12 else np.array([1., 0.])
        acceleration = self.gains.radial * (self.mission.radius - distance) * radial
        acceleration -= self.gains.damping * observation.velocity
        for relative in observation.relative_positions:
            separation = np.linalg.norm(relative)
            if separation <= 1e-12:
                # Robots identiques superposés : aucune direction locale ne les distingue.
                continue
            away = -relative / separation
            tangent = away - np.dot(away, radial) * radial
            acceleration += self.gains.spacing * max(0., self.mission.spacing - separation) * tangent
            acceleration += self.gains.avoidance * max(0., self.gains.safety_distance - separation) * away
        return acceleration
