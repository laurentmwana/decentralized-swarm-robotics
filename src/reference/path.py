"""Géométrie des chemins ouverts/fermés, sans affectation globale des robots."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PathProjection:
    point: np.ndarray
    coordinate: float
    tangent: np.ndarray
    distance: float
    segment: int


@dataclass(frozen=True)
class PathMission:
    vertices: tuple[tuple[float, float], ...]
    robot_count: int
    closed: bool = False

    def __post_init__(self):
        vertices = np.asarray(self.vertices, dtype=float)
        if not isinstance(self.closed, bool):
            raise ValueError('closed doit être un booléen')
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < (3 if self.closed else 2) or not np.isfinite(vertices).all():
            raise ValueError('Sommets 2D finis requis pour un chemin non vide')
        if self.closed:
            vertices = np.vstack((vertices, vertices[0]))
        lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        if (lengths <= 1e-12).any():
            raise ValueError('Segments de longueur positive requis')
        if len(vertices) == 3:
            directions = np.diff(vertices, axis=0)/lengths[:, None]
            if np.dot(directions[0], directions[1]) <= -1+1e-12:
                raise ValueError('Chemin replié sur lui-même interdit')
        if isinstance(self.robot_count, bool) or not isinstance(self.robot_count, int) or self.robot_count < 2:
            raise ValueError('Effectif entier supérieur ou égal à deux requis')
        object.__setattr__(self, 'vertices', tuple(tuple(float(v) for v in p) for p in (vertices[:-1] if self.closed else vertices)))

    @property
    def segment_vertices(self):
        points = np.asarray(self.vertices)
        return np.vstack((points, points[0])) if self.closed else points

    @property
    def lengths(self):
        return np.linalg.norm(np.diff(self.segment_vertices, axis=0), axis=1)

    @property
    def length(self):
        return float(self.lengths.sum())

    @property
    def spacing(self):
        """Espacement curviligne ; peut différer de la distance au coin du V."""
        return self.length/(self.robot_count if self.closed else self.robot_count-1)

    def point_at(self, coordinate):
        if not np.isfinite(coordinate) or not 0 <= coordinate <= self.length:
            raise ValueError('Abscisse finie requise dans le chemin')
        vertices = self.segment_vertices
        cumulative = np.r_[0., np.cumsum(self.lengths)]
        segment = min(int(np.searchsorted(cumulative, coordinate, side='right'))-1, len(self.lengths)-1)
        fraction = (coordinate-cumulative[segment])/self.lengths[segment]
        return vertices[segment] + fraction*(vertices[segment+1]-vertices[segment])

    def reference_points(self):
        """Points géométriques de référence, sans attribuer un robot à une place."""
        return np.array([self.point_at(float(s)) for s in np.linspace(0, self.length, self.robot_count, endpoint=not self.closed)])

    def project(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError('Position 2D finie requise')
        vertices = self.segment_vertices
        delta = np.diff(vertices, axis=0)
        fraction = np.clip(((position-vertices[:-1])*delta).sum(axis=1)/self.lengths**2, 0., 1.)
        points = vertices[:-1] + fraction[:, None]*delta
        distances = np.linalg.norm(position-points, axis=1)
        # En cas d'égalité exacte au sommet, le premier segment gagne.
        segment = int(np.argmin(distances))
        coordinate = float(self.lengths[:segment].sum()+fraction[segment]*self.lengths[segment])
        return PathProjection(points[segment].copy(), coordinate,
                              delta[segment]/self.lengths[segment], float(distances[segment]), segment)

    def validate_environment(self, config, safety_distance):
        if not np.isfinite(safety_distance) or safety_distance <= 2*config.physics.robot_radius:
            raise ValueError('Distance de sécurité supérieure au diamètre requise')
        half = config.physics.arena_size/2-config.physics.robot_radius
        if (np.abs(np.asarray(self.vertices)) > half).any():
            raise ValueError('La formation doit tenir dans l’arène')
        points = self.reference_points()
        distances = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < safety_distance-1e-12:
            raise ValueError('Formation trop dense en distance euclidienne')
        ordered = np.vstack((points, points[0])) if self.closed else points
        if np.linalg.norm(np.diff(ordered, axis=0), axis=1).max() > config.communication_radius+1e-12:
            raise ValueError('La portée doit couvrir les voisins consécutifs')


def _center(center):
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Centre 2D fini requis')
    return center


def line_mission(center, length, robot_count, orientation=0.):
    center = _center(center)
    if not math.isfinite(length) or length <= 0 or not math.isfinite(orientation):
        raise ValueError('Longueur positive et orientation finies requises')
    direction = np.array([math.cos(orientation), math.sin(orientation)])
    return PathMission((center-length/2*direction, center+length/2*direction), robot_count)


def v_mission(vertex, arm_length, opening_angle, robot_count, orientation=math.pi/2):
    vertex = _center(vertex)
    if not math.isfinite(arm_length) or arm_length <= 0 or not math.isfinite(opening_angle) or not 0 < opening_angle < math.pi or not math.isfinite(orientation):
        raise ValueError('Bras positif, angle dans ]0, pi[ et orientation finis requis')
    angles = np.array([orientation-opening_angle/2, orientation+opening_angle/2])
    ends = vertex + arm_length*np.column_stack((np.cos(angles), np.sin(angles)))
    return PathMission((ends[0], vertex, ends[1]), robot_count)
