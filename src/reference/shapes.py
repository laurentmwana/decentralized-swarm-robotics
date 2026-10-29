"""Consignes multitraits en mètres : lettres et figures, indépendantes du contrôleur."""
from dataclasses import dataclass
import math

import numpy as np

from .path import PathMission, _center
from .catalog import shape_definition


@dataclass(frozen=True)
class ShapeMission:
    name: str
    strokes: tuple[tuple[tuple[float, float], ...], ...]
    closed: tuple[bool, ...]
    robot_count: int
    category: str = 'custom'

    @property
    def identifier(self):
        label = self.name.upper() if self.category == 'letters' else self.name
        return f'{self.category}_{label}'

    def __post_init__(self):
        if not self.strokes or len(self.strokes) != len(self.closed):
            raise ValueError('Traits et indicateurs de fermeture compatibles requis')
        if isinstance(self.robot_count, bool) or not isinstance(self.robot_count, int) or self.robot_count < 2:
            raise ValueError('Effectif entier supérieur ou égal à deux requis')
        paths = [PathMission(stroke, 2, closed) for stroke, closed in zip(self.strokes, self.closed)]
        object.__setattr__(self, 'strokes', tuple(p.vertices for p in paths))
        object.__setattr__(self, 'closed', tuple(self.closed))

    @property
    def paths(self):
        return tuple(PathMission(stroke, 2, closed) for stroke, closed in zip(self.strokes, self.closed))

    def project(self, position):
        """Retourne le trait et sa projection ; aucun accès aux autres robots."""
        projections = [p.project(position) for p in self.paths]
        index = min(range(len(projections)), key=lambda i: projections[i].distance)
        return index, projections[index]

    def reference_points(self):
        """Couverture déterministe par points éloignés, sans affectation robot/place.

        Échantillonner les traits puis dédupliquer leurs intersections évite les
        places répétées. Cette approximation n'impose pas un espacement uniforme.
        """
        paths = self.paths
        if len(paths) == 1:
            return PathMission(paths[0].vertices, self.robot_count, paths[0].closed).reference_points()
        total = sum(p.length for p in paths)
        budget = max(256, 16*self.robot_count)
        candidates = []
        for path in paths:
            count = max(2, math.ceil(budget*path.length/total))
            candidates.extend(path.point_at(float(s)) for s in np.linspace(0, path.length, count, endpoint=not path.closed))
            candidates.extend(np.asarray(path.vertices))
        candidates = np.unique(np.asarray(candidates), axis=0)
        if len(candidates) < self.robot_count:
            raise ValueError('Pas assez de points géométriques distincts')
        chosen = [0]
        distances = np.sum((candidates-candidates[0])**2, axis=1)
        for _ in range(self.robot_count-1):
            index = int(np.argmax(distances))
            if distances[index] <= 0:
                raise ValueError('Impossible de créer des places distinctes')
            chosen.append(index)
            distances = np.minimum(distances, np.sum((candidates-candidates[index])**2, axis=1))
        return candidates[chosen].copy()

    def validate_environment(self, config, safety_distance):
        if not np.isfinite(safety_distance) or safety_distance <= 2*config.physics.robot_radius:
            raise ValueError('Distance de sécurité supérieure au diamètre requise')
        vertices = np.concatenate([np.asarray(stroke) for stroke in self.strokes])
        half = config.physics.arena_size/2-config.physics.robot_radius
        if (np.abs(vertices) > half).any():
            raise ValueError('La formation doit tenir dans l’arène')
        points = self.reference_points()
        distances = np.linalg.norm(points[:, None]-points[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if distances.min() < safety_distance-1e-12:
            raise ValueError('Formation trop dense en distance euclidienne')
        adjacency = distances <= config.communication_radius+1e-12
        visited = {0}
        while True:
            expanded = visited | set(np.flatnonzero(adjacency[list(visited)].any(axis=0)))
            if expanded == visited:
                break
            visited = expanded
        if len(visited) != self.robot_count:
            raise ValueError('Les places ne sont pas connectées à cette portée')


def named_shape(name, robot_count, width=6., height=None, center=(0., 0.), orientation=0.):
    """Construire une entrée du catalogue en mètres, sans affecter les robots."""
    category, token, strokes, closed = shape_definition(name)
    center = _center(center)
    height = (width*2/3 if category == 'geometries' and token == 'rectangle' else width) if height is None else height
    if not all(math.isfinite(v) for v in (width, height, orientation)) or min(width, height) <= 0:
        raise ValueError('Dimensions positives et orientation finies requises')
    if category == 'geometries' and token in ('square', 'circle') and width != height:
        raise ValueError('Un carré ou cercle exige largeur et hauteur égales')
    rotation = np.array([[math.cos(orientation), -math.sin(orientation)],
                         [math.sin(orientation), math.cos(orientation)]])
    transformed = tuple(tuple(map(tuple, (np.asarray(stroke)*[width, height])@rotation.T+center)) for stroke in strokes)
    return ShapeMission(token, transformed, closed, robot_count, category)
