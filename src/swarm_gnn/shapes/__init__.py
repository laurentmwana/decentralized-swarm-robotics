"""Target shape generation for swarm formations.

Provides geometric primitives and shape I/O.
"""

import numpy as np
from pathlib import Path
from typing import Literal


def circle_points(n: int) -> np.ndarray:
    """n points uniformly distributed on unit circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def square_points(n: int) -> np.ndarray:
    """n points uniformly distributed on unit square perimeter."""
    # 4 edges, distribute points evenly
    per_edge = n // 4
    remainder = n % 4
    pts = []

    # Bottom edge: (-1, -1) to (1, -1)
    n_bottom = per_edge + (1 if remainder > 0 else 0)
    if n_bottom > 0:
        xs = np.linspace(-1, 1, n_bottom)
        pts.append(np.column_stack([xs, -np.ones(n_bottom)]))

    # Right edge: (1, -1) to (1, 1)
    n_right = per_edge + (1 if remainder > 1 else 0)
    if n_right > 0:
        ys = np.linspace(-1, 1, n_right)
        pts.append(np.column_stack([np.ones(n_right), ys]))

    # Top edge: (1, 1) to (-1, 1)
    n_top = per_edge + (1 if remainder > 2 else 0)
    if n_top > 0:
        xs = np.linspace(1, -1, n_top)
        pts.append(np.column_stack([xs, np.ones(n_top)]))

    # Left edge: (-1, 1) to (-1, -1)
    n_left = per_edge
    if n_left > 0:
        ys = np.linspace(1, -1, n_left)
        pts.append(np.column_stack([-np.ones(n_left), ys]))

    if not pts:
        return np.array([[-1.0, -1.0]])
    result = np.vstack(pts)
    return result[:n]  # trim if we have extra due to corner double-counting


def triangle_points(n: int) -> np.ndarray:
    """n points on equilateral triangle perimeter."""
    # Vertices of equilateral triangle centered at origin
    vertices = np.array([
        [0.0, 1.0],           # top
        [-np.sqrt(3)/2, -0.5], # bottom-left
        [np.sqrt(3)/2, -0.5],  # bottom-right
    ])
    return _polygon_perimeter_points(vertices, n)


def line_points(n: int) -> np.ndarray:
    """n points on horizontal line segment from (-1, 0) to (1, 0)."""
    xs = np.linspace(-1, 1, n)
    return np.column_stack([xs, np.zeros(n)])


def cross_points(n: int) -> np.ndarray:
    """n points forming a cross (horizontal + vertical lines)."""
    # Split points between horizontal and vertical
    n_h = n // 2 + n % 2
    n_v = n // 2

    # Horizontal line
    xs = np.linspace(-1, 1, n_h)
    h_pts = np.column_stack([xs, np.zeros(n_h)])

    # Vertical line
    ys = np.linspace(-1, 1, n_v)
    v_pts = np.column_stack([np.zeros(n_v), ys])

    return np.vstack([h_pts, v_pts])


def star_points(n: int) -> np.ndarray:
    """n points on 5-pointed star perimeter."""
    # 5-pointed star: outer and inner radii
    outer_r, inner_r = 1.0, 0.38
    angles = np.linspace(-np.pi/2, 3*np.pi/2, 11)[:-1]  # 10 points (5 outer, 5 inner)
    radii = np.tile([outer_r, inner_r], 5)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    # Interpolate to get exactly n points
    if len(pts) == n:
        return pts
    indices = np.linspace(0, len(pts) - 1, n).astype(int)
    return pts[indices]


def ring_points(n: int) -> np.ndarray:
    """n points on annulus (ring) between inner and outer radius."""
    inner_r, outer_r = 0.5, 1.0
    # Sample uniformly in angle and radius
    angles = np.random.uniform(0, 2*np.pi, n)
    # Uniform in area => sqrt(uniform) in radius
    radii = np.sqrt(np.random.uniform(inner_r**2, outer_r**2, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def generate_geometry(
    name: Literal["circle", "square", "triangle", "line", "cross", "star", "ring"],
    n: int,
) -> np.ndarray:
    """Generate n points for named geometry."""
    generators = {
        "circle": circle_points,
        "square": square_points,
        "triangle": triangle_points,
        "line": line_points,
        "cross": cross_points,
        "star": star_points,
        "ring": ring_points,
    }
    if name not in generators:
        raise ValueError(f"Forme géométrique inconnue : {name}")
    return generators[name](n)


def _polygon_perimeter_points(vertices: np.ndarray, n: int) -> np.ndarray:
    """Distribute n points evenly along polygon perimeter."""
    # Compute edge lengths
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_lengths = np.linalg.norm(edges, axis=1)
    perimeter = edge_lengths.sum()

    # Points per edge proportional to length
    n_per_edge = np.round(edge_lengths / perimeter * n).astype(int)
    # Fix rounding errors
    while n_per_edge.sum() < n:
        n_per_edge[np.argmax(edge_lengths)] += 1
    while n_per_edge.sum() > n:
        n_per_edge[np.argmax(n_per_edge)] -= 1

    pts = []
    for i, (v0, n_edge) in enumerate(zip(vertices, n_per_edge)):
        if n_edge > 0:
            v1 = vertices[(i + 1) % len(vertices)]
            t = np.linspace(0, 1, n_edge, endpoint=False)
            edge_pts = v0 + t[:, None] * (v1 - v0)
            pts.append(edge_pts)

    if not pts:
        return np.array([[0.0, 0.0]])
    result = np.vstack(pts)
    return result[:n]


def save_shape(points: np.ndarray, path: Path) -> None:
    """Save shape points to .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, points)


def load_shape(path: Path) -> np.ndarray:
    """Load shape points from .npy file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Shape file not found: {path}")
    return np.load(path)