"""Tests for target shape generation (Phase 2)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.swarm_gnn.shapes import (
    circle_points,
    square_points,
    triangle_points,
    line_points,
    cross_points,
    star_points,
    ring_points,
    generate_geometry,
    save_shape,
    load_shape,
)


class TestGeometryShapes:
    """Tests for basic geometric shapes."""

    def test_circle_points(self):
        """Circle points are on unit circle."""
        pts = circle_points(20)
        assert pts.shape == (20, 2)
        radii = np.linalg.norm(pts, axis=1)
        assert np.allclose(radii, 1.0, atol=1e-6)

    def test_square_points(self):
        """Square points form a square perimeter."""
        pts = square_points(20)
        assert pts.shape == (20, 2)
        # Points should be on perimeter of unit square [-1, 1] x [-1, 1]
        assert np.all(np.abs(pts) <= 1.0 + 1e-6)
        # At least some points on each edge
        assert np.any(np.abs(pts[:, 0]) > 0.99)  # left/right edges
        assert np.any(np.abs(pts[:, 1]) > 0.99)  # top/bottom edges

    def test_triangle_points(self):
        """Triangle points form equilateral triangle."""
        pts = triangle_points(21)
        assert pts.shape == (21, 2)
        # All points within bounding box
        assert np.all(np.abs(pts) <= 1.0 + 1e-6)

    def test_line_points(self):
        """Line points on horizontal segment."""
        pts = line_points(10)
        assert pts.shape == (10, 2)
        assert np.allclose(pts[:, 1], 0.0)  # y = 0
        assert np.all(np.abs(pts[:, 0]) <= 1.0)

    def test_cross_points(self):
        """Cross points form horizontal + vertical lines."""
        pts = cross_points(20)
        assert pts.shape == (20, 2)
        # Points on x=0 or y=0
        on_horizontal = np.abs(pts[:, 1]) < 1e-6
        on_vertical = np.abs(pts[:, 0]) < 1e-6
        assert np.all(on_horizontal | on_vertical)

    def test_star_points(self):
        """Star points form 5-pointed star."""
        pts = star_points(20)
        assert pts.shape == (20, 2)
        assert np.all(np.abs(pts) <= 1.0 + 1e-6)

    def test_ring_points(self):
        """Ring points form annulus."""
        pts = ring_points(20)
        assert pts.shape == (20, 2)
        radii = np.linalg.norm(pts, axis=1)
        # Should be between inner (0.5) and outer (1.0) radius
        assert np.all(radii >= 0.5)
        assert np.all(radii <= 1.0)


class TestGenerateGeometry:
    """Tests for generate_geometry function."""

    def test_all_geometries(self):
        """generate_geometry works for all shape names."""
        for name in ["circle", "square", "triangle", "line", "cross", "star", "ring"]:
            pts = generate_geometry(name, 20)
            assert pts.shape == (20, 2)

    def test_unknown_geometry_raises(self):
        """Unknown geometry name raises ValueError."""
        with pytest.raises(ValueError, match="Forme géométrique inconnue"):
            generate_geometry("unknown", 10)


class TestShapeIO:
    """Tests for save/load shape points."""

    def test_save_load_roundtrip(self):
        """save_shape + load_shape preserves points."""
        pts = circle_points(15)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_shape.npy"
            save_shape(pts, path)
            loaded = load_shape(path)
            assert np.allclose(pts, loaded)

    def test_load_nonexistent_raises(self):
        """Loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_shape(Path("nonexistent.npy"))

    def test_save_creates_parent_dirs(self):
        """save_shape creates parent directories."""
        pts = circle_points(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "shape.npy"
            save_shape(pts, path)
            assert path.exists()