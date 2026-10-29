"""Géométrie et faisabilité ; aucun entraînement et aucune affectation implicite."""
from dataclasses import replace
import math

import numpy as np
import pytest

from src.reference.path import PathMission, line_mission, v_mission
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def test_line_metric_spacing_and_projection():
    mission = line_mission((1, 2), 6, 4)
    np.testing.assert_allclose(mission.reference_points(), [[-2, 2], [0, 2], [2, 2], [4, 2]])
    assert mission.spacing == 2
    projection = mission.project((0, 5))
    np.testing.assert_allclose(projection.point, [0, 2])
    np.testing.assert_allclose(projection.tangent, [1, 0])
    assert projection.coordinate == 2
    assert projection.distance == 3
    assert mission.project((-10, 2)).coordinate == 0


def test_v_uses_arclength_and_checks_actual_corner_separation():
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    mission = v_mission((0, 0), 3, math.pi/2, 12)
    assert mission.length == pytest.approx(6)
    assert mission.spacing > .4
    # Avec 12 places, le coin réduit la séparation euclidienne sous 0,4 m.
    with pytest.raises(ValueError, match='dense'):
        mission.validate_environment(config, .4)
    v_mission((0, 0), 3.5, math.pi/2, 12).validate_environment(config, .4)


def test_closed_square_does_not_repeat_first_place():
    mission = PathMission(((-1, -1), (1, -1), (1, 1), (-1, 1)), 8, True)
    assert mission.length == 8
    assert mission.spacing == 1
    assert len(np.unique(mission.reference_points(), axis=0)) == 8
    np.testing.assert_allclose(mission.point_at(0), mission.point_at(8))


@pytest.mark.parametrize('name', ['line', 'V', 'square', 'rectangle', 'A', 'B', 'C', 'ligne', 'carré'])
def test_named_shapes_are_distinct_deterministic_and_on_strokes(name):
    mission = named_shape(name, 12, width=6)
    points = mission.reference_points()
    assert points.shape == (12, 2)
    assert np.isfinite(points).all()
    assert len(np.unique(points, axis=0)) == 12
    np.testing.assert_array_equal(points, mission.reference_points())
    assert max(mission.project(p)[1].distance for p in points) < 1e-12
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    config = replace(config, communication_radius=3.)
    mission.validate_environment(config, .4)


def test_a_crossbar_b_loops_and_c_opening_are_preserved():
    a, b, c = (named_shape(name, 24, width=4) for name in ('A', 'B', 'C'))
    assert len(a.strokes) == 2
    assert a.project((0, 0))[0] == 1
    assert len(b.strokes) == 3
    assert not any(c.closed)
    # Le centre de l'ouverture droite du C n'est pas sur le contour.
    assert c.project((2, 0))[1].distance > 1


def test_rectangle_dimensions_and_rigid_transform():
    rectangle = named_shape('rectangle', 12, width=6, height=4)
    vertices = np.asarray(rectangle.strokes[0])
    np.testing.assert_allclose(np.ptp(vertices, axis=0), [6, 4])
    shifted = named_shape('rectangle', 12, width=6, height=4, center=(1, 2), orientation=math.pi/2)
    expected = np.column_stack((-vertices[:, 1], vertices[:, 0]))+[1, 2]
    np.testing.assert_allclose(shifted.strokes[0], expected)


@pytest.mark.parametrize('kwargs', [dict(width=0), dict(height=-1), dict(orientation=float('nan')),
                                   dict(robot_count=True), dict(robot_count=1), dict(center=(0,))])
def test_invalid_shape_parameters_rejected(kwargs):
    params = dict(name='A', robot_count=12)
    params.update(kwargs)
    with pytest.raises(ValueError):
        named_shape(**params)


@pytest.mark.parametrize('case', ['walls', 'density', 'communication'])
def test_infeasible_shapes_rejected(case):
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    mission = named_shape('square', 12, width=4)
    if case == 'walls': mission = named_shape('square', 12, width=20)
    elif case == 'density': mission = named_shape('square', 100, width=1)
    else: config = replace(config, communication_radius=.01)
    with pytest.raises(ValueError):
        mission.validate_environment(config, .4)


def test_unsupported_shape_and_non_square_dimensions_rejected():
    with pytest.raises(ValueError, match='inconnue'):
        named_shape('unknown', 12)
    with pytest.raises(ValueError, match='carré'):
        named_shape('square', 12, width=4, height=3)


def test_large_b_requires_sufficient_communication_range():
    mission = named_shape('B', 12, width=6)
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    with pytest.raises(ValueError, match='connectées'):
        mission.validate_environment(config, .4)
    mission.validate_environment(replace(config, communication_radius=3.), .4)
