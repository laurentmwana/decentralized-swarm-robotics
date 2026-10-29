"""Projection locale analytique, infaisabilité explicite et réciprocité."""
from dataclasses import replace

import numpy as np
import pytest

from src.reference.safety import project_acceleration, filter_command
from src.reference.shape_control import ShapeGains, execute_shape, evaluate_shape, ShapeMetricConfig
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig, LocalObservation


def test_projection_to_one_half_plane_and_ball():
    np.testing.assert_allclose(project_acceleration([-1, 0], [[1, 0]], [1], 2), [1, 0])
    np.testing.assert_allclose(project_acceleration([3, 4], [], [], 2), [1.2, 1.6])
    np.testing.assert_allclose(project_acceleration([0, 0], [[1, 0], [0, 1]], [1, 1], 2), [1, 1])


def test_infeasible_projection_does_not_fabricate_safe_command():
    assert project_acceleration([0, 0], [[1, 0], [-1, 0]], [1, 1], 2) is None
    assert project_acceleration([0, 0], [[1, 0]], [3], 2) is None


def test_pairwise_reciprocal_constraint_and_bounded_commands():
    gains = ShapeGains()
    left = LocalObservation(np.array([-.25, 0]), np.zeros(2), np.array([[.5, 0]]), np.zeros((1, 2)), 0.)
    right = LocalObservation(np.array([.25, 0]), np.zeros(2), np.array([[-.5, 0]]), np.zeros((1, 2)), 0.)
    a, first = filter_command(left, np.array([2., 0.]), gains, 2.)
    b, second = filter_command(right, np.array([-2., 0.]), gains, 2.)
    assert first and second
    assert np.linalg.norm(a) <= 2+1e-9 and np.linalg.norm(b) <= 2+1e-9
    r = left.position-right.position
    assert 2*np.dot(r, a-b)+gains.barrier_alpha**2*(np.dot(r, r)-gains.safety_distance**2) >= -1e-8
    assert a[1]*b[1] < 0


def test_safety_filter_is_invariant_to_neighbor_order():
    p = np.array([[.5, 0.], [0., .6], [-.7, -.2]])
    v = np.array([[-.1, 0.], [0., -.2], [.1, .1]])
    first = LocalObservation(np.zeros(2), np.zeros(2), p, v, 0.)
    second = LocalObservation(np.zeros(2), np.zeros(2), p[::-1], v[::-1], 0.)
    a, ok = filter_command(first, np.ones(2), ShapeGains(), 2)
    b, other_ok = filter_command(second, np.ones(2), ShapeGains(), 2)
    assert ok == other_ok
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_superposition_reports_filter_failure():
    obs = LocalObservation(np.zeros(2), np.ones(2), np.zeros((1, 2)), np.zeros((1, 2)), 0.)
    command, feasible = filter_command(obs, np.ones(2), ShapeGains(), 2)
    assert not feasible
    assert np.linalg.norm(command) <= 2+1e-9


def test_filter_preserves_exact_formation_equilibrium():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=50)
    mission = named_shape('square', 4, width=1.)
    result = execute_shape(mission, config, mission.reference_points(), safety_filter=True)
    report = evaluate_shape(result, ShapeMetricConfig(.1, .05, 1., 5.))
    assert report['success']
    assert report['safety_infeasible_robot_steps'] == 0
    assert report['stabilization_time_s'] == pytest.approx(1.6)
