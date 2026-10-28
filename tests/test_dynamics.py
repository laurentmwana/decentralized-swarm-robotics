"""Contrat physique du nouveau plan, sans entraînement."""
import numpy as np
import pytest

from src.simulator.dynamics import AccelerationDynamics, DynamicsConfig


def engine(**kwargs):
    return AccelerationDynamics(DynamicsConfig(**dict(
        dict(arena_size=10., dt=.1, max_speed=1., max_acceleration=2., robot_radius=.2),
        **kwargs)))


def test_acceleration_and_inertia():
    model = engine()
    p, v = model.step([[0., 0.]], [[0., 0.]], [[1., 0.]])
    np.testing.assert_allclose(v, [[.1, 0.]])
    np.testing.assert_allclose(p, [[.01, 0.]])
    p, v = model.step(p, v, [[0., 0.]])
    np.testing.assert_allclose(p, [[.02, 0.]])
    np.testing.assert_allclose(v, [[.1, 0.]])


def test_acceleration_limit_is_euclidean():
    p, v = engine().step([[0., 0.]], [[0., 0.]], [[30., 40.]])
    np.testing.assert_allclose(v, [[.12, .16]])
    np.testing.assert_allclose(p, [[.012, .016]])


def test_speed_limit_is_euclidean():
    _, v = engine().step([[0., 0.]], [[.6, .8]], [[30., 40.]])
    np.testing.assert_allclose(v, [[.6, .8]])


@pytest.mark.parametrize('sign', [-1, 1])
def test_wall_keeps_disk_inside_and_preserves_tangent(sign):
    p, v = engine().step([[sign * 4.79, 0.]], [[sign * .8, .3]], [[0., 0.]])
    np.testing.assert_allclose(p, [[sign * 4.8, .03]])
    np.testing.assert_allclose(v, [[0., .3]])


def test_inputs_unchanged_and_repeatable():
    args = [np.zeros((3, 2)), np.zeros((3, 2)), np.ones((3, 2))]
    original = [a.copy() for a in args]
    first = engine().step(*args)
    second = engine().step(*args)
    for a, b in zip(args, original):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('field', ['arena_size', 'dt', 'max_speed', 'max_acceleration', 'robot_radius'])
@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_invalid_parameters(field, value):
    with pytest.raises(ValueError):
        engine(**{field: value})


def test_robot_must_fit_arena():
    with pytest.raises(ValueError):
        engine(robot_radius=5.)


@pytest.mark.parametrize('p,v,a', [
    ([[0., 0.]], [[0., 0.]], [[np.nan, 0.]]),
    ([[0., 0.]], [[0., 0.]], [[np.inf, 0.]]),
    ([[0., 0.]], [[0., 0.]], [1., 2.]),
    ([[0., 0.]], [[0., 0.], [0., 0.]], [[0., 0.]]),
    ([[4.9, 0.]], [[0., 0.]], [[0., 0.]]),
    ([[0., 0.]], [[2., 0.]], [[0., 0.]]),
])
def test_invalid_state_or_commands(p, v, a):
    with pytest.raises(ValueError):
        engine().step(p, v, a)


def test_empty_swarm():
    empty = np.empty((0, 2))
    p, v = engine().step(empty, empty, empty)
    assert p.shape == v.shape == (0, 2)
