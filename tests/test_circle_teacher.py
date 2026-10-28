from dataclasses import replace

import numpy as np
import pytest

from src.reference.circle import CircleTeacher, CircleMission, CircleGains
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig, LocalObservation


def setup():
    cfg = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    return cfg, CircleTeacher.from_yaml('config/circle-teacher.yaml', cfg)


def obs(position, velocity=(0, 0), neighbors=()):
    return LocalObservation(np.array(position, dtype=float), np.array(velocity, dtype=float),
                            np.array(neighbors, dtype=float).reshape(-1, 2),
                            np.zeros((len(neighbors), 2)), 0.)


def test_radial_restoring_force_and_equilibrium():
    _, teacher = setup()
    assert teacher(obs([3, 0]))[0] < 0
    assert teacher(obs([1, 0]))[0] > 0
    np.testing.assert_allclose(teacher(obs([2, 0])), [0, 0])


def test_braking_and_center_are_finite():
    _, teacher = setup()
    teacher = replace(teacher, gains=replace(teacher.gains, damping=3.))
    np.testing.assert_allclose(teacher(obs([2, 0], [0, 1])), [0, -3])
    assert np.isfinite(teacher(obs([0, 0]))).all()
    assert teacher(obs([0, 0]))[0] > 0


def test_tangential_spacing_and_safety_repulsion():
    _, teacher = setup()
    command = teacher(obs([2, 0], neighbors=[[0, .1]]))
    assert command[1] < 0
    # Voisin proche radial : répulsion physique même sans composante tangentielle.
    assert teacher(obs([2, 0], neighbors=[[.1, 0]]))[0] < 0


def test_neighbor_order_does_not_change_command():
    _, teacher = setup()
    neighbors = [[.1, .2], [-.2, -.3], [.9, .1]]
    np.testing.assert_allclose(teacher(obs([2, 0], neighbors=neighbors)),
                               teacher(obs([2, 0], neighbors=neighbors[::-1])))


def test_regular_circle_is_stationary():
    cfg, teacher = setup()
    angles = np.arange(12) * 2 * np.pi / 12
    initial = 2 * np.column_stack((np.cos(angles), np.sin(angles)))
    result = AccelerationEpisode(replace(cfg, horizon=10)).execute(teacher, 12, 0, initial)
    np.testing.assert_allclose(result.positions[-1], initial, atol=1e-12)


def test_isolated_robot_converges_and_stops():
    cfg, teacher = setup()
    teacher = replace(teacher, mission=replace(teacher.mission, robot_count=1))
    result = AccelerationEpisode(replace(cfg, horizon=200)).execute(teacher, 1, 0, [[3.5, 0]])
    assert abs(np.linalg.norm(result.positions[-1, 0]) - 2) < .001
    assert np.linalg.norm(result.velocities[-1, 0]) < .001


def test_perturbed_circle_converges_without_overlaps():
    cfg, teacher = setup()
    angles = np.arange(12) * 2 * np.pi / 12 + .07 * np.sin(np.arange(12))
    radii = 2.4 + .2 * np.cos(np.arange(12))
    initial = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    result = AccelerationEpisode(replace(cfg, horizon=400)).execute(teacher, 12, 0, initial)
    assert np.max(np.abs(np.linalg.norm(result.positions[-1], axis=1) - 2)) < .01
    assert np.max(np.linalg.norm(result.velocities[-1], axis=1)) < .01
    distances = np.linalg.norm(result.positions[:, :, None] - result.positions[:, None, :], axis=-1)
    assert distances[:, ~np.eye(12, dtype=bool)].min() > 2 * cfg.physics.robot_radius
    final_angles = np.sort(np.mod(np.arctan2(result.positions[-1, :, 1], result.positions[-1, :, 0]), 2*np.pi))
    gaps = np.diff(np.r_[final_angles, final_angles[0]+2*np.pi])
    assert np.max(np.abs(gaps - 2*np.pi/12)) < .06


@pytest.mark.parametrize('change', [dict(radius=0), dict(radius=np.nan), dict(center=[0, np.inf]),
    dict(robot_count=0), dict(robot_count=True)])
def test_invalid_mission(change):
    with pytest.raises(ValueError):
        replace(setup()[1].mission, **change)


@pytest.mark.parametrize('field', ['radial', 'damping', 'spacing', 'avoidance', 'safety_distance'])
def test_invalid_gains(field):
    with pytest.raises(ValueError):
        replace(setup()[1].gains, **{field: 0})


@pytest.mark.parametrize('case', ['outside', 'dense', 'short_range', 'diameter'])
def test_incompatible_environment(case):
    cfg, teacher = setup()
    if case == 'outside':
        teacher = replace(teacher, mission=replace(teacher.mission, center=(4, 0)))
    elif case == 'dense':
        teacher = replace(teacher, mission=replace(teacher.mission, robot_count=100))
    elif case == 'short_range':
        cfg = replace(cfg, communication_radius=.5)
    else:
        teacher = replace(teacher, gains=replace(teacher.gains, safety_distance=.1))
    with pytest.raises(ValueError):
        teacher.validate_environment(cfg)


def test_demo_exports_trajectory_and_mission(tmp_path):
    import json
    from src.reference.demo import main
    result = main(['--output', str(tmp_path), '--seed', '9'])
    assert result.seed == 9
    with np.load(tmp_path / 'episode.npz', allow_pickle=False) as data:
        assert data['positions'].shape == (101, 12, 2)
    metadata = json.loads((tmp_path / 'teacher.json').read_text())
    assert metadata['mission']['radius'] == 2
    assert metadata['gains']['damping'] == 6


def test_close_pair_separates_from_rest():
    cfg, teacher = setup()
    result = AccelerationEpisode(replace(cfg, horizon=20)).execute(
        teacher, 2, 0, [[2, -.11], [2, .11]])
    separation = np.linalg.norm(result.positions[:, 0] - result.positions[:, 1], axis=1)
    assert separation.min() >= .22 - 1e-12
    assert separation[-1] > teacher.gains.safety_distance
