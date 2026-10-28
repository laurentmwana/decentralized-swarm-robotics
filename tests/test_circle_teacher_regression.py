"""Régressions issues du pilote dispersé, sans optimisation ni apprentissage."""
from dataclasses import replace

import numpy as np
import pytest

from src.reference.circle import CircleTeacher
from src.reference.metrics import CircleMetricConfig, evaluate_circle
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig


def setup():
    cfg = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    return cfg, CircleTeacher.from_yaml('config/circle-teacher.yaml', cfg), CircleMetricConfig.from_yaml('config/circle-metrics.yaml')


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_dispersed_pilot_without_initial_overlap_converges(seed):
    cfg, teacher, protocol = setup()
    result = AccelerationEpisode(cfg).execute(teacher, 12, seed)
    report = evaluate_circle(result, teacher.mission, protocol)
    assert report['success']
    assert report['collisions'] == 0
    assert report['stabilization_time_s'] + protocol.hold_seconds <= protocol.deadline_seconds
    assert report['formation_error_m'] < protocol.error_threshold
    assert np.linalg.norm(result.velocities, axis=-1).max() <= cfg.physics.max_speed + 1e-12
    assert np.linalg.norm(result.accelerations, axis=-1).max() <= cfg.physics.max_acceleration + 1e-12


def test_original_gains_reproduce_slow_angular_distribution():
    cfg, teacher, protocol = setup()
    old = replace(teacher, gains=replace(teacher.gains, radial=2., damping=3., spacing=2., avoidance=12.))
    result = AccelerationEpisode(cfg).execute(old, 12, 0)
    # Le rayon converge, mais la répartition reste insuffisante dans le délai fixé.
    assert np.max(np.abs(np.linalg.norm(result.positions[-1], axis=1)-2)) < .01
    report = evaluate_circle(result, old.mission, protocol)
    assert report['formation_error_m'] > .4
    assert not report['success']


def test_initial_collision_remains_a_failure_after_gain_correction():
    cfg, teacher, protocol = setup()
    result = AccelerationEpisode(cfg).execute(teacher, 12, 9)
    report = evaluate_circle(result, teacher.mission, protocol)
    assert report['initial_collision'] and report['collisions'] == 1
    assert not report['success'] and report['stabilization_time_s'] is None
    assert report['formation_error_m'] < protocol.error_threshold
