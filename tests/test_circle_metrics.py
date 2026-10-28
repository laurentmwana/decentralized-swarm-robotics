from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.reference.circle import CircleMission
from src.reference.metrics import (CircleMetricConfig, collision_events, evaluate_circle,
                                   formation_error, summarize)
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def circle(n=4, angle=0):
    angles = np.arange(n) * 2*np.pi/n + angle
    return np.column_stack((np.cos(angles), np.sin(angles)))


def trace(positions, velocities=None):
    positions = np.asarray(positions, dtype=float)
    cfg = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    return SimpleNamespace(positions=positions, velocities=np.zeros_like(positions) if velocities is None else velocities,
                           config=cfg, times=np.arange(len(positions))*.1)


def protocol(**changes):
    return CircleMetricConfig(**dict(dict(error_threshold=.1, speed_threshold=.05,
                                         hold_seconds=.3, deadline_seconds=1.), **changes))


def test_error_rotation_permutation_translation():
    mission = CircleMission((3, 4), 1., 4)
    assert formation_error(circle(angle=.4)[::-1] + [3, 4], mission) < 1e-12


def test_error_radial_and_missing_coverage():
    mission = CircleMission((0, 0), 1., 4)
    assert formation_error(1.2*circle(), mission) == pytest.approx(.2)
    assert formation_error(np.tile([1., 0.], (4, 1)), mission) > 1
    assert formation_error([[1.2, 0]], CircleMission((0, 0), 1., 1)) == pytest.approx(.2)


@pytest.mark.parametrize('positions', [np.zeros((3, 2)), np.full((4, 2), np.nan)])
def test_invalid_formation(positions):
    with pytest.raises(ValueError):
        formation_error(positions, CircleMission((0, 0), 1., 4))


def test_sustained_contact_counts_once_and_recontact_twice():
    positions = np.array([[[0, 0], [x, 0]] for x in [.1, .1, .5, .1, .1]])
    count, intervals, initial = collision_events(positions, .1)
    assert count == 2 and initial
    assert intervals.all()


def test_crossing_between_frames_is_detected():
    positions = np.array([[[-1., 0], [1., 0]], [[1., 0], [-1., 0]]])
    count, intervals, initial = collision_events(positions, .1)
    assert count == 1 and intervals[0] and not initial


def test_single_robot_has_no_collisions():
    count, intervals, initial = collision_events(np.zeros((3, 1, 2)), .1)
    assert count == 0 and not intervals.any() and not initial


def test_final_contact_and_tangency_count():
    count, _, _ = collision_events(np.array([[[0, 0], [.4, 0]], [[0, 0], [.2, 0]]]), .1)
    assert count == 1


def test_initially_stable_has_time_zero():
    result = evaluate_circle(trace([circle()]*11), CircleMission((0, 0), 1., 4), protocol())
    assert result['success'] and result['stabilization_time_s'] == 0
    assert result['collisions'] == 0


def test_hold_duration_not_number_of_frames():
    positions = [2*circle()]*8 + [circle()]*3
    result = evaluate_circle(trace(positions), CircleMission((0, 0), 1., 4), protocol())
    assert not result['success'] and result['stabilization_time_s'] is None
    positions = [2*circle()]*7 + [circle()]*4
    result = evaluate_circle(trace(positions), CircleMission((0, 0), 1., 4), protocol())
    assert result['success'] and result['stabilization_time_s'] == pytest.approx(.7)


def test_fast_motion_and_intermittent_stability_fail():
    mission = CircleMission((0, 0), 1., 4)
    result = evaluate_circle(trace([circle()]*11, np.ones((11, 4, 2))), mission, protocol())
    assert not result['success']
    result = evaluate_circle(trace([circle() if i%2 else 2*circle() for i in range(11)]), mission, protocol())
    assert not result['success']


def test_any_collision_prevents_success_even_after_recovery():
    positions = [np.tile([1., 0.], (4, 1))] + [circle()]*10
    report = evaluate_circle(trace(positions), CircleMission((0, 0), 1., 4), protocol())
    assert report['collisions'] >= 6 and report['initial_collision']
    assert not report['success'] and report['stabilization_time_s'] is None


def test_short_episode_is_rejected_and_post_deadline_ignored():
    mission = CircleMission((0, 0), 1., 4)
    with pytest.raises(ValueError):
        evaluate_circle(trace([circle()]*3), mission, protocol())
    report = evaluate_circle(trace([circle()]*11 + [np.zeros((4, 2))]), mission, protocol())
    assert report['success'] and report['collisions'] == 0


@pytest.mark.parametrize('field', ['error_threshold', 'speed_threshold', 'hold_seconds', 'deadline_seconds'])
@pytest.mark.parametrize('value', [0, -1, np.nan, np.inf])
def test_invalid_thresholds(field, value):
    with pytest.raises(ValueError):
        protocol(**{field: value})


def test_hold_longer_than_deadline():
    with pytest.raises(ValueError):
        protocol(hold_seconds=2.)


def test_aggregation_keeps_failures_out_of_time_mean():
    reports = [dict(success=True, stabilization_time_s=.4, formation_error_m=.01, collisions=0),
               dict(success=False, stabilization_time_s=None, formation_error_m=1., collisions=3)]
    summary = summarize(reports)
    assert summary['success_rate_percent'] == 50
    assert summary['stabilization_time_mean_s'] == .4
    assert summary['stabilization_successful_episodes'] == 1
    assert summarize(reports[1:])['stabilization_time_mean_s'] is None
    with pytest.raises(ValueError):
        summarize([])


def test_evaluation_command_exports_protocol_and_episodes(tmp_path):
    import json
    from src.reference.evaluate import main
    payload = main(['--seeds', '3', '--output', str(tmp_path)])
    assert payload['reports'][0]['seed'] == 3
    assert (tmp_path/'episode-3.npz').exists()
    saved = json.loads((tmp_path/'report.json').read_text())
    assert saved['protocol']['hold_seconds'] == 1.
    assert saved['summary']['episodes'] == 1
