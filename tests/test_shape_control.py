"""Professeur par places, latence d'affectation et métriques sans apprentissage."""
from dataclasses import replace

import numpy as np
import pytest

from src.reference.shape_control import ShapeGains, ShapeMetricConfig, assigned_command, execute_shape, evaluate_shape, coverage_error
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig, LocalObservation


def test_command_uses_own_target_velocity_and_local_safety():
    own = LocalObservation(np.zeros(2), np.array([1., 0.]), np.empty((0, 2)), np.empty((0, 2)), 0.)
    np.testing.assert_allclose(assigned_command(own, [1., 0.], ShapeGains()), [2., 0.])
    close = LocalObservation(np.zeros(2), np.zeros(2), np.array([[.3, 0.]]), np.zeros((1, 2)), 0.)
    assert assigned_command(close, [0., 0.], ShapeGains())[0] < 0


def test_exact_formation_still_pays_all_communication_rounds():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=50)
    mission = named_shape('square', 4, width=1.)
    positions = mission.reference_points()
    result = execute_shape(mission, config, positions)
    np.testing.assert_allclose(result.episode.positions, np.broadcast_to(positions, result.episode.positions.shape), atol=0)
    report = evaluate_shape(result, ShapeMetricConfig(.1, .05, 1., 5.))
    assert report['success']
    assert report['stabilization_time_s'] == pytest.approx(1.6)
    assert report['allocation_time_s'] == pytest.approx(1.6)
    assert report['collisions'] == 0
    assert (result.messages_per_step[:16] > 0).all()
    assert not result.messages_per_step[16:].any()


def test_disconnected_start_stays_stationary_and_fails_explicitly():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=50)
    mission = named_shape('square', 4, width=1.)
    positions = np.array([[-3., -3.], [-3., -2.], [3., 3.], [3., 2.]])
    result = execute_shape(mission, config, positions)
    np.testing.assert_array_equal(result.episode.positions[-1], positions)
    report = evaluate_shape(result, ShapeMetricConfig(.1, .05, 1., 5.))
    assert not report['allocation_resolved']
    assert not report['success']
    assert report['stabilization_time_s'] is None


def test_coverage_metric_penalizes_empty_places_and_is_permutation_invariant():
    targets = np.array([[0., 0.], [1., 0.], [2., 0.]])
    assert coverage_error(targets[::-1], targets) == 0
    assert coverage_error(np.zeros((3, 2)), targets) == 2


def test_initial_collision_is_failure_even_if_formation_recovers():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=50)
    mission = named_shape('square', 4, width=1.)
    result = execute_shape(mission, config, mission.reference_points())
    positions = result.episode.positions.copy()
    positions[0, 1] = positions[0, 0]
    result = replace(result, episode=replace(result.episode, positions=positions))
    report = evaluate_shape(result, ShapeMetricConfig(.1, .05, 1., 5.))
    assert report['initial_collision']
    assert report['collisions'] >= 1
    assert not report['success']
    assert report['stabilization_time_s'] is None


def test_movement_begins_only_after_allocation_and_respects_physical_limits():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=50)
    mission = named_shape('square', 4, width=1.)
    positions = mission.reference_points()*.5
    result = execute_shape(mission, config, positions)
    np.testing.assert_array_equal(result.episode.positions[16], positions)
    assert np.linalg.norm(result.episode.positions[17]-positions) > 0
    assert np.linalg.norm(result.episode.accelerations, axis=-1).max() <= 2+1e-12
    assert np.linalg.norm(result.episode.velocities, axis=-1).max() <= 1+1e-12


def test_short_horizon_refuses_hiding_allocation_latency():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=16)
    mission = named_shape('square', 4, width=1.)
    with pytest.raises(ValueError, match='horizon'):
        execute_shape(mission, config, mission.reference_points())


def test_pilot_exports_protocol_trajectory_and_allocation_trace(tmp_path):
    from src.reference.evaluate_shapes import evaluate_shapes
    report = evaluate_shapes(tmp_path/'pilot', names=['square'], seeds=[7000], robot_count=4, width=1., horizon=50)
    assert report['status'] == 'complete'
    assert report['requested_episodes'] == 1
    assert report['reports'][0]['status'] == 'evaluated'
    with np.load(tmp_path/'pilot/geometries_square-7000-allocation.npz') as trace:
        assert trace['assignments'].shape == (51, 4)
        assert trace['allocation_consensus'][-1]


def test_dispersed_pilot_retains_initial_states_and_uses_same_start_for_shapes(tmp_path):
    from src.reference.evaluate_shapes import evaluate_shapes
    report = evaluate_shapes(tmp_path/'dispersed', names=['square', 'geometries_trapezoid'], seeds=[7001],
        robot_count=4, width=1., horizon=50, initialization='dispersed', safety_filter=True)
    initial = np.asarray(report['initial_states']['7001'])
    assert report['initialization']['kind'] == 'dispersed'
    distances = np.linalg.norm(initial[:, None]-initial[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= .45
    for entry in report['reports']:
        with np.load(tmp_path/'dispersed'/entry['trajectory']) as trajectory:
            np.testing.assert_array_equal(trajectory['positions'][0], initial)
