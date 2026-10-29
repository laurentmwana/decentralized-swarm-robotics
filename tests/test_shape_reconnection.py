"""Rendez-vous local, freinage physique et entrée dans l'affectation."""
from dataclasses import replace

import numpy as np
import pytest

from src.reference.reconnection import ReconnectionConfig, rendezvous_targets
from src.reference.shape_control import ShapeGains, ShapeMetricConfig, execute_shape, evaluate_shape
from src.reference.safety import filter_command
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig, LocalObservation


def test_shared_rendezvous_geometry_is_connected_and_separated():
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    target = rendezvous_targets(12, config, ReconnectionConfig(), .4)
    distances = np.linalg.norm(target[:, None]-target[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() == pytest.approx(.6)
    assert (distances.min(axis=1) <= config.communication_radius).all()
    assert ReconnectionConfig().steps(.1) == (80, 10)


def test_benign_neighbor_does_not_perturb_near_goal_with_risk_detours():
    obs = LocalObservation(np.zeros(2), np.zeros(2), np.array([[.8, 0.]]), np.zeros((1, 2)), 0.)
    gains = ShapeGains(detour_only_on_constraint=True)
    command, feasible = filter_command(obs, np.array([.06, 0.]), gains, 2)
    assert feasible
    np.testing.assert_allclose(command, [.06, 0.], atol=1e-12)
    legacy, _ = filter_command(obs, np.array([.06, 0.]), ShapeGains(), 2)
    assert abs(legacy[1]) > .1


def test_disconnected_start_reconnects_without_teleportation_or_free_gossip():
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=250)
    initial = np.array([[-3., -3.], [-3., -2.], [3., 3.], [3., 2.]])
    mission = named_shape('square', 4, width=2.)
    result = execute_shape(mission, config, initial, gains=ShapeGains(detour_only_on_constraint=True),
                           safety_filter=True, reconnection=ReconnectionConfig(6., 1.))
    report = evaluate_shape(result, ShapeMetricConfig(.1, .05, 1., 25.))
    assert report['allocation_resolved']
    assert report['reconnection_time_s'] == 7
    assert report['control_release_time_s'] == pytest.approx(8.6)
    assert report['allocation_graph_changed_steps'] == 0
    assert report['max_allocation_speed_m_s'] < 1e-7
    assert report['collisions'] == 0
    assert report['success']
    assert not result.messages_per_step[:70].any()
    assert (result.messages_per_step[70:86] > 0).all()
    assert np.array_equal(result.assignments[0], [-1]*4)
    assert np.linalg.norm(np.diff(result.episode.positions, axis=0), axis=-1).max() <= .1+1e-12
    assert np.linalg.norm(result.episode.accelerations, axis=-1).max() <= 2+1e-12
    assert set(result.phases) == {0, 1, 2, 3}


def test_regression_isolated_robot_seed_7001_reconnects_and_forms_square():
    from src.imitation.data import nonoverlapping_positions
    from src.simulator.acceleration_episode import AccelerationEpisode
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),
                     horizon=400, communication_radius=3.)
    positions = nonoverlapping_positions(config, 12, 7001, .45, 20000)
    graph = AccelerationEpisode(config).adjacency(positions)
    assert (graph.sum(axis=1) == 0).sum() == 1
    result = execute_shape(named_shape('square', 12), config, positions,
        gains=ShapeGains(detour_only_on_constraint=True), safety_filter=True, reconnection=ReconnectionConfig())
    report = evaluate_shape(result, ShapeMetricConfig(.1, .05, 1., 40.))
    assert report['allocation_resolved'] and report['allocation_stationary']
    assert report['collisions'] == 0 and report['success']
    assert report['control_release_time_s'] == pytest.approx(23.4)
    assert [int((result.phases == k).sum()) for k in range(4)] == [80, 10, 144, 166]


@pytest.mark.parametrize('case', ['time', 'spacing', 'center', 'walls', 'horizon', 'no_filter'])
def test_invalid_reconnection_contract_rejected(case):
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    mission = named_shape('square', 4, width=1.)
    with pytest.raises(ValueError):
        if case == 'time': ReconnectionConfig(gather_seconds=0)
        elif case == 'spacing': rendezvous_targets(12, config, ReconnectionConfig(spacing=.3), .4)
        elif case == 'center': ReconnectionConfig(center=(float('nan'), 0.))
        elif case == 'walls': rendezvous_targets(120, config, ReconnectionConfig(), .4)
        elif case == 'horizon': execute_shape(mission, config, mission.reference_points(), safety_filter=True, reconnection=ReconnectionConfig())
        else: execute_shape(mission, replace(config, horizon=200), mission.reference_points(), reconnection=ReconnectionConfig())
