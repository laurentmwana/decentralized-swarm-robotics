import json
from dataclasses import replace

import numpy as np
import pytest

from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig


def config():
    return replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=3)


def zero(observation):
    return np.zeros(2)


def test_reproducibility_and_seed_variation():
    episode = AccelerationEpisode(config())
    a, b, c = [episode.execute(zero, 5, seed) for seed in (7, 7, 8)]
    np.testing.assert_array_equal(a.positions, b.positions)
    assert not np.array_equal(a.positions[0], c.positions[0])
    assert a.positions.shape == a.velocities.shape == (4, 5, 2)
    np.testing.assert_array_equal(a.positions[0], a.positions[-1])


def test_local_observations_and_range_boundary():
    observations = []
    def controller(obs):
        observations.append(obs)
        return [0, 0]
    AccelerationEpisode(replace(config(), horizon=1)).execute(
        controller, 3, 0, [[0, 0], [2, 0], [-3, 0]], [[.1, 0], [.3, 0], [0, 0]])
    np.testing.assert_allclose(observations[0].relative_positions, [[2, 0]])
    np.testing.assert_allclose(observations[0].relative_velocities, [[.2, 0]])
    assert observations[2].relative_positions.shape == (0, 2)
    assert observations[2].relative_velocities.shape == (0, 2)
    assert not observations[0].position.flags.writeable


def test_graph_updates_and_includes_coincident_robots():
    cfg = replace(config(), horizon=1, communication_radius=.2)
    result = AccelerationEpisode(cfg).execute(zero, 2, 0, [[0, 0], [.19, 0]], [[-.5, 0], [.5, 0]])
    assert result.adjacency[0, 0, 1]
    assert not result.adjacency[1, 0, 1]
    graph = AccelerationEpisode(cfg).adjacency(np.zeros((2, 2)))
    assert graph[0, 1] and graph[1, 0]
    assert not graph.diagonal().any()


def test_bounded_commands_and_inertia_history():
    result = AccelerationEpisode(config()).execute(lambda obs: [300, 400], 1, 0, [[0, 0]])
    np.testing.assert_allclose(result.requested_accelerations[:, 0], [[300, 400]] * 3)
    np.testing.assert_allclose(result.accelerations[:, 0], [[1.2, 1.6]] * 3)
    np.testing.assert_allclose(result.velocities[1, 0], [.12, .16])
    np.testing.assert_allclose(result.positions[1, 0], [.012, .016])
    np.testing.assert_allclose(result.times, [0, .1, .2, .3])


def test_archive_is_self_contained(tmp_path):
    result = AccelerationEpisode(config()).execute(zero, 2, 9)
    path = tmp_path / 'episode.npz'
    result.save(path)
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['positions'], result.positions)
        np.testing.assert_array_equal(data['adjacency'], result.adjacency)
        metadata = json.loads(str(data['metadata']))
        assert metadata['seed'] == 9
        assert metadata['units']['position'] == 'm'
        assert metadata['config']['physics']['dt'] == .1


@pytest.mark.parametrize('command', [[np.nan, 0], [np.inf, 0], [0], [[0, 0]]])
def test_invalid_controller_output(command):
    with pytest.raises(ValueError):
        AccelerationEpisode(config()).execute(lambda obs: command, 1, 0)


@pytest.mark.parametrize('change', [dict(horizon=0), dict(horizon=1.5), dict(horizon=True),
    dict(communication_radius=0), dict(communication_radius=np.nan),
    dict(spawn_radius=5), dict(spawn_radius=-1)])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(config(), **change)


@pytest.mark.parametrize('n,seed,p,v', [(0, 0, None, None), (True, 0, None, None),
    (1, -1, None, None), (1, 0, [[0, 0], [1, 1]], None),
    (1, 0, [[5, 0]], None), (1, 0, [[0, 0]], [[2, 0]])])
def test_invalid_initial_state_before_controller(n, seed, p, v):
    def never(obs):
        pytest.fail('Contrôleur appelé avant validation')
    with pytest.raises(ValueError):
        AccelerationEpisode(config()).execute(never, n, seed, p, v)


def test_initial_arrays_untouched():
    p, v = np.zeros((1, 2)), np.array([[.1, 0]])
    AccelerationEpisode(config()).execute(zero, 1, 0, p, v)
    np.testing.assert_array_equal(p, [[0, 0]])
    np.testing.assert_array_equal(v, [[.1, 0]])


def test_observations_are_simultaneous_snapshots():
    seen = []
    def controller(obs):
        seen.append(obs)
        return [1, 0]
    result = AccelerationEpisode(replace(config(), horizon=1)).execute(
        controller, 2, 0, [[0, 0], [1, 0]])
    np.testing.assert_array_equal(seen[1].relative_positions, [[-1, 0]])
    np.testing.assert_array_equal(seen[0].position, [0, 0])
    assert result.positions[-1, 0, 0] > 0


def test_distant_robot_cannot_change_local_observation():
    def capture(far):
        seen = []
        def controller(obs):
            seen.append(obs)
            return [0, 0]
        AccelerationEpisode(replace(config(), horizon=1)).execute(
            controller, 2, 0, [[0, 0], far])
        return seen[0]
    a, b = capture([3, 0]), capture([-4, 0])
    for name in ('position', 'velocity', 'relative_positions', 'relative_velocities'):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.parametrize('content', ['[]', 'physics: 4', '''physics:
  arena_size: 10
  dt: 0.1
  max_speed: 1
  max_acceleration: 2
  robot_radius: 0.1
horizon: 2
communication_radius: 1
spawn_radius: 2
unknown: true
'''])
def test_malformed_configuration(tmp_path, content):
    path = tmp_path / 'config.yaml'
    path.write_text(content)
    with pytest.raises((ValueError, TypeError)):
        AccelerationEpisodeConfig.from_yaml(path)
