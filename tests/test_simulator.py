import numpy as np
import pytest

from src.sim import SimulationConfig, SwarmSimulator, local_gathering_and_repulsion


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(
        seed=7,
        robot_count=3,
        arena_size=(10.0, 10.0),
        communication_radius=2.0,
        max_speed=1.0,
        time_step=0.5,
        episode_steps=10,
        safety_distance=0.5,
    )


def test_neighbor_detection_respects_radius_and_activity(config: SimulationConfig) -> None:
    simulator = SwarmSimulator(config, positions=np.array([[0, 0], [1, 0], [4, 0]], dtype=float))
    assert simulator.neighbor_matrix().tolist() == [[False, True, False], [True, False, False], [False, False, False]]
    simulator.fail([1])
    assert not simulator.neighbor_matrix().any()


def test_commands_are_limited_by_max_speed(config: SimulationConfig) -> None:
    simulator = SwarmSimulator(config, positions=np.zeros((3, 2)))
    simulator.step(np.array([[3.0, 4.0], [0.0, 10.0], [0.0, 0.0]]))
    assert np.all(np.linalg.norm(simulator.velocities, axis=1) <= config.max_speed)


def test_failed_robot_stays_still_and_is_not_a_collision(config: SimulationConfig) -> None:
    simulator = SwarmSimulator(config, positions=np.array([[1, 1], [1.1, 1], [5, 5]], dtype=float))
    simulator.fail([0])
    simulator.step(np.ones((3, 2)))
    assert np.allclose(simulator.positions[0], [1, 1])
    assert simulator.velocities[0].tolist() == [0.0, 0.0]
    assert simulator.collision_pairs() == []


def test_same_seed_produces_same_initial_state(config: SimulationConfig) -> None:
    first, second = SwarmSimulator(config), SwarmSimulator(config)
    assert np.allclose(first.positions, second.positions)
    first.reset()
    assert np.allclose(first.positions, second.positions)


def test_local_rule_is_valid_for_an_isolated_robot(config: SimulationConfig) -> None:
    simulator = SwarmSimulator(config, positions=np.array([[0, 0], [5, 5], [9, 9]], dtype=float))
    commands = local_gathering_and_repulsion(simulator)
    assert commands.shape == (3, 2)
    assert np.allclose(commands, 0.0)
