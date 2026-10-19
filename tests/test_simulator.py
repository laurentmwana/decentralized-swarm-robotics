import numpy as np
import pytest

from src.sim import SimulationConfig, SwarmSimulator, local_gathering_and_repulsion, local_manual_with_shape_consensus


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


def test_random_failures_at_midpoint(config: SimulationConfig) -> None:
    """Test aléatoire de pannes à mi-épisode avec fraction spécifiée."""
    # Create positions matching the config robot count
    positions = np.zeros((config.robot_count, 2))
    simulator = SwarmSimulator(config, positions=positions)
    initial_active = np.sum(simulator.active)
    assert initial_active == config.robot_count

    # Apply 50% failures
    simulator.apply_random_failures_at_midpoint(0.5)
    active_after = np.sum(simulator.active)
    # Should be approximately half (allowing for randomness)
    expected_min = max(1, config.robot_count // 2 - 1)
    expected_max = config.robot_count // 2 + 1
    assert expected_min <= active_after <= expected_max, f"Expected {expected_min}-{expected_max} active, got {active_after}"

    # Verify failed robots don't move
    positions_before = simulator.positions.copy()
    simulator.step(np.ones((config.robot_count, 2)))  # Try to move all robots
    positions_after = simulator.positions.copy()

    failed_mask = ~simulator.active
    if np.any(failed_mask):
        position_changes = np.linalg.norm(
            positions_after[failed_mask] - positions_before[failed_mask], axis=1
        )
        assert np.allclose(position_changes, 0.0), "Failed robots should not move"


def test_manual_rule_with_shape_consensus_isolated_robot() -> None:
    """Test de la règle manuelle avec consigne de forme pour un robot isolé."""
    config = SimulationConfig(
        seed=7,
        robot_count=1,
        arena_size=(10.0, 10.0),
        communication_radius=2.0,
        max_speed=1.0,
        time_step=0.5,
        episode_steps=10,
        safety_distance=0.5,
    )
    # Positionner le robot loin de la cible
    positions = np.array([[0.0, 0.0]], dtype=float)
    # Cible à proximité du robot
    target_points = np.array([[5.0, 5.0]], dtype=float)  # Un seul point cible

    simulator = SwarmSimulator(config, positions=positions)
    commands = local_manual_with_shape_consensus(simulator, target_points)

    # Un robot isolé devrait être attiré vers la cible
    expected_attraction = target_points[0] - positions[0]  # [5, 5] - [0, 0] = [5, 5]
    expected_commands = expected_attraction  # Pas de répulsion pour un robot isolé

    assert commands.shape == (1, 2)
    assert np.allclose(commands[0], expected_commands)


def test_manual_rule_with_shape_consensus_with_neighbors() -> None:
    """Test de la règle manuelle avec consigne de forme avec des voisins."""
    config = SimulationConfig(
        seed=7,
        robot_count=2,
        arena_size=(10.0, 10.0),
        communication_radius=2.0,
        max_speed=1.0,
        time_step=0.5,
        episode_steps=10,
        safety_distance=0.5,
    )
    # Deux robots proches l'un de l'autre
    positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    # Cible éloignée des deux robots
    target_points = np.array([
        [5.0, 5.0],  # Premier robot cible
        [5.0, 5.0]   # Deuxième robot même cible
    ], dtype=float)

    simulator = SwarmSimulator(config, positions=positions)
    commands = local_manual_with_shape_consensus(simulator, target_points)

    # Les deux robots devraient avoir des commandes non nulles (attraction + répulsion)
    assert commands.shape == (2, 2)
    # Les commandes ne devraient pas être nulles
    assert not np.allclose(commands[0], 0.0) or not np.allclose(commands[1], 0.0)


def test_manual_rule_with_shape_consensus_no_collision_when_far() -> None:
    """Test que la répulsion ne s'active pas quand les robots sont éloignés."""
    config = SimulationConfig(
        seed=7,
        robot_count=2,
        arena_size=(10.0, 10.0),
        communication_radius=2.0,
        max_speed=1.0,
        time_step=0.5,
        episode_steps=10,
        safety_distance=0.5,
    )
    # Deux robots éloignés (plus loin que la distance de sécurité)
    positions = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=float)  # Distance de 3 > safety_distance de 0.5
    # Cible placée ailleurs
    target_points = np.array([
        [5.0, 5.0],
        [5.0, 5.0]
    ], dtype=float)

    simulator = SwarmSimulator(config, positions=positions)
    commands = local_manual_with_shape_consensus(simulator, target_points)

    # Comme les robots sont éloignés, il ne devrait y avoir que de l'attraction (pas de répulsion)
    # Chaque robot devrait être attiré vers son point cible
    expected_attraction_0 = target_points[0] - positions[0]  # [5, 5] - [0, 0] = [5, 5]
    expected_attraction_1 = target_points[1] - positions[1]  # [5, 5] - [3, 0] = [2, 5]

    assert commands.shape == (2, 2)
    # Avec des gains par défaut de 1.0, les commandes devraient être égales à l'attraction
    assert np.allclose(commands[0], expected_attraction_0)
    assert np.allclose(commands[1], expected_attraction_1)
