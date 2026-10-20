"""Tests for 2D Swarm Simulator (Phase 1)."""

import numpy as np
import pytest

from src.swarm_gnn.config import SimulatorConfig
from src.swarm_gnn.simulator import SwarmSimulator


class TestSwarmSimulator:
    """Tests for SwarmSimulator core dynamics."""

    @pytest.fixture
    def config(self):
        return SimulatorConfig(
            seed=42,
            robot_count=5,
            arena_size=(10.0, 10.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=0.5,
            episode_steps=10,
            safety_distance=0.5,
        )

    @pytest.fixture
    def simulator(self, config):
        return SwarmSimulator(config)

    def test_initial_positions_in_arena(self, simulator):
        """All robots start inside arena bounds."""
        pos = simulator.positions
        assert pos.shape == (5, 2)
        assert np.all(pos >= 0)
        assert np.all(pos <= 10.0)

    def test_initial_velocities_zero(self, simulator):
        """Initial velocities are zero."""
        assert np.allclose(simulator.velocities, 0.0)

    def test_all_active_initially(self, simulator):
        """All robots active at start."""
        assert np.all(simulator.active)

    def test_step_count_zero_initially(self, simulator):
        """Step counter starts at zero."""
        assert simulator.step_count == 0

    def test_reset_restores_state(self, simulator):
        """Reset restores initial state with same seed."""
        # Move robots
        simulator.step(np.ones((5, 2)) * 0.5)
        assert simulator.step_count == 1
        # Reset
        simulator.reset()
        assert simulator.step_count == 0
        assert np.all(simulator.active)
        assert np.allclose(simulator.velocities, 0.0)
        # Positions should be same as initial (same seed)
        # Note: reset reinitializes RNG with same seed

    def test_step_updates_positions(self, simulator):
        """Step updates positions based on velocity commands."""
        initial_pos = simulator.positions.copy()
        commands = np.ones((5, 2)) * 0.5  # 0.5 m/s in x and y
        simulator.step(commands)
        expected = initial_pos + commands * simulator.config.time_step
        # Clip expected to arena (simulator does this)
        expected = np.clip(expected, (0.0, 0.0), simulator.config.arena_size)
        assert np.allclose(simulator.positions, expected)
        assert simulator.step_count == 1

    def test_step_clips_to_arena(self, simulator):
        """Positions clipped to arena boundaries."""
        # Put robot at edge
        simulator.positions[0] = np.array([9.9, 5.0])
        # Command pushing outside
        commands = np.zeros((5, 2))
        commands[0] = [1.0, 0.0]  # push right
        simulator.step(commands)
        assert simulator.positions[0, 0] <= 10.0

    def test_step_clips_speed(self, simulator):
        """Velocity commands clipped to max_speed."""
        commands = np.ones((5, 2)) * 2.0  # 2.0 > max_speed (1.0)
        simulator.step(commands)
        speeds = np.linalg.norm(simulator.velocities, axis=1)
        assert np.all(speeds <= simulator.config.max_speed + 1e-6)

    def test_inactive_robots_dont_move(self, simulator):
        """Inactive robots stay at position with zero velocity."""
        simulator.active[0] = False
        initial_pos = simulator.positions[0].copy()
        commands = np.ones((5, 2))
        simulator.step(commands)
        assert np.allclose(simulator.positions[0], initial_pos)
        assert np.allclose(simulator.velocities[0], 0.0)

    def test_neighbor_matrix_shape(self, simulator):
        """Neighbor matrix is (N, N) boolean."""
        neighbors = simulator.neighbor_matrix()
        assert neighbors.shape == (5, 5)
        assert neighbors.dtype == bool

    def test_neighbor_matrix_no_self(self, simulator):
        """No self-loops in neighbor matrix."""
        neighbors = simulator.neighbor_matrix()
        assert np.all(~np.diag(neighbors))

    def test_neighbor_matrix_symmetric(self, simulator):
        """Neighbor relation is symmetric."""
        neighbors = simulator.neighbor_matrix()
        assert np.array_equal(neighbors, neighbors.T)

    def test_neighbor_matrix_respects_radius(self, simulator):
        """Only robots within communication_radius are neighbors."""
        # Place robots at known positions
        simulator.positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],  # distance 1.0
            [5.0, 0.0],  # distance 5.0 > radius 3.0
            [0.0, 2.0],  # distance 2.0
            [0.0, 4.0],  # distance 4.0 > radius 3.0
        ])
        neighbors = simulator.neighbor_matrix()
        # Robot 0 neighbors: 1 (dist 1.0), 3 (dist 2.0) - within radius 3.0
        assert neighbors[0, 1] == True
        assert neighbors[0, 3] == True
        assert neighbors[0, 2] == False  # too far
        assert neighbors[0, 4] == False  # too far

    def test_neighbor_matrix_excludes_inactive(self, simulator):
        """Inactive robots don't appear as neighbors."""
        simulator.active[0] = False
        neighbors = simulator.neighbor_matrix()
        assert np.all(neighbors[0, :] == False)
        assert np.all(neighbors[:, 0] == False)

    def test_fail_marks_inactive(self, simulator):
        """fail() marks robots as inactive with zero velocity."""
        simulator.fail([0, 2])
        assert not simulator.active[0]
        assert not simulator.active[2]
        assert np.allclose(simulator.velocities[0], 0.0)
        assert np.allclose(simulator.velocities[2], 0.0)
        assert simulator.active[1]  # others unchanged

    def test_fail_invalid_id_raises(self, simulator):
        """fail() with invalid ID raises IndexError."""
        with pytest.raises(IndexError):
            simulator.fail([5])  # out of bounds
        with pytest.raises(IndexError):
            simulator.fail([-1])

    def test_collision_pairs_detects_close_robots(self):
        """collision_pairs returns pairs closer than safety_distance."""
        config = SimulatorConfig(
            seed=42,
            robot_count=4,
            arena_size=(10.0, 10.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=0.5,
            episode_steps=10,
            safety_distance=0.5,
        )
        simulator = SwarmSimulator(config)
        simulator.positions = np.array([
            [0.0, 0.0],
            [0.3, 0.0],   # distance 0.3 < safety 0.5
            [10.0, 10.0],
            [10.2, 10.0], # distance 0.2 < safety 0.5
        ])
        pairs = simulator.collision_pairs()
        assert (0, 1) in pairs or (1, 0) in pairs
        assert (2, 3) in pairs or (3, 2) in pairs

    def test_collision_pairs_excludes_inactive(self):
        """Inactive robots not counted in collisions."""
        config = SimulatorConfig(
            seed=42,
            robot_count=2,
            arena_size=(10.0, 10.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=0.5,
            episode_steps=10,
            safety_distance=0.5,
        )
        simulator = SwarmSimulator(config)
        simulator.positions = np.array([
            [0.0, 0.0],
            [0.3, 0.0],  # would be collision
        ])
        simulator.active[0] = False
        pairs = simulator.collision_pairs()
        assert len(pairs) == 0

    def test_observed_positions_no_noise(self, simulator):
        """observed_positions returns true positions when noise=0."""
        observed = simulator.observed_positions()
        assert np.allclose(observed, simulator.positions)

    def test_observed_positions_with_noise(self):
        """observed_positions adds Gaussian noise when std > 0."""
        config = SimulatorConfig(
            seed=42,
            robot_count=5,
            arena_size=(10.0, 10.0),
            communication_radius=3.0,
            max_speed=1.0,
            time_step=0.5,
            episode_steps=10,
            safety_distance=0.5,
            measurement_noise_std=0.1,
        )
        sim = SwarmSimulator(config)
        observed = sim.observed_positions()
        # Should differ from true positions (with very high probability)
        assert not np.allclose(observed, sim.positions, atol=1e-6)
        # But close (noise is small)
        assert np.allclose(observed, sim.positions, atol=0.5)

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical trajectories."""
        config = SimulatorConfig(
            seed=123,
            robot_count=3,
            arena_size=(10.0, 10.0),
            communication_radius=2.5,
            max_speed=1.0,
            time_step=0.3,
            episode_steps=10,
            safety_distance=0.3,
        )
        sim1 = SwarmSimulator(config)
        sim2 = SwarmSimulator(config)
        
        for _ in range(5):
            commands = np.ones((3, 2)) * 0.5
            sim1.step(commands)
            sim2.step(commands)
            assert np.allclose(sim1.positions, sim2.positions)
            assert np.allclose(sim1.velocities, sim2.velocities)

    def test_validate_commands_shape(self, simulator):
        """Step validates command shape."""
        with pytest.raises(ValueError):
            simulator.step(np.ones((3, 2)))  # wrong robot count
        with pytest.raises(ValueError):
            simulator.step(np.ones((5, 3)))  # wrong dimension

    def test_step_count_increments(self, simulator):
        """Step count increments each step."""
        assert simulator.step_count == 0
        simulator.step(np.zeros((5, 2)))
        assert simulator.step_count == 1
        simulator.step(np.zeros((5, 2)))
        assert simulator.step_count == 2