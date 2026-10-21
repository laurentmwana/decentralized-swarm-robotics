"""Tests obligatoires de la phase 1 — simulateur (PLAN, section Tests)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.base import Controller
from src.simulator import Arena, CommunicationGraph, Episode, SwarmState
from src.simulator.engine import Simulator

FLOAT_DTYPE = np.float64


class ConstantController(Controller):
    def __init__(self, value: tuple[float, float]) -> None:
        self.value = np.array(value, dtype=FLOAT_DTYPE)

    def command(self, state: SwarmState, adjacency: np.ndarray, mission: Mission) -> np.ndarray:
        return np.tile(self.value, (state.n_robots, 1))


@pytest.fixture(scope="module")
def cfg_sim():
    return ConfigLoader().load_simulator()


@pytest.fixture()
def mission():
    return Mission(shape_id=0, target=np.array([[1.0, 0.0], [0.0, 1.0]]))


class TestCommunicationGraph:
    def test_communication_range_respected(self, cfg_sim):
        graph = CommunicationGraph(cfg_sim.r_c)
        positions = np.array([[0.0, 0.0], [0.9 * cfg_sim.r_c, 0.0], [1.5 * cfg_sim.r_c, 0.0]])
        state = SwarmState(positions=positions, active=np.ones(3, dtype=bool), step=0)
        adj = graph.adjacency(state)
        assert adj[0, 1] and adj[1, 0]
        assert not adj[0, 2] and not adj[2, 0]
        assert adj[1, 2] and adj[2, 1]

    def test_no_self_loops(self, cfg_sim):
        graph = CommunicationGraph(cfg_sim.r_c)
        state = SwarmState(positions=np.zeros((4, 2)), active=np.ones(4, dtype=bool), step=0)
        adj = graph.adjacency(state)
        assert not adj.diagonal().any()

    def test_removed_robot_outside_graph(self, cfg_sim):
        graph = CommunicationGraph(cfg_sim.r_c)
        positions = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        state = SwarmState(positions=positions, active=np.array([True, True, False]), step=0)
        adj = graph.adjacency(state)
        assert adj[0, 1]
        assert not adj[2].any() and not adj[:, 2].any()

    def test_connected_components(self, cfg_sim):
        graph = CommunicationGraph(cfg_sim.r_c)
        positions = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        state = SwarmState(positions=positions, active=np.ones(3, dtype=bool), step=0)
        adj = graph.adjacency(state)
        assert graph.connected_components(adj, state.active) == 2


class TestEngine:
    def test_command_always_bounded(self, cfg_sim):
        engine = Simulator(Arena(cfg_sim.arena_size), cfg_sim)
        commands = engine.bound_commands(np.array([[1e6, 0.0], [0.0, -1e6], [3.0, 4.0]]), np.ones(3, dtype=bool))
        norms = np.linalg.norm(commands, axis=1)
        assert (norms <= cfg_sim.v_max + 1e-9).all()

    def test_removed_robot_receives_no_command(self, cfg_sim):
        engine = Simulator(Arena(cfg_sim.arena_size), cfg_sim)
        commands = engine.bound_commands(np.array([[1.0, 0.0], [1.0, 0.0]]), np.array([True, False]))
        assert (commands[1] == 0.0).all()

    def test_applied_velocity_bounded_during_episode(self, cfg_sim, mission):
        episode = Episode(cfg_sim)
        result = episode.execute(mission, ConstantController((1e6, 0.0)), 5, episode_seed=7)
        displacements = np.linalg.norm(np.diff(result.positions, axis=0), axis=2)
        active = result.active[:-1] & result.active[1:]
        assert (displacements[active] <= cfg_sim.v_max * cfg_sim.dt + 1e-9).all()
        inactive = ~result.active[1:]
        assert (displacements[inactive] == 0.0).all() if inactive.any() else True

    def test_positions_stay_in_arena(self, cfg_sim, mission):
        episode = Episode(cfg_sim)
        result = episode.execute(mission, ConstantController((1e6, 1e6)), 5, episode_seed=11)
        arena = Arena(cfg_sim.arena_size)
        for t in range(result.step + 1):
            assert arena.contains(result.positions[t]).all()


class TestEpisode:
    def test_same_seed_same_episode(self, cfg_sim, mission):
        episode = Episode(cfg_sim)
        a = episode.execute(mission, ConstantController((0.5, 0.2)), 12, episode_seed=42)
        b = episode.execute(mission, ConstantController((0.5, 0.2)), 12, episode_seed=42)
        assert np.array_equal(a.positions, b.positions)
        assert np.array_equal(a.commands, b.commands)

    def test_different_seeds_different_positions(self, cfg_sim, mission):
        episode = Episode(cfg_sim)
        a = episode.execute(mission, ConstantController((0.5, 0.2)), 12, episode_seed=42)
        b = episode.execute(mission, ConstantController((0.5, 0.2)), 12, episode_seed=43)
        assert not np.array_equal(a.positions[0], b.positions[0])

    def test_isolated_robot_handled_without_error(self, cfg_sim, mission):
        episode = Episode(cfg_sim)
        result = episode.execute(mission, ConstantController((0.1, 0.0)), 1, episode_seed=3)
        assert result.positions.shape == (cfg_sim.horizon + 1, 1, 2)
        assert np.isfinite(result.positions).all()

    def test_failed_robot_stationary_and_outside_graph(self, cfg_sim, mission):
        from dataclasses import replace

        cfg_failures = replace(cfg_sim, failure_fraction=0.5, horizon=10)
        episode = Episode(cfg_failures)
        result = episode.execute(mission, ConstantController((0.9, 0.0)), 8, episode_seed=5)
        failure_step = 5
        removed = np.flatnonzero(~result.active[failure_step])
        assert len(removed) == 4
        for i in removed:
            assert (result.positions[failure_step:, i] == result.positions[failure_step, i]).all()
        graph = CommunicationGraph(cfg_sim.r_c)
        state = SwarmState(positions=result.positions[failure_step], active=result.active[failure_step], step=failure_step)
        adj = graph.adjacency(state)
        assert not adj[removed].any() and not adj[:, removed].any()
