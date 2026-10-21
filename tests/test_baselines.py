"""Tests de la phase 3 — baselines non apprises.

Meme contrat d'interface que le futur GNN : commande(state, adjacence, mission) -> (N, 2),
borné par le simulateur, independant de la taille, decentralise. Plus la garde du
budget de reglage et le comportement attendu du regroupement (amas, pas de forme).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.flocking import FlockingController
from src.controllers.manual_shape import ManualShapeController
from src.simulator.episode import simulate_episode
from src.simulator.graph import CommunicationGraph
from src.simulator.state import SwarmState


def _circle(n: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _settings(cfg: dict) -> dict:
    return {
        "flocking": FlockingController(**cfg["rules"]["flocking"]),
        "manual": ManualShapeController(**cfg["rules"]["manual"]),
    }


@pytest.fixture(scope="module")
def cfg():
    loader = ConfigLoader("config")
    return loader.load_raw("baselines"), loader.load_simulator()


@pytest.fixture()
def settings(cfg):
    return _settings(cfg[0])


class TestTuningBudget:
    def test_grids_respect_budget(self, cfg):
        cfg_base, _ = cfg
        for name, grid in cfg_base["grids"].items():
            n = len(list(itertools.product(*grid.values())))
            assert n <= int(cfg_base["budget_trials"]), f"{name} : {n} essais > budget {cfg_base['budget_trials']}"

    def test_tuned_rules_in_grid(self, cfg):
        cfg_base, _ = cfg
        for name in cfg_base["grids"]:
            for key, value in cfg_base["rules"][name].items():
                assert value in cfg_base["grids"][name][key], f"{name}.{key} hors grille"


class TestCommonInterface:
    @pytest.mark.parametrize("swarm_size", [20, 40, 100])
    def test_command_shape_and_finite(self, settings, swarm_size):
        graph = CommunicationGraph(1.5)
        positions = _circle(swarm_size) * 1.4
        state = SwarmState(positions=positions, active=np.ones(swarm_size, bool), step=0)
        mission = Mission(shape_id=0, target=_circle(swarm_size))
        for controller in settings.values():
            cmd = controller.command(state, graph.adjacency(state), mission)
            assert cmd.shape == (swarm_size, 2)
            assert np.all(np.isfinite(cmd))

    def test_same_state_same_command(self, settings):
        state = SwarmState(positions=_circle(20) * 1.3, active=np.ones(20, bool), step=0)
        mission = Mission(shape_id=0, target=_circle(20))
        for controller in settings.values():
            a = controller.command(state, np.zeros((20, 20), bool), mission)
            b = controller.command(state, np.zeros((20, 20), bool), mission)
            assert np.array_equal(a, b)

    @pytest.mark.parametrize("rule", ["flocking", "manual"])
    def test_simulator_episode_commands_bounded(self, cfg, rule):
        _, cfg_sim = cfg
        controller = _settings(ConfigLoader("config").load_raw("baselines"))[rule]
        result = simulate_episode(_circle(20), controller, cfg_sim, n_robots=20, episode_seed=7)
        norms = np.linalg.norm(result.commands, axis=2)
        assert np.all(norms <= cfg_sim.v_max + 1e-9)
        displacements = np.linalg.norm(np.diff(result.positions, axis=0), axis=2)
        assert np.all(displacements <= cfg_sim.v_max * cfg_sim.dt + 1e-9)


class TestFlocking:
    def test_ignores_shape_mission(self, cfg):
        cfg_base, _ = cfg
        controller = FlockingController(**cfg_base["rules"]["flocking"])
        state = SwarmState(positions=_circle(20) * 1.3, active=np.ones(20, bool), step=0)
        adj = np.ones((20, 20), bool) & ~np.eye(20, dtype=bool)
        a = controller.command(state, adj, Mission(shape_id=0, target=_circle(20)))
        b = controller.command(state, adj, Mission(shape_id=1, target=-_circle(20)))
        assert np.array_equal(a, b)

    def test_isolated_robot_does_not_move(self, cfg):
        cfg_base, _ = cfg
        controller = FlockingController(**cfg_base["rules"]["flocking"])
        positions = np.array([[0.0, 0.0], [10.0, 0.0]])
        state = SwarmState(positions=positions, active=np.ones(2, bool), step=0)
        cmd = controller.command(state, np.zeros((2, 2), bool), Mission(shape_id=0, target=np.zeros((4, 2))))
        assert np.allclose(cmd, 0.0)

    def test_cluster_without_final_shape(self, cfg):
        cfg_base, cfg_sim = cfg
        controller = FlockingController(**cfg_base["rules"]["flocking"])
        target = _circle(40)
        result = simulate_episode(target, controller, cfg_sim, n_robots=40, episode_seed=7)
        final_positions = result.positions[-1][result.active[-1]]
        mean_radius = float(np.linalg.norm(final_positions, axis=1).mean())
        assert mean_radius < 0.9  # contracte vers le centre : amas, pas contour


class TestManualRule:
    def test_attracts_toward_nearest_contour_point(self, cfg):
        cfg_base, _ = cfg
        controller = ManualShapeController(**cfg_base["rules"]["manual"])
        target = np.array([[1.0, 0.0], [-1.0, 0.0]])
        positions = np.array([[0.8, 0.05]])
        state = SwarmState(positions=positions, active=np.ones(1, bool), step=0)
        cmd = controller.command(state, np.zeros((1, 1), bool), Mission(shape_id=0, target=target))[0]
        assert cmd[0] > 0  # vers (1,0) le plus proche
        assert cmd[1] < 0

    def test_forms_circle_at_all_sizes(self, cfg):
        cfg_base, cfg_sim = cfg
        controller = ManualShapeController(**cfg_base["rules"]["manual"])
        for n in (20, 40, 100):
            result = simulate_episode(_circle(n), controller, cfg_sim, n_robots=n, episode_seed=7)
            final_positions = result.positions[-1]
            distances = np.abs(np.linalg.norm(final_positions, axis=1) - 1.0)
            assert float(distances.mean()) < 0.15  # rayon 1 par construction
