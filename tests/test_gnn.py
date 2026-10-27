"""Tests obligatoires de la phase 4 — contrôleur GNN (PLAN, section Tests)."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.gnn import GNNController, NoCommGNNController
from src.models.gnn import SwarmGNN
from src.simulator.graph import CommunicationGraph
from src.simulator.state import SwarmState


@pytest.fixture(scope="module")
def cfg():
    return ConfigLoader().load_model()


@pytest.fixture(scope="module")
def sim():
    return ConfigLoader().load_simulator()


@pytest.fixture()
def controller(cfg, sim):
    return GNNController(cfg, sim.v_max)


def _state(positions):
    return SwarmState(positions=positions, active=np.ones(len(positions), dtype=bool), step=0)


def _pure_command(controller, positions, adjacency, mission):
    controller.reset()  # vitesse nulle -> commande fonction pure de (positions, graphe, consigne)
    return controller.command(_state(positions), adjacency, mission)


def _positions(n, seed=0):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 2))


def _adj(positions, r_c):
    return CommunicationGraph(r_c).adjacency(_state(positions))


@pytest.fixture()
def mission():
    return Mission(shape_id=3, target=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))


class TestInvariance:
    def test_permutation_invariance(self, controller, sim, mission):
        positions = _positions(30, seed=1)
        adj = _adj(positions, sim.r_c)
        perm = np.random.default_rng(7).permutation(30)
        base = _pure_command(controller, positions, adj, mission)
        permuted = _pure_command(controller, positions[perm], adj[np.ix_(perm, perm)], mission)
        assert np.allclose(base[perm], permuted, atol=1e-5)

    def test_translation_invariance(self, cfg, sim):
        from dataclasses import replace

        cfg_no_geometry = replace(cfg, use_target_geometry=False)
        controller = GNNController(cfg_no_geometry, sim.v_max)
        mission = Mission(shape_id=2, target=np.array([[1.0, 0.0], [0.0, 1.0]]))
        assert not controller.cfg.use_target_geometry, "variante sans géométrie : translation pure"
        positions = _positions(25, seed=2)
        adj = _adj(positions, sim.r_c)
        base = _pure_command(controller, positions, adj, mission)
        shifted = positions + np.array([3.7, -2.1])
        adj_shift = _adj(shifted, sim.r_c)
        assert np.allclose(adj, adj_shift)
        assert np.allclose(base, _pure_command(controller, shifted, adj_shift, mission), atol=1e-5)

    def test_geometry_equivariant_under_scene_translation(self, cfg, sim):
        from dataclasses import replace

        ctrl = GNNController(replace(cfg, use_target_geometry=True), sim.v_max, seed=5)
        positions = _positions(20, seed=3)
        target = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        adj = _adj(positions, sim.r_c)
        base = _pure_command(ctrl, positions, adj, Mission(shape_id=1, target=target))
        shift = np.array([2.0, 2.0])
        scene = _pure_command(ctrl, positions + shift, _adj(positions + shift, sim.r_c), Mission(shape_id=1, target=target + shift))
        assert np.allclose(base, scene, atol=1e-5)


class TestVariableSizes:
    def _parameter_count(self, cfg, sim):
        return SwarmGNN(cfg, sim.v_max).parameter_count()

    def test_same_model_for_20_40_100(self, controller, sim, mission):
        for n in (20, 40, 100):
            positions = _positions(n, seed=n)
            adj = _adj(positions, sim.r_c)
            cmd = _pure_command(controller, positions, adj, mission)
            assert cmd.shape == (n, 2)
            assert np.isfinite(cmd).all()

    def test_parameter_count_independent_of_n(self, cfg, sim, mission):
        controller = GNNController(cfg, sim.v_max)
        ref = controller.parameter_count()
        for n in (20, 40, 100):
            _pure_command(controller, _positions(n, seed=n), _adj(_positions(n, seed=n), sim.r_c), mission)
        assert controller.parameter_count() == ref


class TestEdgeCases:
    def test_isolated_robot_without_error(self, controller, sim, mission):
        positions = np.array([[0.0, 0.0]])
        adj = np.zeros((1, 1), dtype=bool)
        cmd = _pure_command(controller, positions, adj, mission)
        assert cmd.shape == (1, 2) and np.isfinite(cmd).all()

    def test_empty_neighborhood_zero_aggregate(self, cfg, sim):
        # Variante sans géométrie : deux robots isolés reçoivent une commande identique.
        from dataclasses import replace

        controller = GNNController(replace(cfg, use_target_geometry=False), sim.v_max)
        positions = np.array([[0.0, 0.0], [9.9, 9.9]])
        adj = np.zeros((2, 2), dtype=bool)
        mission = Mission(shape_id=4, target=np.array([[1.0, 0.0]]))
        cmd = _pure_command(controller, positions, adj, mission)
        assert np.allclose(cmd[0], cmd[1], atol=1e-6)

    def test_commands_always_bounded(self, controller, sim, mission):
        for n in (20, 40, 100):
            positions = _positions(n, seed=n + 10)
            cmd = _pure_command(controller, positions, _adj(positions, sim.r_c), mission)
            assert (np.linalg.norm(cmd, axis=1) <= sim.v_max + 1e-6).all()

    @pytest.mark.parametrize("v_max", [0.5, 2.0, 3.0])
    def test_decoder_scales_with_v_max(self, v_max):
        # Sorties saturées : la norme doit atteindre v_max, ni plafonner à √2 ni la dépasser.
        import torch

        from src.models.decoder import Decoder

        decoder = Decoder(hidden_dim=4, v_max=v_max)
        with torch.no_grad():
            decoder.mlp[-1].weight.zero_()
            decoder.mlp[-1].bias.copy_(torch.tensor([50.0, 0.0]))
            norms = torch.linalg.norm(decoder(torch.zeros(3, 4)), dim=-1)
        assert torch.allclose(norms, torch.full((3,), v_max), atol=1e-5)


class TestConditioning:
    def test_two_shapes_two_commands(self, controller, sim):
        positions = _positions(20, seed=8)
        adj = _adj(positions, sim.r_c)
        target = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = _pure_command(controller, positions, adj, Mission(shape_id=1, target=target))
        b = _pure_command(controller, positions, adj, Mission(shape_id=2, target=target))
        assert not np.allclose(a, b, atol=1e-4), "l'identifiant de forme doit conditionner la commande"


class TestNoCommunication:
    def test_independent_of_neighbors(self, cfg, sim, mission):
        ctrl = NoCommGNNController(cfg, sim.v_max)
        positions = _positions(20, seed=9)
        empty_adjacency = np.zeros((20, 20), dtype=bool)
        full_adjacency = np.ones((20, 20), dtype=bool)
        np.fill_diagonal(full_adjacency, False)
        a = _pure_command(ctrl, positions, empty_adjacency, mission)
        b = _pure_command(ctrl, positions, full_adjacency, mission)
        assert np.allclose(a, b, atol=1e-6), "sans communication, les voisins ne doivent rien changer"

    def test_size_comparable_to_gnn(self, cfg, sim):
        gnn = GNNController(cfg, sim.v_max)
        nocomm = NoCommGNNController(cfg, sim.v_max)
        assert gnn.parameter_count() == nocomm.parameter_count()


class TestAggregators:
    @pytest.mark.parametrize("name", ["mean", "normalized_sum", "attention"])
    def test_aggregation_variants_run(self, cfg, sim, name, mission):
        from dataclasses import replace

        cfg2 = replace(cfg, aggregator=name)
        ctrl = GNNController(cfg2, sim.v_max)
        positions = _positions(20, seed=11)
        cmd = _pure_command(ctrl, positions, _adj(positions, sim.r_c), mission)
        assert cmd.shape == (20, 2) and np.isfinite(cmd).all()
        assert (np.linalg.norm(cmd, axis=1) <= sim.v_max + 1e-6).all()
