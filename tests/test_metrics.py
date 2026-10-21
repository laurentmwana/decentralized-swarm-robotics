"""Tests obligatoires de la phase 2 — métriques (PLAN, section Tests).

Valeurs exactes calculées à la main, cas limites de seuil, cohérence du critère de
réussite, pureté (recalcul depuis positions enregistrées) et comparabilité de taille.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import ConfigLoader
from src.controllers.trivial import TrivialController
from src.metrics import (
    ChamferDistance,
    CollisionRate,
    GraphConnectivity,
    MetricSuite,
    SpreadRegularity,
    SuccessCriterion,
    chamfer_step,
    convergence_step,
)
from src.simulator.episode import simulate_episode

FLOAT_DTYPE = np.float64


def circle(n: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


@pytest.fixture(scope="module")
def cfg():
    loader = ConfigLoader("config")
    return loader.load_simulator(), loader.load_metrics(), loader.load_shapes()


class TestExactChamferValues:
    def test_identical_sets_zero_distance(self):
        target = circle(16)
        assert chamfer_step(target.copy(), target) == pytest.approx(0.0)

    def test_known_offset_exact_distance(self):
        # cible [(0,0),(10,0)], robots decalés de +0.3 en x : aller = retour = 0.3
        target = np.array([[0.0, 0.0], [10.0, 0.0]])
        robots = target + np.array([0.3, 0.0])
        assert chamfer_step(robots, target) == pytest.approx(0.6)

    def test_normalization_by_target_radius(self):
        target = circle(16)
        positions = (target + np.array([0.3, 0.0]))[None]
        active = np.ones((1, 16), bool)
        raw = ChamferDistance(target_radius=1.0).compute(positions, active, target)["final_distance"]
        double = ChamferDistance(target_radius=2.0).compute(positions, active, target)["final_distance"]
        assert double == pytest.approx(float(raw) / 2.0)

    def test_empty_swarm_nan(self):
        assert np.isnan(chamfer_step(np.empty((0, 2)), circle(4)))


class TestCollisions:
    def test_colliding_pair_detected(self):
        # 2 robots superposes pendant 2 pas : 1 paire, 2 paires-pas -> taux = 1.0
        positions = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.1, 0.0], [0.1, 0.0]], [[0.2, 0.0], [0.2, 0.0]]])
        active = np.ones((3, 2), bool)
        r = CollisionRate(collision_radius=0.1).compute(positions, active, circle(2))
        assert r["collision_rate"] == pytest.approx(1.0)

    def test_distance_above_radius_no_collision(self):
        positions = np.array([[[0.0, 0.0], [0.5, 0.0]], [[0.0, 0.0], [0.5, 0.0]]])
        r = CollisionRate(collision_radius=0.1).compute(positions, np.ones((2, 2), bool), circle(2))
        assert r["collision_rate"] == pytest.approx(0.0)

    def test_removed_robot_excluded_from_denominator(self):
        # 3 robots, le 3e inactif partout : seules les paires actives comptent
        positions = np.zeros((2, 3, 2))
        active = np.array([[True, True, False], [True, True, False]])
        r = CollisionRate(collision_radius=0.1).compute(positions, active, circle(3))
        assert r["collision_rate"] == pytest.approx(1.0)  # 1 paire active, en collision aux 2 pas


class TestSuccessThreshold:
    def test_first_stable_convergence_crossing(self):
        curve = np.array([0.9, 0.4, 0.1, 0.2, 0.1, 0.1])
        # instable en t=2 (remontee en t=3) : premier k stable = 4
        assert convergence_step(curve, 0.15) == 4

    def test_inclusive_threshold_rejects_strict_excess(self):
        assert convergence_step(np.array([0.5, 0.15]), 0.15) == 1  # <= seuil compte
        assert convergence_step(np.array([0.5, 0.16]), 0.15) is None

    def test_never_below_threshold_no_convergence(self):
        assert convergence_step(np.array([0.9, 0.5, 0.3]), 0.15) is None

    def test_initially_stable_converges_at_zero(self):
        assert convergence_step(np.array([0.1, 0.1, 0.1]), 0.15) == 0


class TestSuccessConsistency:
    def test_initial_distance_above_threshold_not_valid_at_zero(self, cfg):
        _, metrics, shapes = cfg
        target = circle(20)
        positions = np.stack([target * 0.0 + np.array([1.5, 0.0]) for _ in range(3)])  # immobile, loin
        active = np.ones((3, 20), bool)
        r = SuccessCriterion(metrics.success_threshold, shapes.target_radius).compute(positions, active, target)
        assert r["success"] is False
        assert r["convergence_step"] is None

    def test_convergence_consistent_forward_backward(self, cfg):
        _, metrics, shapes = cfg
        threshold = metrics.success_threshold
        target = circle(20)
        # approche lineaire puis micro-remontee a mi-parcours : franchissement non monotone
        trajectory = []
        for t in range(30):
            factor = 0.2 + 0.03 * t
            if 12 <= t <= 14:
                factor = 1.0  # franchissement instantane puis rupture : non stable
            trajectory.append(target * factor)
        positions = np.array(trajectory)
        active = np.ones((30, 20), bool)
        r = SuccessCriterion(threshold, shapes.target_radius).compute(positions, active, target)
        curve = ChamferDistance(shapes.target_radius).compute(positions, active, target)["distance_curve"]
        k = r["convergence_step"]
        assert r["success"] == (k is not None)
        if k is not None:
            assert np.all(curve[k:] <= threshold)
            assert curve[k - 1] > threshold or k == 0
        # la remontee en t=12..14 interdit toute convergence <= 14
        assert k is None or k >= 15


class TestRecomputationPurity:
    def test_recomputation_from_saved_positions_identical(self, cfg, tmp_path):
        sim, metrics, shapes = cfg
        target = circle(20)
        controller = TrivialController(attraction_gain=1.0, repulsion_gain=2.0, repulsion_radius=0.3)
        result = simulate_episode(target, controller, sim, n_robots=20, episode_seed=142)

        suite = MetricSuite(metrics, sim, shapes.target_radius)
        first = suite.compute(result.positions, result.active, target)

        np.save(tmp_path / "positions.npy", result.positions)
        np.save(tmp_path / "active.npy", result.active)
        recomputed = suite.compute(np.load(tmp_path / "positions.npy"), np.load(tmp_path / "active.npy"), target)

        for key in first.values:
            a, b = first.values[key], recomputed.values[key]
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), key
            else:
                assert a == b, key


class TestSizeComparability:
    @pytest.mark.parametrize("n_robots", [20, 40, 100])
    def test_normalized_distance_scale_invariance(self, n_robots):
        # scene multiplier par un facteur (robots + cible) -> chamfer normalisee invariante
        target = circle(n_robots)
        for factor in (1.0, 2.5):
            positions = (target + np.array([0.2, 0.0]))[None] * factor
            active = np.ones((1, n_robots), bool)
            distance = ChamferDistance(target_radius=factor).compute(positions, active, target * factor)["final_distance"]
            reference = ChamferDistance(target_radius=1.0).compute((target + np.array([0.2, 0.0]))[None], np.ones((1, n_robots), bool), target)["final_distance"]
            assert distance == pytest.approx(reference)

    @pytest.mark.parametrize("n_robots", [20, 40, 100])
    def test_normalized_regularity_comparable_across_sizes(self, n_robots):
        r = SpreadRegularity("auto").compute(circle(n_robots)[None], np.ones((1, n_robots), bool), circle(n_robots))
        assert r["regularity_value"] == pytest.approx(0.0)
        assert r["normalized_regularity"] == pytest.approx(0.0)


class TestConnectivity:
    def test_single_connected_cluster(self, cfg):
        sim, _, _ = cfg
        positions = np.array([circle(10) * 0.3 for _ in range(3)])
        r = GraphConnectivity(sim.r_c).compute(positions, np.ones((3, 10), bool), circle(10))
        assert r["final_components"] == 1
        assert r["initially_connected"] is True

    def test_two_separated_groups_two_components(self, cfg):
        sim, _, _ = cfg
        group_a = np.array([[-2.0, 0.0], [-2.3, 0.0]])
        group_b = np.array([[2.0, 0.0], [2.3, 0.0]])
        positions = np.stack([np.vstack([group_a, group_b])])
        r = GraphConnectivity(sim.r_c).compute(positions, np.ones((1, 4), bool), circle(4))
        assert r["final_components"] == 2

    def test_removed_robot_not_counted(self, cfg):
        sim, _, _ = cfg
        positions = np.array([[[-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]]])
        active = np.array([[True, True, False]])
        r = GraphConnectivity(sim.r_c).compute(positions, active, circle(3))
        assert r["final_components"] == 2  # le retire ne forme pas une composante


class TestConfigurationMetrics:
    def test_metrics_yaml_loads_strictly(self, cfg):
        _, metrics, _ = cfg
        assert metrics.success_threshold == pytest.approx(0.15)
        assert metrics.collision_radius == pytest.approx(0.1)
        assert metrics.regularity_reference == "auto"

    def test_negative_threshold_rejected(self):
        from src.common.config import MetricsConfig

        with pytest.raises(ValueError):
            MetricsConfig(success_threshold=0.0, collision_radius=0.1, regularity_reference="auto")
