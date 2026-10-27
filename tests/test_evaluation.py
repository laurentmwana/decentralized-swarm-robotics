"""Tests de la phase 6 — harnais d'évaluation.

Grille de scénarios complète et déterministe, mêmes positions initiales pour tous les
contrôleurs, agrégats exacts sur cas synthétique, comparaisons appariées et
reproductibilité de bout en bout (deux campagnes identiques -> fichiers identiques).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.base import Controller
from src.controllers.flocking import FlockingController
from src.controllers.manual_shape import ManualShapeController
from src.evaluation.campaign import expected_row_count, run
from src.evaluation.evaluator import Evaluator
from src.evaluation.results import PairedComparison, ResultsTable
from src.evaluation.scenario import ScenarioGrid
from src.simulator.episode import Episode


class _Stationary(Controller):
    def command(self, state, adjacency, mission):
        return np.zeros((state.n_robots, 2))


@pytest.fixture(scope="module")
def cfg_eval():
    return ConfigLoader("config").load_raw("evaluation")


class TestGrid:
    def test_scenario_count(self, cfg_eval):
        grid = ScenarioGrid.from_config(cfg_eval)
        expected = len(cfg_eval["swarm_sizes"]) * len(cfg_eval["failure_fractions"]) * len(cfg_eval["shapes"]) * cfg_eval["repetitions"]
        assert len(grid.scenarios()) == expected
        assert expected == 3 * 4 * 2 * 50
        # avec une graine par variante apprise : une serie de lignes par controleur logique
        assert expected_row_count(cfg_eval, trained_seed_count=1) == expected * len(cfg_eval["controllers"])

    def test_episode_seeds_disjoint_from_training_seeds(self, cfg_eval):
        grid = ScenarioGrid.from_config(cfg_eval)
        seeds = {s.episode_seed for s in grid.scenarios()}
        assert len(seeds) == cfg_eval["repetitions"]
        training_seeds = set(ConfigLoader("config").load_raw("training")["profiles"]["standard"]["seeds"])
        assert seeds.isdisjoint(training_seeds)
        assert min(seeds) == cfg_eval["evaluation_seed"] * 1000


class TestSharedInitialPositions:
    def test_same_seed_same_start_for_two_controllers(self, cfg_eval):
        loader = ConfigLoader("config")
        cfg_sim = loader.load_simulator()
        grid = ScenarioGrid.from_config(cfg_eval)
        cfg = grid.simulator_config(cfg_sim)
        target = np.stack([np.cos(np.linspace(0, 2 * np.pi, 20, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False))], axis=1)
        mission = Mission(shape_id=0, target=target)
        settings = loader.load_raw("baselines")["rules"]
        a = Episode(cfg).execute(mission, FlockingController(**settings["flocking"]), 20, 555000)
        b = Episode(cfg).execute(mission, ManualShapeController(**settings["manual"]), 20, 555000)
        assert np.array_equal(a.positions[0], b.positions[0])

    def test_failure_fraction_applied(self, cfg_eval):
        loader = ConfigLoader("config")
        cfg_sim = loader.load_simulator()
        grid = ScenarioGrid.from_config(cfg_eval)
        import dataclasses

        cfg = dataclasses.replace(grid.simulator_config(cfg_sim), failure_fraction=0.2)
        target = np.zeros((50, 2))
        target[:, 0] = np.linspace(-1, 1, 50)
        result = Episode(cfg).execute(Mission(shape_id=0, target=target), _Stationary(), 50, 555000)
        final_active_count = int(result.active[-1].sum())
        assert final_active_count == 40  # 20 % de 50


class TestAggregates:
    def _rows(self):
        rows = []
        for i, d in enumerate([0.1, 0.2, 0.3, 0.4]):
            rows.append(
                {"controller": "x", "training_seed": None, "episode_seed": i, "n_robots": 20,
                 "failure_fraction": 0.0, "shape": "f", "final_distance": d, "success": d <= 0.15,
                 "collision_rate": 0.0, "normalized_regularity": 0.0, "final_components": 1,
                 "convergence_step": None, "step": 60, "shape_id": 0}
            )
        return rows

    def test_mean_std_ci(self):
        table = ResultsTable(rows=self._rows())
        agg = table.aggregates("final_distance")[("x", None, 20, 0.0, "f")]
        assert agg["mean"] == pytest.approx(0.25)
        assert agg["std"] == pytest.approx(float(np.std([0.1, 0.2, 0.3, 0.4], ddof=1)))
        assert agg["ci95"] == pytest.approx(3.1824 * agg["std"] / 2.0)  # Student, 3 ddl
        assert agg["n"] == 4

    def test_baseline_vs_trained_pairing(self):
        rows = self._rows()
        for row in rows.copy():
            rows.append(dict(row, controller="y", training_seed=11, final_distance=0.0, success=True))
        comp = PairedComparison.build(rows, "y", "x")
        summary = comp.summary()["final_distance"]
        assert summary["paired_count"] == 4  # baseline sans graine : apparie avec tous les appris
        assert summary["mean_gain"] == pytest.approx(-0.25)  # y - x = -d en moyenne

    def test_two_trained_controllers_require_same_seed(self):
        rows = []
        for g in (11, 12):
            for i in range(1):
                rows.append({"controller": "a", "training_seed": g, "episode_seed": i, "n_robots": 20,
                               "failure_fraction": 0.0, "shape": "f", "final_distance": 0.1 * g, "success": False,
                               "collision_rate": 0.0, "normalized_regularity": 0.0, "final_components": 1,
                               "convergence_step": None, "step": 60, "shape_id": 0})
                rows.append({"controller": "b", "training_seed": g, "episode_seed": i, "n_robots": 20,
                               "failure_fraction": 0.0, "shape": "f", "final_distance": 0.05, "success": False,
                               "collision_rate": 0.0, "normalized_regularity": 0.0, "final_components": 1,
                               "convergence_step": None, "step": 60, "shape_id": 0})
        summary = PairedComparison.build(rows, "a", "b").summary()["final_distance"]
        assert summary["paired_count"] == 2  # pas de croisement 11x12
        assert summary["mean_gain"] == pytest.approx(1.1)  # (1.1-0.05 + 1.2-0.05)/2


class TestCampaignReproducibility:
    def test_two_runs_same_file(self, tmp_path, monkeypatch, cfg_eval):
        import json

        from src.shapes.index import ShapeIndex

        config = dict(cfg_eval, swarm_sizes=[4], failure_fractions=[0.0], shapes=["geometries_circle"], evaluation_horizon=4)
        target = np.stack([np.cos(np.linspace(0, 2 * np.pi, 50, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 50, endpoint=False))], axis=1)
        index = ShapeIndex([], {"geometries_circle": target}, ["geometries_circle"])
        original_load = ConfigLoader.load_raw
        monkeypatch.setattr(ConfigLoader, "load_raw", lambda self, name: config if name == "evaluation" else original_load(self, name))
        monkeypatch.setattr(ShapeIndex, "load", lambda directory: index)
        output_dir_a = tmp_path / "a"
        output_dir_b = tmp_path / "b"
        summary = run(controllers=["flocking", "manual"], repetitions=2, output=output_dir_a)
        run(controllers=["flocking", "manual"], repetitions=2, output=output_dir_b)
        content_a = (output_dir_a / "episodes.csv").read_text(encoding="utf-8")
        content_b = (output_dir_b / "episodes.csv").read_text(encoding="utf-8")
        assert content_a == content_b
        assert len(content_a.strip().splitlines()) == 5
        assert summary["rows"] == summary["expected_rows"] == 4
        for artifact in ("effective_config.yaml", "seeds.json", "version.txt", "summary.json", "aggregates.csv"):
            assert (output_dir_a / artifact).is_file()
        assert json.loads((output_dir_a / "summary.json").read_text())["status"] == "complete"
        with pytest.raises(FileExistsError):
            run(controllers=["flocking"], repetitions=2, output=output_dir_a)

    def test_variation_between_seeds_preserves_conditions(self):
        rows = TestAggregates()._rows()
        records = [dict(row, training_seed=seed, n_robots=size, final_distance=distance)
                   for row in rows for seed in (11, 12) for size, distance in ((20, 0.2), (100, 0.8))]
        variations = ResultsTable(records).variation_between_seeds()
        assert variations[("x", 20, 0.0, "f")]["mean"] == pytest.approx(0.2)
        assert variations[("x", 100, 0.0, "f")]["mean"] == pytest.approx(0.8)
        assert variations[("x", 20, 0.0, "f")]["n"] == 2

    def test_multiple_training_seeds_are_not_independent_episodes(self):
        rows = TestAggregates()._rows()
        records = rows + [dict(row, controller="y", training_seed=seed, final_distance=0.0)
                          for seed in (11, 12) for row in rows]
        comparison = PairedComparison.build(records, "y", "x")
        assert comparison.summary()["final_distance"]["paired_count"] == 8
        assert comparison.seed_summary()["final_distance"]["n"] == 2
        assert comparison.seed_summary()["final_distance"]["mean"] == pytest.approx(-0.25)

    def test_duplicate_pair_rejected(self):
        rows = TestAggregates()._rows()
        with pytest.raises(ValueError, match="même épisode"):
            PairedComparison.build(rows + [rows[0]], "x", "y")

    def test_partial_checkpoints_rejected(self, tmp_path, cfg_eval):
        from src.evaluation.factory import ControllerFactory

        directory = tmp_path / "gnn_2_rounds" / "seed_11"
        directory.mkdir(parents=True)
        (directory / "best.pt").write_bytes(b"partial")
        factory = ControllerFactory(evaluation_config=cfg_eval, training_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="run incomplet"):
            factory.versions("gnn_2_rounds")

    def test_partial_mode_keeps_strict_mode_blocking(self, tmp_path, cfg_eval):
        import json

        from src.evaluation.factory import ControllerFactory

        config = dict(cfg_eval, training_seeds=[11, 12])
        ready = tmp_path / "gnn_2_rounds" / "seed_11"
        ready.mkdir(parents=True)
        for artifact in ("best.pt", "effective_config.yaml", "summary.json", "targets.json"):
            (ready / artifact).write_bytes(b"x")
        (ready / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        factory = ControllerFactory(evaluation_config=config, training_dir=tmp_path)
        assert factory.inspect("gnn_2_rounds") == ([11], [("run incomplet", "variante gnn_2_rounds graine 12: run incomplet : " + str(tmp_path / "gnn_2_rounds" / "seed_12" / "best.pt"))])
        with pytest.raises(FileNotFoundError, match="run incomplet"):
            factory.versions("gnn_2_rounds")

    def test_partial_campaign_records_reasons_instead_of_omitting_them(self, tmp_path, cfg_eval):
        import json

        from src.evaluation.campaign import Campaign
        from src.evaluation.factory import ControllerFactory

        config = dict(cfg_eval, training_seeds=[11, 12])
        ready = tmp_path / "gnn_2_rounds" / "seed_11"
        ready.mkdir(parents=True)
        for artifact in ("best.pt", "effective_config.yaml", "summary.json", "targets.json"):
            (ready / artifact).write_bytes(b"x")
        (ready / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        campaign = Campaign.__new__(Campaign)
        campaign.factory = ControllerFactory(evaluation_config=config, training_dir=tmp_path)
        versions, skipped = campaign._resolve_versions(["gnn_2_rounds", "manual"], allow_partial=True)
        assert versions == {"gnn_2_rounds": [11], "manual": [None]}
        assert len(skipped) == 1 and "graine 12" in skipped[0]
        with pytest.raises(FileNotFoundError, match="run incomplet"):
            campaign._resolve_versions(["gnn_2_rounds"], allow_partial=False)

    def test_checkpoint_uses_saved_configuration_and_frozen_weights(self, tmp_path, monkeypatch, cfg_eval):
        import json
        from dataclasses import asdict, replace

        import yaml

        from src.evaluation.factory import ControllerFactory
        from src.models.gnn import SwarmGNN
        from src.training.checkpoint import CheckpointManager

        loader = ConfigLoader()
        config = dict(cfg_eval, training_seeds=[11])
        cfg_model = replace(loader.load_model(), init_seed=11)
        directory = tmp_path / "gnn_2_rounds" / "seed_11"
        directory.mkdir(parents=True)
        snapshot = {"model": asdict(cfg_model), "simulator": asdict(loader.load_simulator()),
                    "shapes": asdict(loader.load_shapes()), "communication": True}
        (directory / "effective_config.yaml").write_text(yaml.safe_dump(snapshot), encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps({"seed": 11}), encoding="utf-8")
        (directory / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        (directory / "targets.json").write_text("{}", encoding="utf-8")
        model = SwarmGNN(cfg_model, loader.load_simulator().v_max)
        CheckpointManager(directory).save(model, "best", {"seed": 11, "communication": True})
        monkeypatch.setattr(loader, "load_model", lambda: pytest.fail("configuration courante utilisée"))
        factory = ControllerFactory(loader, config, tmp_path)
        assert factory.versions("gnn_2_rounds") == [11]
        controller = factory.build("gnn_2_rounds", 11)
        assert controller.cfg == cfg_model
        assert all(not parameter.requires_grad for parameter in controller.model.parameters())
        (directory / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
        with pytest.raises(ValueError, match="non terminé"):
            factory.versions("gnn_2_rounds")


class TestStudentInterval:
    def test_t_critical_values(self):
        from src.evaluation.results import t_critical_95

        assert t_critical_95(4) == pytest.approx(2.7764, abs=1e-4)
        # Raccord continu entre la table (30 ddl) et le développement (au-delà).
        assert t_critical_95(31) == pytest.approx(2.0395, abs=2e-4)
        assert t_critical_95(49) == pytest.approx(2.0096, abs=2e-4)
        assert t_critical_95(10_000) == pytest.approx(1.96, abs=1e-3)

    def test_single_value_has_zero_interval(self):
        assert ResultsTable.statistics([0.3])["ci95"] == 0.0
