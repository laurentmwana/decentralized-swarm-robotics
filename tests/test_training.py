"""Tests obligatoires de la phase 5 — entraînement différentiable.

Pertes proches de zéro sur cas à réponse connue, absence d'auto-boucle dans le graphe
utilisé à l'entraînement, cohérence de la cible (points distincts), gradient qui circule
et profil smoke exécuté de bout en bout.
"""

from __future__ import annotations

import csv
import importlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest
import torch
import yaml

from src.common.config import ConfigLoader, ModelConfig
from src.models.gnn import SwarmGNN
from src.training import (
    CollisionLoss,
    CompositeLoss,
    CoverageLoss,
    DifferentiableSimulator,
    MotionLoss,
    Rollout,
    ShapeLoss,
)
from src.training.train import validate_target


def _rollout(positions: np.ndarray, commands: np.ndarray | None = None) -> Rollout:
    p = torch.tensor(positions, dtype=torch.float32)
    if positions.ndim == 2:
        p = p[None]
    c = torch.zeros((p.shape[0] - 1, p.shape[1], 2), dtype=torch.float32) if commands is None else torch.tensor(commands, dtype=torch.float32)
    return Rollout(positions=p, commands=c)


def _circle(n: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


@pytest.fixture(scope="module")
def cfgs():
    loader = ConfigLoader("config")
    return loader.load_simulator(), loader.load_metrics(), loader.load_shapes(), loader.load_model(), loader.load_raw("training")


class TestKnownLossCases:
    def test_shape_and_coverage_zero_on_target(self):
        target = _circle(16)
        rollout = _rollout(np.stack([target * 1.4, target]))
        t = torch.tensor(target, dtype=torch.float32)
        assert float(ShapeLoss(1.0).forward(rollout, t)) == pytest.approx(0.0, abs=1e-6)
        assert float(CoverageLoss(1.0).forward(rollout, t)) == pytest.approx(0.0, abs=1e-6)

    def test_exact_shape_loss_on_known_offset(self):
        target = np.array([[0.0, 0.0], [4.0, 0.0]])
        robots = target + np.array([0.25, 0.0])
        value = float(ShapeLoss(1.0).forward(_rollout(np.stack([robots, robots])), torch.tensor(target, dtype=torch.float32)))
        assert value == pytest.approx(0.25, abs=1e-5)

    def test_zero_collisions_when_separated_positive_when_overlapping(self):
        separated = np.array([[[0.0, 0.0], [1.0, 0.0]]])
        loss = CollisionLoss(0.1)
        assert float(loss.forward(_rollout(separated), torch.zeros(2, 2))) == pytest.approx(0.0)
        overlapping = np.array([[[0.0, 0.0], [0.05, 0.0]]])
        assert float(loss.forward(_rollout(overlapping), torch.zeros(2, 2))) > 0.0

    def test_zero_motion_for_zero_commands(self):
        positions = _circle(8)
        assert float(MotionLoss(1.0).forward(_rollout(np.stack([positions, positions])), torch.zeros(8, 2))) == pytest.approx(0.0)

    def test_composite_rejects_missing_weight(self, cfgs):
        _, _, shapes, _, cfg_train = cfgs
        with pytest.raises(ValueError):
            CompositeLoss([ShapeLoss(shapes.target_radius)], {"other": 1.0})


class TestTrainingGraph:
    def test_no_self_loops_false_diagonal(self, cfgs):
        cfg_sim, _, _, _, _ = cfgs
        simulator = DifferentiableSimulator(None, cfg_sim)
        positions = torch.tensor(_circle(10) * 0.5)
        assert not simulator.adjacency(positions).diagonal().any()

    def test_coincident_robots_do_not_form_edge(self, cfgs):
        # coherence avec le simulateur numpy : d > 0 exclut l'auto-boucle, y compris a distance nulle
        cfg_sim, _, _, _, _ = cfgs
        simulator = DifferentiableSimulator(None, cfg_sim)
        positions = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        adj = simulator.adjacency(positions)
        assert not adj.diagonal().any()
        assert not bool(adj[0, 1])


class TestTrainingTarget:
    def test_indexed_target_without_duplicates(self, cfgs, tmp_path):
        from src.shapes.index import generate_index

        _, _, shapes, _, cfg_train = cfgs
        index, _ = generate_index(shapes, tmp_path / "shapes")
        validate_target(index.point_cloud_by_key(cfg_train["training_target"]))

    def test_duplicate_rejected(self):
        target = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            validate_target(target)


class TestGradientAndSmoke:
    def test_gradient_flows_through_rollout(self, cfgs):
        cfg_sim, _, _, cfg_model, _ = cfgs
        torch.manual_seed(0)
        model = SwarmGNN(cfg_model, v_max=cfg_sim.v_max)
        simulator = DifferentiableSimulator(model, replace(cfg_sim, horizon=5))
        target = torch.tensor(_circle(20), dtype=torch.float32)
        positions = torch.tensor(_circle(20) * 1.6, dtype=torch.float32)
        rollout = simulator.rollout(positions, shape_id=0, target=target)
        total, _ = CompositeLoss([ShapeLoss(1.0)], {"shape": 1.0})(rollout, target)
        total.backward()
        assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())

    @pytest.mark.training
    def test_end_to_end_smoke(self, tmp_path, cfgs):
        from src.training.train import run

        summary = run("gnn_2_rounds", seed=1, profile="smoke", output=tmp_path)
        output_dir = tmp_path / "gnn_2_rounds" / "seed_1"
        for name in ("best.pt", "last.pt", "losses.csv", "validation.csv", "effective_config.yaml", "seeds.json", "version.txt", "summary.json"):
            assert (output_dir / name).exists(), name
        assert summary["iterations"] == 20
        rows = (output_dir / "losses.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 21  # en-tete + 20 iterations
        assert {"iteration", "total", "shape", "coverage", "collisions", "motion"} <= set(rows[0].split(","))


class SpyModel(torch.nn.Module):
    """Commande déterministe ; conserve les entrées sans entraîner de paramètres."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.calls = []

    def forward(self, velocity, positions, adjacency, shape_id, geometry=None):
        self.calls.append({
            "positions": positions.detach().clone(),
            "geometry": None if geometry is None else geometry.detach().clone(),
            "adjacency": adjacency.clone(),
            "shape_id": int(shape_id),
            "grad_enabled": torch.is_grad_enabled(),
        })
        command = torch.tensor([0.2, -0.1], dtype=positions.dtype).expand_as(positions)
        return command if geometry is None else command + geometry * 0.1


def _expected_geometry(positions, target):
    delta = target[None] - positions[:, None]
    nearest = delta.norm(dim=-1).argmin(dim=1)
    toward = delta[torch.arange(len(positions)), nearest]
    return toward / toward.norm(dim=1, keepdim=True).clamp_min(1e-9)


class TestRolloutGeometry:
    def test_geometry_recomputed_from_current_positions_each_step(self, cfgs):
        cfg_sim, _, _, cfg_model, _ = cfgs
        model = SpyModel(replace(cfg_model, use_target_geometry=True))
        simulator = DifferentiableSimulator(model, replace(cfg_sim, horizon=4))
        positions = torch.tensor([[0.2, 0.4], [-0.3, 0.1]])
        target = torch.tensor([[1.5, 1.8], [-1.4, 1.7]])
        result = simulator.rollout(positions, 3, target)
        assert len(model.calls) == 4
        for step, call in enumerate(model.calls):
            torch.testing.assert_close(call["positions"], result.positions[step])
            torch.testing.assert_close(call["geometry"], _expected_geometry(result.positions[step], target))
        assert not torch.allclose(model.calls[0]["geometry"], model.calls[-1]["geometry"])

    @pytest.mark.parametrize("use_geometry,target_present", [(False, True), (False, False), (True, False)])
    def test_no_geometry_computed_or_passed_when_disabled_or_absent(self, cfgs, monkeypatch, use_geometry, target_present):
        cfg_sim, _, _, cfg_model, _ = cfgs
        model = SpyModel(replace(cfg_model, use_target_geometry=use_geometry))
        simulator = DifferentiableSimulator(model, replace(cfg_sim, horizon=3))
        forbidden = Mock(side_effect=AssertionError("géométrie interdite"))
        monkeypatch.setattr(simulator, "target_vector", forbidden)
        target = torch.tensor([[1.0, 1.0]]) if target_present else None
        simulator.rollout(torch.tensor([[0.2, 0.1], [0.4, 0.3]]), 2, target)
        forbidden.assert_not_called()
        assert all(call["geometry"] is None for call in model.calls)

    def test_translation_consistency_without_arena_clipping(self, cfgs):
        cfg_sim, _, _, cfg_model, _ = cfgs
        positions = torch.tensor([[0.2, 0.4], [-0.3, 0.1]])
        target = torch.tensor([[1.5, 1.8], [-1.4, 1.7]])
        shift = torch.tensor([0.6, -0.4])
        models = [SpyModel(replace(cfg_model, use_target_geometry=True)) for _ in range(2)]
        simulators = [DifferentiableSimulator(model, replace(cfg_sim, horizon=4)) for model in models]
        original = simulators[0].rollout(positions, 3, target)
        translated = simulators[1].rollout(positions + shift, 3, target + shift)
        torch.testing.assert_close(original.commands, translated.commands)
        torch.testing.assert_close(original.positions + shift, translated.positions)
        for first, second in zip(models[0].calls, models[1].calls):
            torch.testing.assert_close(first["geometry"], second["geometry"])
            assert torch.equal(first["adjacency"], second["adjacency"])


class TestValidationSeeds:
    @pytest.mark.parametrize("training_seed,communication,initial_mode", [(11, True, True), (99, False, False)])
    def test_explicit_seeds_used_for_every_target_and_metrics_averaged(self, cfgs, training_seed, communication, initial_mode):
        from src.training.trainer import Trainer

        cfg_sim, _, _, cfg_model, cfg_training = cfgs
        profile = cfg_training["profiles"]["standard"]
        assert profile["validation_seeds"] == cfg_training["profiles"]["smoke"]["validation_seeds"]
        trainer = Trainer.__new__(Trainer)  # Aucun optimiseur ni entraînement dans ce test.
        trainer.model = SpyModel(cfg_model).train(initial_mode)
        trainer.cfg_sim = replace(cfg_sim, horizon=2)
        trainer.simulator = DifferentiableSimulator(trainer.model, trainer.cfg_sim)
        trainer.profile = profile
        trainer.seed = training_seed
        trainer.communication = communication
        distances = [1, 3, 5, 2, 4, 6]
        successes = [True, False, True, False, True, False]
        collisions = [0.1, 0.3, 0.5, 0.2, 0.4, 0.6]
        trainer.suite = Mock()
        trainer.suite.compute.side_effect = [
            Mock(values={"final_distance": distance, "success": success, "collision_rate": collision})
            for distance, success, collision in zip(distances, successes, collisions)
        ]
        targets = [(torch.tensor(_circle(8), dtype=torch.float32), 2),
                   (torch.tensor(_circle(10), dtype=torch.float32), 3)]
        score, values = trainer._validate(targets, n_robots=4)
        assert score == pytest.approx(3.5)
        assert values["validation_distance"] == pytest.approx(3.0)
        assert values["secondary_validation_distance"] == pytest.approx(4.0)
        assert values["validation_success"] == pytest.approx(2 / 3)
        assert values["validation_collision_rate"] == pytest.approx(0.3)
        assert values["selection_success_rate"] == pytest.approx(0.5)
        assert values["selection_collision_rate"] == pytest.approx(0.35)
        assert len(values["validation_targets"]) == 2
        assert [row["validation_seed"] for row in values["validation_episodes"]] == profile["validation_seeds"] * 2
        initial_calls = trainer.model.calls[::2]
        for target_index in range(2):
            for seed_index, seed in enumerate(profile["validation_seeds"]):
                expected = DifferentiableSimulator.initial_positions(4, cfg_sim.spawn_radius, torch.Generator().manual_seed(seed))
                call = initial_calls[target_index * 3 + seed_index]
                torch.testing.assert_close(call["positions"], expected, rtol=0, atol=0)
                assert call["shape_id"] == targets[target_index][1]
                assert not call["grad_enabled"]
                if not communication:
                    assert not call["adjacency"].any()
        assert trainer.model.training is initial_mode

    def test_model_mode_restored_after_metric_failure(self, cfgs):
        from src.training.trainer import Trainer

        cfg_sim, _, _, cfg_model, cfg_training = cfgs
        trainer = Trainer.__new__(Trainer)
        trainer.model = SpyModel(cfg_model).eval()
        trainer.cfg_sim = replace(cfg_sim, horizon=1)
        trainer.simulator = DifferentiableSimulator(trainer.model, trainer.cfg_sim)
        trainer.profile = cfg_training["profiles"]["smoke"]
        trainer.communication = True
        trainer.suite = Mock()
        trainer.suite.compute.side_effect = RuntimeError("métrique indisponible")
        with pytest.raises(RuntimeError, match="métrique"):
            trainer._validate([(torch.tensor(_circle(8), dtype=torch.float32), 0)], 4)
        assert not trainer.model.training

    @pytest.mark.parametrize("seeds", [[], [9001, 9001]])
    def test_invalid_validation_seed_lists_rejected(self, seeds):
        from src.training.trainer import validation_seeds

        with pytest.raises(ValueError):
            validation_seeds({"validation_seeds": seeds})


@pytest.fixture
def mocked_entry(cfgs, monkeypatch):
    """Index en mémoire et appel d'entraînement remplacé ; aucune sortie du dépôt."""
    from src.shapes.index import ShapeIndex, build_split

    entry = importlib.import_module("src.training.train")
    cfg_sim, cfg_metrics, cfg_shapes, cfg_model, cfg_training = cfgs
    training = deepcopy(cfg_training)
    keys = [training["training_target"], training["validation_target"]]
    index = ShapeIndex(rows=[], point_clouds={keys[0]: _circle(8), keys[1]: _circle(5)},
                       identifiers=[key.split("@")[0] for key in keys])
    loader = Mock()
    loader.load_raw.return_value = training
    loader.load_simulator.return_value = cfg_sim
    loader.load_metrics.return_value = cfg_metrics
    loader.load_shapes.return_value = cfg_shapes
    loader.load_model.return_value = cfg_model
    monkeypatch.setattr(entry, "ConfigLoader", Mock(return_value=loader))
    generated = Mock(return_value=(index, build_split(cfg_shapes)))
    monkeypatch.setattr(entry, "generate_index", generated)
    monkeypatch.setattr(entry, "_version", Mock(return_value="test-version"))
    train = create_autospec(entry.Trainer.train, return_value={"seed": 1, "iterations": 20, "best_validation_score": 0.2})
    monkeypatch.setattr(entry.Trainer, "train", train)

    def forbidden_step(*args, **kwargs):
        raise AssertionError("aucun pas d'optimisation autorisé")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_step)
    return entry, training, index, generated, train


class TestTrainingEntry:
    @pytest.mark.parametrize("variant", ["gnn_1_round", "no_comm"])
    def test_all_artifacts_written_before_mock_training(self, mocked_entry, tmp_path, variant):
        entry, training, index, generated, train = mocked_entry
        output_dir = tmp_path / variant / "seed_1"

        def inspect_artifacts(trainer, *args):
            for name in ("effective_config.yaml", "seeds.json", "version.txt", "targets.json", "status.json"):
                assert (output_dir / name).is_file(), name
            saved = yaml.safe_load((output_dir / "effective_config.yaml").read_text(encoding="utf-8"))
            assert saved["communication"] is (variant != "no_comm")
            assert saved["communication"] == trainer.communication
            assert saved["model"] == asdict(trainer.model.cfg)
            assert "communication" not in saved["model"]
            effective_model = ModelConfig(**saved["model"])
            assert effective_model.message_rounds == training["variants"][variant]["message_rounds"]
            assert effective_model.init_seed == 1
            assert saved["shapes"]["splits"]
            assert saved["simulator"]["horizon"] == training["profiles"]["smoke"]["horizon"]
            seeds = json.loads((output_dir / "seeds.json").read_text(encoding="utf-8"))
            assert seeds["seed"] == seeds["model_init_seed"] == 1
            assert seeds["training_seeds"] == [1]
            assert seeds["validation_seeds"] == training["profiles"]["smoke"]["validation_seeds"]
            assert (output_dir / "version.txt").read_text(encoding="utf-8").strip() == "test-version"
            targets = json.loads((output_dir / "targets.json").read_text(encoding="utf-8"))
            assert [record["split"] for record in targets["validation_targets"]] == ["train", "val"]
            np.testing.assert_array_equal(targets["training_target"]["points"], index.point_cloud_by_key(training["training_target"]))
            assert json.loads((output_dir / "status.json").read_text(encoding="utf-8"))["status"] == "running"
            assert not (output_dir / "summary.json").exists()
            return {"seed": 1, "iterations": 20, "best_validation_score": 0.2}

        train.side_effect = inspect_artifacts
        summary = entry.run(variant, 1, "smoke", tmp_path)
        train.assert_called_once()
        generated.assert_called_once()
        assert summary["parameter_count"] > 0
        assert json.loads((output_dir / "status.json").read_text(encoding="utf-8"))["status"] == "completed"

    def test_standard_artifact_includes_all_training_and_validation_seeds(self, mocked_entry, tmp_path):
        entry, training, _, _, train = mocked_entry
        train.return_value = {"seed": 11, "iterations": 1200, "best_validation_score": 0.2}
        entry.run("gnn_2_rounds", 11, "standard", tmp_path)
        seeds = json.loads((tmp_path / "gnn_2_rounds" / "seed_11" / "seeds.json").read_text(encoding="utf-8"))
        assert seeds["training_seeds"] == training["profiles"]["standard"]["seeds"]
        assert seeds["validation_seeds"] == training["profiles"]["standard"]["validation_seeds"]
        assert seeds["seed"] == seeds["model_init_seed"] == 11
        train.assert_called_once()

    @pytest.mark.parametrize("field,actual_split", [
        ("training_target", "val"), ("training_target", "test"), ("training_target", "missing"),
        ("validation_target", "train"), ("validation_target", "test"), ("validation_target", "missing"),
    ])
    def test_split_leakage_rejected_before_model_or_training(self, mocked_entry, tmp_path, monkeypatch, field, actual_split):
        entry, training, _, generated, train = mocked_entry
        split = generated.return_value[1]
        identifier = next((key for key, value in split.mapping.items() if value == actual_split), "unknown_shape")
        training[field] = identifier + "@50"
        build = Mock(side_effect=AssertionError("modèle construit avant contrôle du split"))
        monkeypatch.setattr(entry, "build_model", build)
        with pytest.raises(ValueError, match="split"):
            entry.run("gnn_2_rounds", 1, "smoke", tmp_path)
        build.assert_not_called()
        train.assert_not_called()
        assert not (tmp_path / "gnn_2_rounds" / "seed_1").exists()

    @pytest.mark.parametrize("status", ["completed", "failed"])
    def test_occupied_directory_never_overwritten(self, mocked_entry, tmp_path, status):
        entry, _, _, generated, train = mocked_entry
        output_dir = tmp_path / "gnn_2_rounds" / "seed_1"
        output_dir.mkdir(parents=True)
        sentinel = output_dir / "status.json"
        content = json.dumps({"status": status})
        sentinel.write_text(content, encoding="utf-8")
        with pytest.raises(FileExistsError):
            entry.run("gnn_2_rounds", 1, "smoke", tmp_path)
        assert sentinel.read_text(encoding="utf-8") == content
        assert list(output_dir.iterdir()) == [sentinel]
        generated.assert_not_called()
        train.assert_not_called()

    def test_failed_run_keeps_provenance_and_refuses_restart(self, mocked_entry, tmp_path):
        entry, _, _, _, train = mocked_entry
        train.side_effect = RuntimeError("interruption simulée")
        with pytest.raises(RuntimeError, match="interruption"):
            entry.run("gnn_2_rounds", 1, "smoke", tmp_path)
        output_dir = tmp_path / "gnn_2_rounds" / "seed_1"
        status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "failed"
        for name in ("effective_config.yaml", "seeds.json", "version.txt", "targets.json"):
            assert (output_dir / name).is_file()
        with pytest.raises(FileExistsError):
            entry.run("gnn_2_rounds", 1, "smoke", tmp_path)
        assert train.call_count == 1


class TestTrainingArtifacts:
    def test_checkpoint_config_matches_effective_model_without_optimization(self, mocked_entry, tmp_path):
        from src.training.checkpoint import CheckpointManager

        entry, _, _, _, _ = mocked_entry
        entry.run("no_comm", 1, "smoke", tmp_path)
        output_dir = tmp_path / "no_comm" / "seed_1"
        saved = yaml.safe_load((output_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        model = entry.build_model(ModelConfig(**saved["model"]), {}, saved["simulator"]["v_max"])
        manager = CheckpointManager(output_dir)
        manager.save(model, "best", {"communication": False, "model_config": {"outdated": True}})
        reloaded = entry.build_model(ModelConfig(**saved["model"]), {}, saved["simulator"]["v_max"])
        metadata = manager.load(reloaded, "best")
        assert metadata["model_config"] == saved["model"] == asdict(model.cfg)
        assert metadata["communication"] is False
        for first, second in zip(model.parameters(), reloaded.parameters()):
            torch.testing.assert_close(first, second)

    def test_validation_logger_writes_target_and_episode_rows(self, tmp_path):
        from src.training.logger import TrainingLogger

        logger = TrainingLogger(tmp_path)
        logger.loss(1, 0.2, {"shape": 0.1})
        logger.validation(1, {"selection_score": 0.2, "validation_distance": 0.1,
                              "validation_success": 2 / 3, "validation_collision_rate": 0.3,
                              "secondary_validation_distance": 0.3,
                              "validation_targets": [{"target_index": 0, "final_distance": 0.1}],
                              "validation_episodes": [{"target_index": 0, "validation_seed": 9001}]})
        logger.write()
        for name in ("validation.csv", "validation_targets.csv", "validation_episodes.csv"):
            with (tmp_path / name).open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 1
            assert rows[0]["iteration"] == "1"
        assert "validation_targets" not in (tmp_path / "validation.csv").read_text(encoding="utf-8")


@pytest.fixture
def complete_mock_run(mocked_entry, tmp_path):
    from src.training.checkpoint import CheckpointManager
    from src.training.logger import TrainingLogger

    entry, training, _, _, _ = mocked_entry
    entry.run("no_comm", 1, "smoke", tmp_path)
    output_dir = tmp_path / "no_comm" / "seed_1"
    saved = yaml.safe_load((output_dir / "effective_config.yaml").read_text(encoding="utf-8"))
    model = entry.build_model(ModelConfig(**saved["model"]), {}, saved["simulator"]["v_max"])
    CheckpointManager(output_dir).save(model, "best")
    logger = TrainingLogger(output_dir)
    logger.loss(1, 0.2, {"shape": 0.1})
    logger.validation(1, {"selection_score": 0.2, "validation_success": 0.5, "validation_collision_rate": 0.1})
    logger.write()
    return tmp_path, output_dir, training


class TestTrainingSummary:
    @pytest.mark.parametrize("missing", [
        "summary.json", "best.pt", "losses.csv", "validation.csv", "effective_config.yaml",
        "seeds.json", "version.txt", "targets.json", "status.json",
    ])
    def test_missing_seed_artifact_is_rejected_not_skipped(self, tmp_path, monkeypatch, missing):
        from src.reports import training_summary

        root = tmp_path / "no_comm" / "seed_1"
        root.mkdir(parents=True)
        for name in ("summary.json", "best.pt", "losses.csv", "validation.csv", "effective_config.yaml",
                     "seeds.json", "version.txt", "targets.json", "status.json"):
            if name != missing:
                (root / name).write_text("placeholder", encoding="utf-8")
        monkeypatch.setattr(training_summary, "INPUT", tmp_path)
        with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
            training_summary._load_runs("no_comm")

    def test_missing_configured_seed_is_rejected(self, complete_mock_run, monkeypatch):
        from src.reports import training_summary

        root, output_dir, _ = complete_mock_run
        saved = yaml.safe_load((output_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        saved["training"]["profiles"]["smoke"]["seeds"] = [1, 2]
        (output_dir / "effective_config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
        seeds = json.loads((output_dir / "seeds.json").read_text(encoding="utf-8"))
        seeds["training_seeds"] = [1, 2]
        (output_dir / "seeds.json").write_text(json.dumps(seeds), encoding="utf-8")
        monkeypatch.setattr(training_summary, "INPUT", root)
        with pytest.raises(FileNotFoundError, match="graines manquantes.*2"):
            training_summary._load_runs("no_comm")

    def test_incomplete_status_is_rejected_even_with_summary(self, complete_mock_run, monkeypatch):
        from src.reports import training_summary

        root, output_dir, _ = complete_mock_run
        (output_dir / "status.json").write_text(json.dumps({"status": "failed"}), encoding="utf-8")
        monkeypatch.setattr(training_summary, "INPUT", root)
        with pytest.raises(ValueError, match="incomplet"):
            training_summary._load_runs("no_comm")

    def test_missing_variant_rejected_before_report_writes(self, complete_mock_run, monkeypatch):
        from src.reports import training_summary

        root, _, _ = complete_mock_run
        figures = root / "figures"
        monkeypatch.setattr(training_summary, "INPUT", root)
        monkeypatch.setattr(training_summary, "FIGURES", figures)
        with pytest.raises(FileNotFoundError, match="variantes manquantes"):
            training_summary.run()
        assert not figures.exists()
        assert not (root / "summary.csv").exists()

    def test_representative_uses_saved_model_simulator_target_and_seeds(self, complete_mock_run, monkeypatch):
        from src.reports import training_summary

        root, output_dir, _ = complete_mock_run
        saved = yaml.safe_load((output_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        saved["model"]["message_rounds"] = 3
        saved["model"]["use_target_geometry"] = False
        saved["simulator"]["horizon"] = 7
        saved["simulator"]["dt"] = 0.17
        (output_dir / "effective_config.yaml").write_text(yaml.safe_dump(saved), encoding="utf-8")
        targets = json.loads((output_dir / "targets.json").read_text(encoding="utf-8"))
        targets["training_target"]["points"] = [[1.2, 0.4], [1.1, 0.5]]
        targets["training_target"]["shape_id"] = 9
        (output_dir / "targets.json").write_text(json.dumps(targets), encoding="utf-8")
        monkeypatch.setattr(training_summary, "INPUT", root)
        figures = root / "figures"
        figures.mkdir()
        monkeypatch.setattr(training_summary, "FIGURES", figures)
        loaded = Mock(return_value={})
        monkeypatch.setattr(training_summary.CheckpointManager, "load", loaded)
        simulation = Mock(return_value=Mock(positions=np.array([[[0.0, 0.0], [0.1, 0.1]]])))
        monkeypatch.setattr(training_summary, "simulate_episode", simulation)
        # Cette configuration courante volontairement différente ne doit pas reconstruire le modèle.
        training_summary._circle_figure({"variants": {"no_comm": {"message_rounds": 1, "communication": True}}})
        loaded.assert_called_once()
        simulation.assert_called_once()
        args, kwargs = simulation.call_args
        np.testing.assert_array_equal(args[0], targets["training_target"]["points"])
        assert asdict(args[1].model.cfg) == saved["model"]
        assert args[1].cfg == args[1].model.cfg
        assert args[1].communication is False
        assert asdict(args[2]) == saved["simulator"]
        assert kwargs == {"n_robots": saved["training"]["profiles"]["smoke"]["n_robots"],
                          "episode_seed": saved["training"]["profiles"]["smoke"]["validation_seeds"][0],
                          "shape_id": 9}
        assert (figures / "training_circle_final.png").is_file()
