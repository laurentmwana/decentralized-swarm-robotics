"""Curriculum vérifié par contrats et artefacts simulés, sans entraînement."""

from __future__ import annotations

import csv
import importlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from src.common.config import ConfigLoader
from src.shapes.index import ShapeIndex, build_split
from src.training.checkpoint import CheckpointManager
from src.training.curriculum import Curriculum, ForgettingMonitor
from src.training.sampling import CyclicTargetSampler


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("aucune optimisation réelle autorisée")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


@pytest.fixture
def curriculum(tmp_path):
    config = ConfigLoader().load_raw("curriculum")
    config["output"] = str(tmp_path / "curriculum")
    return Curriculum(config)


def test_levels_are_cumulative_and_never_use_test(curriculum):
    split = build_split(ConfigLoader().load_shapes())
    assert [len(level.training_targets) for level in curriculum.levels] == [5, 11, 31, 45]
    assert [len(level.validation_targets) for level in curriculum.levels] == [1, 3, 6, 9]
    previous = set()
    for level in curriculum.levels:
        assert previous <= set(level.training_targets)
        assert all(split.mapping[key.split("@")[0]] == "train" for key in level.training_targets)
        assert all(split.mapping[key.split("@")[0]] == "val" for key in level.validation_targets)
        previous = set(level.training_targets)


def test_plan_only_reports_cost_without_touching_outputs(curriculum):
    plan = curriculum.plan("gnn_3_rounds_geometry", 11, "standard")
    assert plan["duration_seconds"] is None
    assert plan["budget"]["n_robots"] == 50
    assert len(plan["levels"]) == 4
    batch = ConfigLoader().load_raw("training")["optimizer"]["batch_episodes"]
    assert plan["levels"][0]["optimization_rollout_steps"] == 1200 * batch * 120
    assert not curriculum.output.exists()


@pytest.mark.parametrize("change", ["repeated_type", "unknown_type", "empty_levels", "invalid_resolution", "invalid_sampling"])
def test_invalid_curriculum_rejected_before_outputs(curriculum, change):
    config = deepcopy(curriculum.config)
    if change == "repeated_type":
        config["levels"][1]["shape_types"] = ["geometries"]
    elif change == "unknown_type":
        config["levels"][0]["shape_types"] = ["unknown"]
    elif change == "empty_levels":
        config["levels"] = []
    elif change == "invalid_resolution":
        config["resolution"] = 100
    else:
        config["target_sampling"] = "random"
    with pytest.raises(ValueError):
        Curriculum(config)
    assert not curriculum.output.exists()


def test_cyclic_sampler_retains_balanced_targets():
    targets = [(torch.zeros(3, 2), shape_id) for shape_id in (4, 7, 8)]
    sampler = CyclicTargetSampler(targets)
    assert [sampler.next()[1] for _ in range(9)] == [4, 7, 8] * 3
    with pytest.raises(ValueError):
        CyclicTargetSampler([])
    with pytest.raises(ValueError):
        CyclicTargetSampler([targets[0], targets[0]])


def test_forgetting_is_paired_with_previous_level(tmp_path):
    monitor = ForgettingMonitor(0.05)
    monitor.record(1, [{"target": "circle", "total_loss": 0.1, "final_distance": 0.1}])
    monitor.record(2, [{"target": "circle", "total_loss": 0.2, "final_distance": 0.2},
                       {"target": "digit", "total_loss": 0.3, "final_distance": 0.3}])
    assert monitor.rows[0]["forgetting"] is None
    assert monitor.rows[1]["forgetting"] == pytest.approx(0.1)
    assert monitor.rows[1]["notable_forgetting"]
    assert monitor.rows[2]["forgetting"] is None
    monitor.write(tmp_path)
    with (tmp_path / "losses_by_target.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_levels_transfer_checkpoint_and_measure_every_seen_target(curriculum, monkeypatch):
    module = importlib.import_module("src.training.curriculum")
    calls = []

    def fake_train(variant, seed, profile, output, **kwargs):
        directory = output / variant / f"seed_{seed}"
        directory.mkdir(parents=True)
        (directory / "best.pt").write_bytes(b"synthetic checkpoint, never loaded")
        calls.append({"directory": directory, **kwargs})
        return {"seed": seed, "best_validation_score": 0.2}

    monkeypatch.setattr(module, "train_run", fake_train)
    monkeypatch.setattr(module.ShapeIndex, "load", Mock(return_value=ShapeIndex([], {})))
    measurements = []

    def fake_measure(self, directory, index, keys):
        measurements.append(tuple(keys))
        return [{"target": key, "total_loss": 0.1, "final_distance": 0.1} for key in keys]

    monkeypatch.setattr(ForgettingMonitor, "measure", fake_measure)
    result = curriculum.run("gnn_3_rounds_geometry", 11, "standard")
    assert result["status"] == "completed"
    assert calls[0]["initial_checkpoint"] is None
    for previous, current in zip(calls, calls[1:]):
        assert current["initial_checkpoint"] == previous["directory"] / "best.pt"
        assert set(previous["training_keys"]) <= set(current["training_keys"])
    assert measurements == [level.training_targets + level.validation_targets for level in curriculum.levels]
    assert len(result["levels"]) == 4
    with pytest.raises(FileExistsError):
        curriculum.run("gnn_3_rounds_geometry", 11, "standard")
    assert len(calls) == 4


def test_partial_curriculum_keeps_failed_status(curriculum, monkeypatch):
    module = importlib.import_module("src.training.curriculum")
    train = Mock(side_effect=RuntimeError("interruption simulée"))
    monkeypatch.setattr(module, "train_run", train)
    with pytest.raises(RuntimeError):
        curriculum.run("gnn_3_rounds_geometry", 11, "standard")
    directory = curriculum.output / "gnn_3_rounds_geometry" / "seed_11"
    status = json.loads((directory / "status.json").read_text())
    assert status["status"] == "failed"
    assert (directory / "effective_config.yaml").is_file()
    with pytest.raises(FileExistsError):
        curriculum.run("gnn_3_rounds_geometry", 11, "standard")
    train.assert_called_once()


def test_dry_run_cli_never_trains(curriculum, monkeypatch, capsys):
    module = importlib.import_module("src.training.curriculum")
    monkeypatch.setattr(module.Curriculum, "from_config", Mock(return_value=curriculum))
    train = Mock(side_effect=AssertionError("entraînement interdit"))
    monkeypatch.setattr(curriculum, "run", train)
    monkeypatch.setattr("sys.argv", ["curriculum", "--all", "--dry-run"])
    module.main()
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 10
    train.assert_not_called()
    assert not curriculum.output.exists()


def test_multitarget_entry_transfers_exact_weights_and_saves_provenance(tmp_path, monkeypatch):
    module = importlib.import_module("src.training.train")
    loader = ConfigLoader()
    config = loader.load_raw("training")
    keys = ["geometries_circle@50", "geometries_square@50", "geometries_pentagon@50"]
    clouds = {key: np.array([[0., 0.], [1., 0.], [0., 1.]]) for key in keys}
    index = ShapeIndex([], clouds, [key.split("@")[0] for key in keys])
    monkeypatch.setattr(module, "generate_index", Mock(return_value=(index, build_split(loader.load_shapes()))))
    model = module.build_model(replace(loader.load_model(), init_seed=11), config["variants"]["gnn_2_rounds"], loader.load_simulator().v_max)
    checkpoint = CheckpointManager(tmp_path / "previous").save(model, "best", {"seed": 11, "communication": True})
    expected = {key: value.clone() for key, value in model.state_dict().items()}

    def inspect(trainer, targets, validations):
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, expected[key])
        assert [shape_id for _, shape_id in targets] == [0, 1]
        assert len(validations) == 3
        return {"seed": 11, "best_validation_score": 0.2}

    monkeypatch.setattr(module.Trainer, "train_targets", inspect)
    module.run("gnn_2_rounds", 11, "standard", tmp_path / "next", training_keys=keys[:2],
               validation_keys=keys[2:], initial_checkpoint=checkpoint, curriculum_context={"level": 2})
    directory = tmp_path / "next" / "gnn_2_rounds" / "seed_11"
    saved = yaml.safe_load((directory / "effective_config.yaml").read_text())
    assert saved["curriculum"]["level"] == 2
    assert len(saved["initial_checkpoint"]["sha256"]) == 64
    targets = json.loads((directory / "targets.json").read_text())
    assert [record["key"] for record in targets["training_targets"]] == keys[:2]
    assert [record["split"] for record in targets["validation_targets"]] == ["train", "train", "val"]


def test_training_cli_plan_does_not_call_run(monkeypatch, capsys):
    module = importlib.import_module("src.training.train")
    train = Mock(side_effect=AssertionError("entraînement interdit"))
    monkeypatch.setattr(module, "run", train)
    monkeypatch.setattr("sys.argv", ["train", "--all-variants", "--all-seeds", "--dry-run"])
    module.main()
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 20
    train.assert_not_called()


def test_curriculum_producer_artifacts_are_accepted_by_report(curriculum, monkeypatch):
    from src.reports.curriculum import CurriculumReport

    module = importlib.import_module("src.training.train")
    curriculum.config["profiles"]["standard"]["seeds"] = [11, 12]
    curriculum.config["variants"] = {"gnn_3_rounds_geometry": curriculum.config["variants"]["gnn_3_rounds_geometry"]}
    keys = curriculum.levels[-1].training_targets + curriculum.levels[-1].validation_targets
    identifiers = [key.split("@")[0] for key in keys]
    clouds = {key: np.array([[0., 0.], [1., 0.], [0., 1.]]) for key in keys}
    index = ShapeIndex([], clouds, identifiers)
    monkeypatch.setattr(module, "generate_index", Mock(return_value=(index, build_split(curriculum.loader.load_shapes()))))
    monkeypatch.setattr(ShapeIndex, "load", Mock(return_value=index))

    def fake_train(trainer, targets, validations):
        metadata = {"seed": trainer.seed, "communication": trainer.communication}
        trainer.checkpoints.save(trainer.model, "best", metadata)
        trainer.checkpoints.save(trainer.model, "last", metadata)
        trainer.logger.loss(1, 0.1, {key: 0.1 for key in ("shape", "coverage", "collisions", "motion")})
        trainer.logger.validation(1, {"selection_score": 0.1})
        trainer.logger.write()
        return {"seed": trainer.seed, "iterations": trainer.profile["iterations"], "best_iteration": 1,
                "best_validation_score": 0.1, "final_loss": trainer.logger.last_loss, "duration_seconds": 0.0}

    def fake_measure(monitor, directory, shape_index, target_keys):
        saved = yaml.safe_load((directory / "effective_config.yaml").read_text())
        terms = {key: 0.1 for key in ("shape", "coverage", "collisions", "motion")}
        total = sum(saved["training"]["losses"]["weights"][key] * value for key, value in terms.items())
        return [{"target": key, "shape_id": shape_index.shape_id(key), "validation_seed_count": 3,
                 "total_loss": total, "final_distance": 0.1, **terms} for key in target_keys]

    monkeypatch.setattr(module.Trainer, "train_targets", fake_train)
    monkeypatch.setattr(ForgettingMonitor, "measure", fake_measure)
    for seed in (11, 12):
        curriculum.run("gnn_3_rounds_geometry", seed, "standard")
    summary = CurriculumReport(curriculum.output).run()
    assert summary["status"] == "completed"
    assert summary["retention_ok"]
    assert summary["row_count"] == 2 * sum(len(level.training_targets + level.validation_targets) for level in curriculum.levels)
    assert all(record["n"] == 2 for record in summary["distance_matrix"])
