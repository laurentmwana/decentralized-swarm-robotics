"""Tests synthétiques des phases 10–11 : ni simulateur réel, ni optimiseur."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.common.config import ConfigLoader
from src.evaluation import shape_studies as studies
from src.evaluation.evaluator import ROW_KEYS, Evaluator
from src.evaluation.results import ResultsTable
from src.evaluation.scenario import ScenarioGrid
from src.reports.readability import HumanReadabilityProtocol, ReadabilityProtocol
from src.shapes.index import ShapeIndex, build_split


def row(name="gnn", seed=11, episode=9001000, shape="letters_A@50", distance=0.1):
    return {"controller": name, "training_seed": seed, "episode_seed": episode,
            "n_robots": 50, "failure_fraction": 0.0, "shape": shape, "shape_id": 0,
            "final_distance": distance, "convergence_step": None, "success": distance <= 0.15,
            "collision_rate": 0.0, "normalized_regularity": 0.1, "final_components": 1, "step": 2}


@pytest.fixture
def study_env(tmp_path, monkeypatch):
    loader = ConfigLoader()
    config = loader.load_raw("complex-shapes")
    config["output"] = str(tmp_path / "study")
    config["comparison"]["output"] = str(tmp_path / "comparison")
    config["evaluation"].update(training_root=str(tmp_path / "weights"), training_seeds=[11, 12],
                                repetitions=2, positions_per_condition=2)
    config["examples"]["episode_seeds"] = [9001000, 9001001]
    config["comparison"]["controllers"]["gnn_3_rounds"]["training_root"] = str(tmp_path / "specialized")
    cfg_shapes = loader.load_shapes()
    split = build_split(cfg_shapes)
    identifiers = [f"{kind}_{name}" for kind in ("geometries", "digits", "letters", "numbers")
                   for name in getattr(cfg_shapes, kind)]
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    cloud = np.column_stack([np.cos(angles), np.sin(angles)])
    index = ShapeIndex([], {}, identifiers)
    for identifier in identifiers:
        kind, name = identifier.split("_", 1)
        index.rows.append({"name": name, "type": kind, "shape_id": identifiers.index(identifier),
                           "resolution": 50, "split": split.mapping[identifier], "n_points": 50})
        index.point_clouds[identifier + "@50"] = cloud.copy()
    monkeypatch.setattr(studies.ShapeIndex, "load", lambda directory: index)
    calls = []

    class FakeFactory:
        def __init__(self, loader, evaluation, training_dir):
            self.config = evaluation
            self.training_dir = Path(training_dir)

        def run_directory(self, name, seed):
            settings = self.config["controllers"][name]
            return Path(settings.get("training_root", self.training_dir)) / settings["variant"] / f"seed_{seed}"

        def versions(self, name):
            if self.config["controllers"][name]["type"] == "baseline":
                return [None]
            for seed in self.config["training_seeds"]:
                directory = self.run_directory(name, seed)
                for artifact in ("best.pt", "effective_config.yaml", "summary.json", "status.json", "targets.json"):
                    if not (directory / artifact).is_file():
                        raise FileNotFoundError(str(directory / artifact))
                if json.loads((directory / "status.json").read_text())["status"] != "completed":
                    raise ValueError("entraînement non terminé")
            return list(self.config["training_seeds"])

        def snapshot(self, name, seed):
            return yaml.safe_load((self.run_directory(name, seed) / "effective_config.yaml").read_text())

        def build(self, name, seed):
            calls.append(("build", name, seed))
            return object()

    monkeypatch.setattr(studies, "ControllerFactory", FakeFactory)
    all_settings = {**config["evaluation"]["controllers"], **config["comparison"]["controllers"]}
    snapshot_paths = {}
    for name, settings in all_settings.items():
        if settings["type"] != "trained":
            continue
        for seed in config["evaluation"]["training_seeds"]:
            directory = Path(settings.get("training_root", config["evaluation"]["training_root"])) / settings["variant"] / f"seed_{seed}"
            directory.mkdir(parents=True)
            snapshot = {"profile": "standard", "communication": True,
                        "model": {**asdict(loader.load_model()), "message_rounds": 3,
                                  "use_target_geometry": settings["expected_use_target_geometry"]},
                        "training": {"training_target": "geometries_circle@50", "validation_target": "geometries_pentagon@50",
                                     "profiles": {"standard": {"n_robots": settings.get("expected_n_robots", 50)}}},
                        "simulator": asdict(loader.load_simulator()), "shapes": asdict(cfg_shapes),
                        "curriculum": {"training_targets": [key + "@50" for key, part in split.mapping.items() if part == "train"],
                                       "validation_targets": [key + "@50" for key, part in split.mapping.items() if part == "val"]}}
            if name == "gnn_3_rounds":
                snapshot.pop("curriculum")
            path = directory / "effective_config.yaml"
            path.write_text(yaml.safe_dump(snapshot), encoding="utf-8")
            snapshot_paths[(name, seed)] = path
            (directory / "best.pt").write_bytes(f"synthetic-{name}-{seed}".encode())
            for artifact in ("summary.json", "status.json", "targets.json"):
                (directory / artifact).write_text(json.dumps({"status": "completed"}))

    def fake_evaluate(self, name, controller, seed, scenarios):
        calls.append(("evaluate", name, seed))
        rows = []
        for scenario in scenarios:
            record = row(name, seed, scenario.episode_seed, scenario.shape_key,
                         0.1 if scenario.episode_seed == 9001000 else 0.4)
            record["shape_id"] = index.shape_id(scenario.shape_key)
            path = studies.archive_path(self.output_dir, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, positions=np.stack([cloud, cloud]), active=np.ones((2, 50), dtype=bool),
                                target=cloud, target_radius=1.0)
            rows.append(record)
        return rows

    monkeypatch.setattr(Evaluator, "evaluate", fake_evaluate)
    monkeypatch.setattr(studies.ShapeStudyReport, "plot_examples", lambda *args: None)
    return config, index, snapshot_paths, calls


def edit_snapshot(path, change):
    snapshot = yaml.safe_load(path.read_text())
    change(snapshot)
    path.write_text(yaml.safe_dump(snapshot))


def test_complex_campaign_uses_existing_engine_and_frozen_metadata(study_env):
    config, index, paths, calls = study_env
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    summary = studies.ComplexShapeStudy(config).run()
    output = Path(config["output"])
    assert summary["status"] == "complete"
    assert summary["rows"] == len(summary["shape_split"]["train"] + summary["shape_split"]["val"]) * 2 * 5
    assert summary["information_equivalent"] is False
    assert summary["readability_status"] == "pending"
    assert not list(output.rglob("*.md"))
    assert all("@50" in key for keys in summary["shape_split"].values() for key in keys)
    assert all(before[path] == hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values())
    snapshot = yaml.safe_load((output / "effective_config.yaml").read_text())
    assert snapshot["evaluation"]["heldout_shapes"] == []
    assert snapshot["training_runs"]["gnn_3_rounds_id"][11]["checkpoint_sha256"]
    assert all(call[0] == "build" for call in calls[:5])
    assert (output / "failing_shapes.csv").is_file()
    examples = list(csv.DictReader((output / "examples.csv").open()))
    assert {example["example_kind"] for example in examples} == {"favorable", "unfavorable"}


def test_heldout_seen_and_test_are_separate(study_env):
    config, index, paths, calls = study_env
    summary = studies.HeldOutEvaluator(config).run()
    assert summary["mode"] == "heldout"
    assert {row["exposure"] for row in summary["seen_vs_heldout"]} == {"seen", "heldout"}
    assert "embedding ID" in (Path(config["output"]) / "limitations.txt").read_text()
    assert set(summary["shape_split"]["train"]).isdisjoint(summary["shape_split"]["test"])


@pytest.mark.parametrize("name,seed,section", [("gnn_3_rounds_id", 11, "training_targets"),
                                              ("gnn_3_rounds_geometry", 12, "validation_targets")])
@pytest.mark.parametrize("resolution", [20, 50])
def test_every_snapshot_and_resolution_checked_for_test_leakage(study_env, name, seed, section, resolution):
    config, index, paths, calls = study_env
    edit_snapshot(paths[(name, seed)], lambda snapshot: snapshot["curriculum"][section].append(f"letters_Y@{resolution}"))
    with pytest.raises(ValueError, match="fuite"):
        studies.HeldOutEvaluator(config).run()
    assert not Path(config["output"]).exists()
    assert not any(call[0] == "evaluate" for call in calls)


def test_seen_label_requires_actual_exposure(study_env):
    config, index, paths, calls = study_env
    edit_snapshot(paths[("gnn_3_rounds_id", 11)], lambda snapshot: snapshot["curriculum"]["training_targets"].remove("letters_A@50"))
    with pytest.raises(ValueError, match="annoncées vues"):
        studies.HeldOutEvaluator(config).run()
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("artifact", ["best.pt", "effective_config.yaml", "summary.json", "status.json", "targets.json"])
def test_missing_artifacts_refused_before_output(study_env, artifact):
    config, index, paths, calls = study_env
    (paths[("gnn_3_rounds_geometry", 12)].parent / artifact).unlink()
    with pytest.raises(FileNotFoundError):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()
    assert not calls


@pytest.mark.parametrize("field,value", [("hidden_dim", 32), ("message_rounds", 2), ("use_target_geometry", True)])
def test_wrong_architecture_refused(study_env, field, value):
    config, index, paths, calls = study_env
    edit_snapshot(paths[("gnn_3_rounds_id", 11)], lambda snapshot: snapshot["model"].update({field: value}))
    with pytest.raises(ValueError, match="architecture"):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()


def test_wrong_training_size_refused(study_env):
    config, index, paths, calls = study_env
    edit_snapshot(paths[("gnn_3_rounds_id", 11)], lambda snapshot: snapshot["training"]["profiles"]["standard"].update(n_robots=40))
    with pytest.raises(ValueError, match="taille"):
        studies.ComplexShapeStudy(config).run()


def test_unfinished_checkpoint_refused(study_env):
    config, index, paths, calls = study_env
    (paths[("gnn_3_rounds_id", 11)].parent / "status.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="non terminé"):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("mutation", ["ids", "split", "duplicates", "missing", "nonfinite"])
def test_shape_index_validation(study_env, mutation):
    config, index, paths, calls = study_env
    if mutation == "ids":
        index.identifiers.reverse()
    elif mutation == "split":
        index.rows[0]["split"] = "test"
    elif mutation == "duplicates":
        index.point_clouds["geometries_circle@50"][:] = 0
    elif mutation == "missing":
        index.rows.pop(0)
    else:
        index.point_clouds["geometries_circle@50"][0, 0] = np.nan
    with pytest.raises(ValueError):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()


def test_occupied_output_is_never_overwritten(study_env):
    config, index, paths, calls = study_env
    output = Path(config["output"])
    output.mkdir()
    (output / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        studies.ComplexShapeStudy(config).run()
    assert (output / "sentinel").read_text() == "keep"
    assert not calls


def test_specialized_root_and_same_episodes(study_env):
    config, index, paths, calls = study_env
    summary = studies.ComplexShapeStudy(config).run("compare")
    table = ResultsTable.read_csv(Path(config["comparison"]["output"]) / "episodes.csv")
    controllers = {record["controller"] for record in table.rows}
    assert "gnn_3_rounds" in controllers
    identities = {name: {(record["shape"], record["episode_seed"]) for record in table.rows if record["controller"] == name}
                  for name in controllers}
    assert all(values == identities["gnn_3_rounds"] for values in identities.values())
    assert summary["rows"] == 4 * 2 * 7
    assert not Path(config["output"]).exists()


def test_missing_root_contract_fails_closed(study_env, monkeypatch):
    config, index, paths, calls = study_env
    factory_class = studies.ControllerFactory
    monkeypatch.setattr(factory_class, "run_directory", lambda self, name, seed:
                        self.training_dir / self.config["controllers"][name]["variant"] / f"seed_{seed}")
    with pytest.raises(ValueError, match="training_root"):
        studies.ComplexShapeStudy(config).run("compare")
    assert not Path(config["comparison"]["output"]).exists()


def test_mutating_weights_detected_before_campaign(study_env, monkeypatch):
    config, index, paths, calls = study_env
    original = studies.ControllerFactory.build
    def mutate(self, name, seed):
        result = original(self, name, seed)
        if name == "gnn_3_rounds_id":
            (self.run_directory(name, seed) / "best.pt").write_bytes(b"changed")
        return result
    monkeypatch.setattr(studies.ControllerFactory, "build", mutate)
    with pytest.raises(ValueError, match="modifié"):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()


@pytest.mark.parametrize("field,value", [("positions_per_condition", 1), ("training_seeds", [9001000])])
def test_archive_or_seed_contract_refused(study_env, field, value):
    config, index, paths, calls = study_env
    config["evaluation"][field] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()


def test_grouped_statistics_use_seeds_not_pooled_episodes():
    rows = [row(seed=seed, episode=episode, distance=distance)
            for seed, count, distance in ((11, 2, 0.1), (12, 10, 0.5)) for episode in range(count)]
    within, between = studies.ShapeStudyReport.grouped(ResultsTable(rows), {"letters_A@50": "seen"})
    distance = next(record for record in between if record["metric"] == "final_distance")
    assert distance["n"] == 2
    assert distance["mean"] == pytest.approx(0.3)
    assert distance["std"] == pytest.approx(np.std([0.1, 0.5], ddof=1))
    assert distance["uncertainty_unit"] == "training_seed"
    assert len(within) == 10


@pytest.mark.parametrize("seed", [None, 11])
def test_shared_episode_seeds_across_shapes_are_not_independent_samples(seed):
    rows = [row(seed=seed, episode=episode, shape=shape, distance=distance)
            for episode, distances in ((1, (0.1, 0.3)), (2, (0.5, 0.7)))
            for shape, distance in zip(("letters_A@50", "letters_B@50"), distances)]
    within, between = studies.ShapeStudyReport.grouped(
        ResultsTable(rows), {"letters_A@50": "seen", "letters_B@50": "seen"})
    distance = next(record for record in within if record["metric"] == "final_distance")
    assert distance["shape_count"] == 2
    assert distance["n"] == 2
    assert distance["mean"] == pytest.approx(0.4)
    assert distance["std"] == pytest.approx(np.std([0.2, 0.6], ddof=1))
    aggregate = next(record for record in between if record["metric"] == "final_distance")
    assert aggregate["n"] == (2 if seed is None else 1)


def test_single_seed_ci_absent_and_baseline_episode_uncertainty():
    table = ResultsTable([row(), row("manual", None)])
    _, between = studies.ShapeStudyReport.grouped(table, {"letters_A@50": "seen"})
    assert all(record["ci95"] is None for record in between)
    assert {record["uncertainty_unit"] for record in between} == {"training_seed", "episode"}


def test_fixed_examples_both_extremes_and_incomplete_refused():
    config = {"training_seed": 11, "episode_seeds": [9001000, 9001001], "failure_fraction": 0.0}
    table = ResultsTable([row(distance=0.5), row(episode=9001001, distance=0.1)])
    examples = studies.ShapeStudyReport.fixed_examples(table, config)
    assert examples[0]["episode_seed"] == 9001001
    assert examples[1]["episode_seed"] == 9001000
    with pytest.raises(ValueError, match="incomplets"):
        studies.ShapeStudyReport.fixed_examples(ResultsTable([row()]), config)


@pytest.mark.parametrize("snapshot", [{}, {"curriculum": {}},
                                      {"curriculum": {"training_targets": "letters_A@50", "validation_targets": []}},
                                      {"curriculum": {"training_targets": ["letters_A@50"] * 2, "validation_targets": ["letters_V@50"]}}])
def test_missing_effective_lists_never_inferred_from_split(snapshot):
    with pytest.raises(ValueError, match="effectives"):
        studies.effective_targets(snapshot)


def test_specialized_effective_targets_contract():
    assert studies.effective_targets({"training": {"training_target": "geometries_circle@50", "validation_target": "geometries_pentagon@50"}}) == (
        ["geometries_circle@50"], ["geometries_pentagon@50"])


@pytest.fixture
def readability_env(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    config = ConfigLoader().load_raw("readability")
    config.update(source=str(source), output=str(tmp_path / "readability"), score_output=str(tmp_path / "scores"),
                  responses=None, controllers=["manual"], n_robots=2, episode_seeds=[9001000, 9001001],
                  bootstrap_repetitions=200, image_pixels=64, image_dpi=64)
    shapes = ["letters_Y@50", "numbers_314@50"]
    rows = []
    for shape in shapes:
        for episode in config["episode_seeds"]:
            record = row("manual", None, episode, shape)
            record["n_robots"] = 2
            path = studies.archive_path(source, record)
            path.parent.mkdir(parents=True, exist_ok=True)
            positions = np.array([[[0.0, 0.0], [0.5, 0.5]], [[0.1, 0.2], [0.4, 0.5]]])
            np.savez_compressed(path, positions=positions, active=np.ones((2, 2), dtype=bool))
            rows.append(record)
    studies.write_records(source / "episodes.csv", rows, ROW_KEYS)
    (source / "effective_config.yaml").write_text(yaml.safe_dump({"evaluation": {"shapes": shapes}}))
    (source / "summary.json").write_text('{"status":"complete"}')
    return config, rows


def test_protocol_is_abstract():
    with pytest.raises(TypeError):
        ReadabilityProtocol()


def test_prepare_blind_images_and_truth_separation(readability_env):
    config, rows = readability_env
    protocol = HumanReadabilityProtocol(config)
    summary = protocol.prepare()
    assert summary["status"] == "pending" and summary["accuracy"] is None
    assert summary["participant_count"] == 0 and summary["sample_count"] == 4
    public = (protocol.output / "questionnaire.html").read_text() + (protocol.output / "questionnaire.csv").read_text()
    assert "letters_Y" not in public and "numbers_314" not in public and "manual" not in public
    assert len(list((protocol.output / "images").glob("sample_*.png"))) == 4
    assert "letters_Y" in (protocol.output / "private" / "truth.csv").read_text()
    assert (protocol.output / "responses_template.csv").read_text().strip() == "participant_id,sample_id,predicted_shape"
    assert not list(protocol.output.rglob("*.md"))
    with pytest.raises(FileExistsError):
        protocol.prepare()


def test_sample_identity_is_deterministic_and_truth_free():
    positions, active = np.array([[0.1, 0.2]]), np.array([True])
    a = HumanReadabilityProtocol.sample_id(positions, active, 0, 123)
    assert a == HumanReadabilityProtocol.sample_id(positions.copy(), active.copy(), 0, 123)
    assert a != HumanReadabilityProtocol.sample_id(positions + 1, active, 0, 123)
    assert a != HumanReadabilityProtocol.sample_id(positions, active, 1, 123)


@pytest.mark.parametrize("mutation", ["missing", "nonfinite", "outside", "inactive", "incomplete"])
def test_invalid_readability_source_refused_before_write(readability_env, mutation):
    config, rows = readability_env
    path = studies.archive_path(Path(config["source"]), rows[0])
    if mutation == "missing":
        path.unlink()
    elif mutation == "incomplete":
        (Path(config["source"]) / "summary.json").write_text('{"status":"running"}')
    else:
        positions = np.zeros((2, 2, 2))
        active = np.ones((2, 2), dtype=bool)
        if mutation == "nonfinite":
            positions[0, 0, 0] = np.nan
        elif mutation == "outside":
            positions[-1, 0, 0] = 3.0
        else:
            active[-1] = False
        np.savez(path, positions=positions, active=active)
    with pytest.raises((ValueError, FileNotFoundError)):
        HumanReadabilityProtocol(config).prepare()
    assert not Path(config["output"]).exists()


def test_absent_responses_produces_pending_no_fabricated_score(readability_env):
    config, rows = readability_env
    protocol = HumanReadabilityProtocol(config)
    protocol.prepare()
    config["responses"] = str(protocol.output / "not_received.csv")
    summary = HumanReadabilityProtocol(config).score()
    assert summary["status"] == "pending"
    assert summary["accuracy"] is None and summary["confidence_interval"] is None
    assert summary["response_count"] == summary["participant_count"] == 0
    assert not (protocol.score_output / "responses.csv").exists()


def test_empty_responses_are_pending(readability_env):
    config, rows = readability_env
    summary, records = HumanReadabilityProtocol(config).analyze({"sample_1": "letters_Y"}, [])
    assert summary["accuracy"] is None and records == []


def test_accuracy_and_interval_are_between_participants(readability_env):
    config, rows = readability_env
    truth = {"sample_1": "letters_Y", "sample_2": "numbers_314"}
    responses = [{"participant_id": participant, "sample_id": sample, "predicted_shape": predicted}
                 for participant, predictions in (("p1", ["letters_Y", "numbers_314"]), ("p2", ["unknown", "unknown"]))
                 for sample, predicted in zip(truth, predictions)]
    protocol = HumanReadabilityProtocol(config)
    summary, records = protocol.analyze(truth, responses)
    assert summary["accuracy"] == 0.5 and summary["participant_count"] == 2
    assert summary["confidence_interval"] == [0.0, 1.0]
    assert summary["uncertainty_unit"] == "participant"
    assert [record["accuracy"] for record in records] == [1.0, 0.0]
    assert summary == protocol.analyze(truth, responses)[0]


def test_single_participant_has_observed_score_but_no_ci(readability_env):
    config, rows = readability_env
    summary, records = HumanReadabilityProtocol(config).analyze({"sample_1": "letters_Y"}, [
        {"participant_id": "p1", "sample_id": "sample_1", "predicted_shape": "letters_Y"}])
    assert summary["status"] == "observed" and summary["accuracy"] == 1.0
    assert summary["confidence_interval"] is None and summary["std_between_participants"] is None


@pytest.mark.parametrize("responses", [
    [{"participant_id": "", "sample_id": "s1", "predicted_shape": "letters_Y"}],
    [{"participant_id": "p1", "sample_id": "unknown", "predicted_shape": "letters_Y"}],
    [{"participant_id": "p1", "sample_id": "s1", "predicted_shape": "letters_Y"}] * 2,
    [{"participant_id": "p1", "sample_id": "s1", "predicted_shape": "letters_Y", "truth": "letters_Y"}],
])
def test_invalid_or_duplicate_responses_rejected(readability_env, responses):
    config, rows = readability_env
    with pytest.raises(ValueError):
        HumanReadabilityProtocol(config).analyze({"s1": "letters_Y"}, responses)


def test_incomplete_responses_cannot_select_easy_stimuli(readability_env):
    config, rows = readability_env
    with pytest.raises(ValueError, match="incomplètes"):
        HumanReadabilityProtocol(config).analyze({"s1": "letters_Y", "s2": "numbers_314"}, [
            {"participant_id": "p1", "sample_id": "s1", "predicted_shape": "letters_Y"}])


def test_raw_csv_preserved_and_score_output_protected(readability_env):
    config, rows = readability_env
    protocol = HumanReadabilityProtocol(config)
    protocol.prepare()
    truth = list(csv.DictReader((protocol.output / "private" / "truth.csv").open()))
    responses = [{"participant_id": "real_p1", "sample_id": record["sample_id"], "predicted_shape": record["true_shape"]}
                 for record in truth]
    path = protocol.output / "responses.csv"
    studies.write_records(path, responses)
    config["responses"] = str(path)
    protocol = HumanReadabilityProtocol(config)
    summary = protocol.score()
    assert summary["accuracy"] == 1.0 and summary["confidence_interval"] is None
    assert (protocol.score_output / "responses.csv").read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        protocol.score()


def test_csv_truth_columns_rejected_without_output(readability_env):
    config, rows = readability_env
    protocol = HumanReadabilityProtocol(config)
    protocol.prepare()
    path = protocol.output / "responses.csv"
    path.write_text("participant_id,sample_id,predicted_shape,truth\n")
    config["responses"] = str(path)
    with pytest.raises(ValueError, match="colonnes"):
        HumanReadabilityProtocol(config).score()
    assert not protocol.score_output.exists()


@pytest.mark.parametrize("name", ["complex-shapes", "shape-transfer"])
def test_formal_configs_are_explicit_and_seeds_disjoint(name):
    config = ConfigLoader().load_raw(name)
    evaluation = config["evaluation"]
    effective = {ScenarioGrid.from_config({**evaluation, "shapes": ["letters_A@50"]}).episode_seed(index)
                 for index in range(50)}
    assert evaluation["repetitions"] == evaluation["positions_per_condition"] == 50
    assert evaluation["swarm_sizes"] == [50] and evaluation["evaluation_horizon"] == 120
    assert effective.isdisjoint(evaluation["training_seeds"])
    assert effective.isdisjoint([9001, 9002, 9003])
    assert set(config["examples"]["episode_seeds"]) <= effective


def test_study_cli_dispatch(monkeypatch):
    import sys
    captured = []
    monkeypatch.setattr(sys, "argv", ["shape_studies", "--config", "shape-transfer", "--mode", "heldout"])
    monkeypatch.setattr(studies.HeldOutEvaluator, "run", lambda self, mode: captured.append(mode))
    studies.main()
    assert captured == ["heldout"]


def test_readability_cli_dispatch(monkeypatch):
    import sys
    from src.reports import readability
    captured = []
    monkeypatch.setattr(sys, "argv", ["readability", "--config", "readability", "--mode", "score"])
    monkeypatch.setattr(HumanReadabilityProtocol, "run", lambda self, mode: captured.append(mode))
    readability.main()
    assert captured == ["score"]


def test_example_plot_reads_archives_only(tmp_path):
    examples = [{**row(distance=0.1), "example_kind": "favorable"},
                {**row(episode=9001001, distance=0.5), "example_kind": "unfavorable"}]
    for record in examples:
        path = studies.archive_path(tmp_path, record)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = np.array([[0.0, 0.0], [0.5, 0.5]])
        np.savez_compressed(path, positions=np.stack([points, points]), active=np.ones((2, 2), dtype=bool), target=points)
    (tmp_path / "figures").mkdir()
    studies.ShapeStudyReport.plot_examples(tmp_path, examples, 2)
    assert (tmp_path / "figures" / "shape_examples_000.png").is_file()


def test_test_split_cannot_be_declared_seen(study_env):
    config, index, paths, calls = study_env
    config["target_selection"]["seen_splits"] = ["train", "test"]
    with pytest.raises(ValueError, match="test réservé"):
        studies.ComplexShapeStudy(config).run()
    assert not Path(config["output"]).exists()


def test_unseen_id_requires_embedding_vocabulary(study_env):
    config, index, paths, calls = study_env
    edit_snapshot(paths[("gnn_3_rounds_id", 11)], lambda snapshot: snapshot["model"].update(n_shapes=10))
    with pytest.raises(ValueError, match="architecture"):
        studies.HeldOutEvaluator(config).run()
    assert not Path(config["output"]).exists()


def test_partial_response_coverage_is_explicit_when_allowed(readability_env):
    config, rows = readability_env
    config["require_complete"] = False
    summary, records = HumanReadabilityProtocol(config).analyze({"s1": "letters_Y", "s2": "numbers_314"}, [
        {"participant_id": "p1", "sample_id": "s1", "predicted_shape": "letters_Y"}])
    assert records[0]["coverage"] == 0.5
    assert summary["response_count"] == 1


@pytest.mark.parametrize("key,value", [("confidence_level", 1.0), ("bootstrap_repetitions", 1)])
def test_invalid_confidence_parameters_refused(readability_env, key, value):
    config, rows = readability_env
    config[key] = value
    with pytest.raises(ValueError, match="intervalle"):
        HumanReadabilityProtocol(config).analyze({"s1": "letters_Y"}, [
            {"participant_id": "p1", "sample_id": "s1", "predicted_shape": "letters_Y"}])


def test_invalid_modes_never_start_campaign(study_env, readability_env):
    config, index, paths, calls = study_env
    with pytest.raises(ValueError, match="inconnu"):
        studies.ComplexShapeStudy(config).run("invalid")
    with pytest.raises(ValueError, match="uniquement"):
        studies.HeldOutEvaluator(config).run("complex")
    with pytest.raises(ValueError, match="inconnu"):
        HumanReadabilityProtocol(readability_env[0]).run("invalid")
    assert not calls


def test_study_cli_accepts_absolute_yaml_path(monkeypatch):
    import sys
    captured = []
    path = Path(__file__).resolve().parents[1] / "config" / "complex-shapes.yaml"
    monkeypatch.setattr(sys, "argv", ["shape_studies", "--config", str(path), "--mode", "compare"])
    monkeypatch.setattr(studies.ComplexShapeStudy, "run", lambda self, mode: captured.append((self.config["evaluation"]["repetitions"], mode)))
    studies.main()
    assert captured == [(50, "compare")]


def test_real_factory_root_contract_on_synthetic_artifacts(study_env):
    from src.evaluation.factory import ControllerFactory
    config, index, paths, calls = study_env
    evaluation = copy.deepcopy(config["evaluation"])
    evaluation["controllers"].update(config["comparison"]["controllers"])
    factory = ControllerFactory(ConfigLoader(), evaluation, Path(evaluation["training_root"]))
    for name in ("gnn_3_rounds_id", "gnn_3_rounds_geometry", "gnn_3_rounds"):
        assert factory.versions(name) == [11, 12]
        assert factory.run_directory(name, 11) == paths[(name, 11)].parent
        assert factory.snapshot(name, 11)["model"]["message_rounds"] == 3
