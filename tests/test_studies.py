"""Études vérifiées sur artefacts synthétiques, sans entraînement de modèle."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from src.common.config import ConfigLoader
from src.evaluation.evaluator import ROW_KEYS
from src.evaluation.results import ResultsTable
from src.evaluation.size_transfer import SizeTransferStudy
from src.reports.communication import CommunicationComparison


def _row(controller, seed, episode_seed, size, distance):
    return {"controller": controller, "training_seed": seed, "episode_seed": episode_seed,
            "n_robots": size, "failure_fraction": 0.0, "shape": "geometries_circle", "shape_id": 0,
            "final_distance": distance, "convergence_step": None, "success": distance <= 0.15,
            "collision_rate": 0.0, "normalized_regularity": 0.0, "final_components": 1, "step": 4}


@pytest.fixture
def comparison_source(tmp_path):
    loader = ConfigLoader()
    source = tmp_path / "source"
    source.mkdir()
    controllers = {"no_comm": {"type": "trained", "variant": "no_comm"},
                   "gnn_1_round": {"type": "trained", "variant": "gnn_1_round"},
                   "manual": {"type": "baseline"}}
    training = loader.load_raw("training")
    model = asdict(loader.load_model())
    simulator = asdict(loader.load_simulator())
    runs = {name: {seed: {"config": {"training": training, "profile": "standard",
                                   "model": model, "simulator": simulator,
                                   "override": {"communication": name != "no_comm"}}}
                   for seed in (11, 12)} for name in ("no_comm", "gnn_1_round")}
    snapshot = {"evaluation": {"controllers": controllers}, "training_runs": runs,
                "metrics": asdict(loader.load_metrics())}
    (source / "effective_config.yaml").write_text(yaml.safe_dump(snapshot), encoding="utf-8")
    rows = [_row(name, seed, episode_seed, size, distance)
            for size in (20, 100) for name, distance in (("no_comm", 0.4), ("gnn_1_round", 0.2))
            for seed in (11, 12) for episode_seed in (1, 2)]
    rows += [_row("manual", None, episode_seed, size, 0.1) for size in (20, 100) for episode_seed in (1, 2)]
    with (source / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_KEYS)
        writer.writeheader()
        writer.writerows(rows)
    return source


class TestCommunicationComparison:
    def test_report_is_paired_and_conditioned(self, tmp_path, comparison_source):
        output = tmp_path / "comparison"
        summary = CommunicationComparison(comparison_source, output, "no_comm", ["gnn_1_round"]).run()
        assert len(summary["gains"]) == 2
        for gain in summary["gains"]:
            assert gain["mean_difference"] == pytest.approx(-0.2)
            assert gain["seed_count"] == 2
            assert gain["paired_count"] == 4
        assert (output / "information.csv").is_file()
        assert "nearest_displacement" in (output / "information.csv").read_text()
        assert (output / "conclusion.txt").is_file()

    def test_unequal_information_rejected(self, tmp_path, comparison_source):
        path = comparison_source / "effective_config.yaml"
        snapshot = yaml.safe_load(path.read_text())
        snapshot["training_runs"]["gnn_1_round"][11]["config"]["model"] = {
            **snapshot["training_runs"]["gnn_1_round"][11]["config"]["model"], "use_target_geometry": False}
        path.write_text(yaml.safe_dump(snapshot))
        with pytest.raises(ValueError, match="information"):
            CommunicationComparison(comparison_source, tmp_path / "comparison", "no_comm", ["gnn_1_round"]).run()

    def test_unmatched_episodes_rejected(self, tmp_path, comparison_source):
        path = comparison_source / "episodes.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows = [row for row in rows if not (row["controller"] == "gnn_1_round" and row["episode_seed"] == "2")]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ROW_KEYS)
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(ValueError, match="non appariés"):
            CommunicationComparison(comparison_source, tmp_path / "comparison", "no_comm", ["gnn_1_round"]).run()


class TestSizeTransfer:
    def test_cross_table_and_relative_degradation(self, tmp_path):
        config = {"output": str(tmp_path / "study"), "training_sizes": [20, 100]}
        study = SizeTransferStudy(config)
        tables = {size: ResultsTable([_row("gnn", seed, episode_seed, evaluation_size, distance)
                                    for seed in (11, 12) for episode_seed in (1, 2)
                                    for evaluation_size, distance in ((20, 0.1), (100, 0.4 if size == 20 else 0.2))])
                  for size in (20, 100)}
        summary = study.build_report(tables)
        assert len(summary["cross_size"]) == 4
        assert summary["degradation"][0]["relative_degradation"] == pytest.approx(1.0)
        assert (study.output / "cross_size.csv").is_file()

    def test_missing_checkpoint_does_not_train(self, tmp_path):
        config = {"output": str(tmp_path / "study"), "training_sizes": [20], "evaluation_sizes": [20, 40, 100],
                  "controller": "gnn_2_rounds", "checkpoint_roots": {20: str(tmp_path / "missing")},
                  "evaluation_config": "evaluation"}
        with pytest.raises(FileNotFoundError):
            SizeTransferStudy(config).run()
        assert not (tmp_path / "study").exists()
