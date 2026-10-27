"""Rapport vérifié sur des fichiers synthétiques, sans optimisation ni tenseurs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.common.config import ConfigLoader
from src.evaluation.results import ResultsTable
from src.reports import curriculum as report
from src.reports.curriculum import CurriculumReport, ROW_KEYS


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=ROW_KEYS)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _edit_json(path: Path, **changes) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    _json(path, {**value, **changes})


def _source_bytes(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*")
            if path.is_file() and path.parent != root}


def _build_campaign(tmp_path, profile="standard"):
    root = tmp_path / "curriculum"
    kinds = ["geometries", "digits", "letters", "numbers"]
    variants = {"gnn_3_rounds_id": {"message_rounds": 3, "use_target_geometry": False},
                "gnn_3_rounds_geometry": {"message_rounds": 3, "use_target_geometry": True}}
    budget = {"iterations": 10, "n_robots": 3, "horizon": 4, "validation_interval": 5,
              "seeds": [11, 12, 13, 14, 15], "validation_seeds": [9001, 9002, 9003]}
    raw = {"training_config": "training", "output": str(root),
           "target_sampling": "cyclic", "transfer_checkpoint": "best", "forgetting_tolerance": 0.05,
           "variants": variants, "profiles": {"standard": budget, "smoke": {**budget, "seeds": [1]}},
           "levels": [{"name": kind, "shape_types": [kind]} for kind in kinds]}
    budget = raw["profiles"][profile]
    shapes = {"sampler_seed": 123, "splits": {kind: {"train": ["a"], "val": ["b"], "test": ["c"]}
                                            for kind in kinds}}
    weights = {"shape": 2.0, "coverage": 3.0, "collisions": 5.0, "motion": 7.0}
    ids = {f"{kind}_{name}": 2 * index + offset
           for index, kind in enumerate(kinds) for offset, name in enumerate(("a", "b"))}
    for variant, override in variants.items():
        for seed_index, seed in enumerate(budget["seeds"]):
            directory = root / variant / f"seed_{seed}"
            directory.mkdir(parents=True)
            _yaml(directory / "effective_config.yaml", raw)
            _json(directory / "seeds.json", {"root_seed": seed, "validation_seeds": budget["validation_seeds"]})
            (directory / "version.txt").write_text("synthetic-version\n", encoding="utf-8")
            identity = {"status": "completed", "variant": variant, "seed": seed, "profile": profile}
            _json(directory / "status.json", identity)
            rows, levels, previous = [], [], {}
            training, validation = [], []
            for number, kind in enumerate(kinds, start=1):
                training.append(f"{kind}_a")
                validation.append(f"{kind}_b")
                level_dir = root / f"level_{number}" / variant / f"seed_{seed}"
                level_dir.mkdir(parents=True)
                saved = {"variant": variant, "override": override, "profile": profile, "communication": True,
                         "model": {"init_seed": seed, "hidden_dim": 8, **override},
                         "training": {"variants": variants, "profiles": raw["profiles"], "losses": {"weights": weights},
                                      "training_target": training[0], "validation_target": validation[0]},
                         "shapes": shapes, "simulator": {"horizon": budget["horizon"]}, "metrics": {"collision_radius": 0.1},
                         "curriculum": {"config": raw, "level": number, "name": kind,
                                        "training_targets": training.copy(), "validation_targets": validation.copy(),
                                        "optimizer_state_transfer": False}}
                if number > 1:
                    best = root / f"level_{number - 1}" / variant / f"seed_{seed}" / "best.pt"
                    saved["initial_checkpoint"] = {"path": str(best), "sha256": hashlib.sha256(best.read_bytes()).hexdigest()}
                _yaml(level_dir / "effective_config.yaml", saved)
                _json(level_dir / "status.json", identity)
                _json(level_dir / "seeds.json", {"seed": seed, "model_init_seed": seed, "sampler_seed": 123,
                                                "training_seeds": budget["seeds"], "validation_seeds": budget["validation_seeds"]})
                (level_dir / "version.txt").write_text("synthetic-version\n", encoding="utf-8")
                # Ce ne sont volontairement pas des checkpoints torch valides.
                (level_dir / "best.pt").write_bytes(f"fake-best-{variant}-{seed}-{number}".encode())
                (level_dir / "last.pt").write_bytes(b"fake-last-different-from-best")
                (level_dir / "losses.csv").write_text("iteration,total\n10,999\n", encoding="utf-8")
                (level_dir / "validation.csv").write_text("iteration,selection_score\n5,777\n", encoding="utf-8")

                def target_record(target):
                    return {"key": target, "shape_id": ids[target], "split": "train" if target in training else "val"}

                _json(level_dir / "targets.json", {"training_target": target_record(training[0]),
                                                  "training_targets": [target_record(key) for key in training],
                                                  "validation_targets": [target_record(key) for key in training + validation]})
                level_summary = {"seed": seed, "iterations": 10, "best_iteration": 5,
                                 "best_validation_score": 777, "final_loss": {"total": 999}, "duration_seconds": 0.0}
                _json(level_dir / "summary.json", level_summary)
                levels.append({"level": number, **level_summary})
                for target in training + validation:
                    introduction = kinds.index(target.split("_")[0]) + 1
                    distance = (0.2 * (seed_index + 1) + ids[target] * 0.01
                                + (number - introduction) * (0.02 * (seed_index + 1))
                                - (0.05 if override["use_target_geometry"] else 0.0))
                    old = previous.get(target)
                    change = None if old is None else distance - old
                    terms = {"shape": 0.4, "coverage": 0.2, "collisions": 0.03, "motion": 0.05}
                    rows.append({"level": number, "target": target, "shape_id": ids[target],
                                 "validation_seed_count": len(budget["validation_seeds"]),
                                 "total_loss": sum(weights[key] * terms[key] for key in terms),
                                 "final_distance": distance, **terms, "previous_distance": old,
                                 "forgetting": change, "notable_forgetting": change is not None and change > 0.05})
                    previous[target] = distance
            _csv(directory / "losses_by_target.csv", rows)
            _json(directory / "summary.json", {**identity, "levels": levels, "forgetting_tolerance": 0.05,
                                              "notable_forgetting_count": sum(row["notable_forgetting"] for row in rows),
                                              "final_checkpoint": str(root / "level_4" / variant / f"seed_{seed}" / "best.pt")})
    return root


@pytest.fixture
def campaign(tmp_path):
    return _build_campaign(tmp_path)


def test_known_statistics_paired_forgetting_and_read_only_sources(campaign, monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("Le rapport ne doit charger aucun tenseur ou modèle.")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    calls = []
    original_statistics = ResultsTable.statistics

    def statistics(values):
        calls.append(len(values))
        return original_statistics(values)

    monkeypatch.setattr(ResultsTable, "statistics", staticmethod(statistics))
    before = _source_bytes(campaign)
    summary = CurriculumReport(campaign).run()
    assert _source_bytes(campaign) == before
    assert summary["row_count"] == 200
    assert summary["forgetting_tolerance"] == 0.05
    assert summary["notable_forgetting_count"] == summary["error_count"] == 72
    assert not summary["retention_ok"]
    assert summary["measurement_checkpoint"] == "best"
    assert summary["aggregation_unit"] == "training_seed"
    assert len(summary["distance_matrix"]) == 40
    assert len(summary["forgetting"]) == 24
    assert calls == [5] * 64  # Jamais 15 épisodes, ni 10 graines de variantes mélangées.
    distances = _read_csv(campaign / "distance_matrix.csv")
    first = next(row for row in distances if row["variant"] == "gnn_3_rounds_id"
                 and row["level"] == "1" and row["target"] == "geometries_a")
    assert float(first["mean"]) == pytest.approx(0.6)
    assert float(first["std_sample"]) == pytest.approx(math.sqrt(0.1))
    assert float(first["ci95"]) == pytest.approx(2.7764 * math.sqrt(0.1) / math.sqrt(5))  # Student, 4 ddl
    assert int(first["n"]) == 5
    geo = next(row for row in distances if row["variant"] == "gnn_3_rounds_geometry"
               and row["level"] == "1" and row["target"] == "geometries_a")
    assert float(geo["mean"]) == pytest.approx(0.55)
    forgetting = _read_csv(campaign / "forgetting.csv")
    old = next(row for row in forgetting if row["variant"] == "gnn_3_rounds_id"
               and row["level"] == "2" and row["target"] == "geometries_a")
    assert float(old["mean"]) == pytest.approx(0.06)
    # La dispersion des différences appariées est dix fois plus faible que
    # celle des distances ; les deux dispersions de niveaux ne sont pas additionnées.
    assert float(old["std_sample"]) == pytest.approx(math.sqrt(0.001))
    assert float(old["ci95"]) == pytest.approx(2.7764 * math.sqrt(0.001) / math.sqrt(5))
    assert int(old["n"]) == 5
    assert int(old["notable_forgetting_count"]) == 3
    assert old["notable_forgetting"] == "True"
    assert not any(row["target"] == "numbers_a" for row in forgetting)
    consolidated = _read_csv(campaign / "losses_by_target.csv")
    assert len(consolidated) == 200
    assert {row["seed"] for row in consolidated} == {"11", "12", "13", "14", "15"}
    introduced = next(row for row in consolidated if row["level"] == "2" and row["target"] == "digits_b")
    assert introduced["previous_distance"] == introduced["forgetting"] == ""
    for variant in summary["variants"]:
        assert (campaign / f"heatmap_{variant}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((campaign / "summary.json").read_text(encoding="utf-8"))["error_count"] == 72
    # Les seules sorties racine sont régénérables, sans modifier les runs.
    (campaign / "distance_matrix.csv").write_text("stale-report", encoding="utf-8")
    assert CurriculumReport(campaign).run() == summary
    assert _source_bytes(campaign) == before


def test_cli_uses_saved_config_not_current_contract(campaign, tmp_path, monkeypatch, capsys):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # Ce contrat courant est obsolète ; seule sa racine doit être utilisée.
    _yaml(config_dir / "locator.yaml", {"output": str(campaign), "profiles": {"standard": {"seeds": [99]}},
                                        "variants": {"wrong_variant": {}}, "forgetting_tolerance": 99})
    monkeypatch.setattr(report, "ConfigLoader", lambda: ConfigLoader(config_dir))
    monkeypatch.setattr(sys, "argv", ["curriculum", "--config", "locator", "--profile", "standard"])
    report.main()
    result = json.loads(capsys.readouterr().out)
    assert result["forgetting_tolerance"] == 0.05
    assert result["notable_forgetting_count"] == 72


def test_output_override_reads_relocated_campaign(campaign, tmp_path):
    relocated = tmp_path / "relocated"
    shutil.copytree(campaign, relocated)
    source_before = _source_bytes(campaign)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _yaml(config_dir / "locator.yaml", {"output": str(tmp_path / "absent")})
    summary = CurriculumReport.from_config("locator", output=relocated, loader=ConfigLoader(config_dir)).run()
    assert summary["row_count"] == 200
    assert (relocated / "distance_matrix.csv").is_file()
    assert not (campaign / "summary.json").exists()
    assert _source_bytes(campaign) == source_before


@pytest.mark.parametrize("damage,match", [
    ("missing_seed", "graines manquantes"),
    ("missing_variant", "variantes manquantes"),
    ("missing_level", "artefact manquant"),
    ("missing_best", "artefact manquant"),
    ("missing_seeds_provenance", "artefact manquant"),
    ("missing_targets_provenance", "artefact manquant"),
    ("missing_level_row", "mesures de niveaux"),
    ("missing_validation_target", "mesures de niveaux"),
    ("duplicate_row", "dupliquée"),
    ("campaign_validation_seeds", "provenance des graines"),
    ("level_validation_seeds", "provenance des graines"),
    ("status_seed", "identité"),
    ("status_variant", "identité"),
    ("status_profile", "identité"),
    ("incomplete_status", "statut"),
    ("nested_summary_identity", "résumés de niveau"),
    ("summary_missing_level", "résumé de niveaux"),
    ("level_summary_seed", "identité"),
    ("raw_config_mismatch", "configurations curriculum"),
    ("effective_config_mismatch", "configurations différentes"),
    ("target_test_split", "split de cible"),
    ("cumulative_context", "contexte cumulatif"),
    ("checkpoint_hash", "empreinte"),
    ("checkpoint_last", "meilleur checkpoint"),
    ("missing_transfer_provenance", "provenance du transfert"),
    ("missing_loss_weights", "poids de perte"),
    ("unweighted_total", "total_loss"),
    ("nonfinite", "non fini"),
    ("previous_distance", "non appariée"),
    ("forgetting", "non appariée"),
    ("notable_flag", "seuil d'oubli"),
    ("new_target_forgetting", "nouvelle cible"),
    ("summary_counter", "compteur d'oubli"),
    ("validation_count", "provenance de mesure"),
])
def test_incomplete_or_inconsistent_campaign_rejected_before_writes(campaign, damage, match):
    # Endommager le dernier run force la vérification complète de la campagne.
    directory = campaign / "gnn_3_rounds_id" / "seed_15"
    level_dir = campaign / "level_4" / "gnn_3_rounds_id" / "seed_15"
    csv_path = directory / "losses_by_target.csv"
    if damage == "missing_seed":
        shutil.rmtree(directory)
    elif damage == "missing_variant":
        shutil.rmtree(campaign / "gnn_3_rounds_id")
    elif damage == "missing_level":
        shutil.rmtree(level_dir)
    elif damage in ("missing_best", "missing_seeds_provenance", "missing_targets_provenance"):
        name = {"missing_best": "best.pt", "missing_seeds_provenance": "seeds.json", "missing_targets_provenance": "targets.json"}[damage]
        (level_dir / name).unlink()
    elif damage in ("campaign_validation_seeds", "level_validation_seeds"):
        origin = directory if damage == "campaign_validation_seeds" else level_dir
        _edit_json(origin / "seeds.json", validation_seeds=[9001, 9002, 9999])
    elif damage.startswith("status_"):
        key = damage.removeprefix("status_")
        _edit_json(directory / "status.json", **{key: {"seed": 999, "variant": "wrong", "profile": "smoke"}[key]})
    elif damage == "incomplete_status":
        _edit_json(level_dir / "status.json", status="running")
    elif damage in ("nested_summary_identity", "summary_missing_level"):
        value = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if damage == "nested_summary_identity":
            value["levels"][-1]["seed"] = 999
        else:
            value["levels"].pop()
        _json(directory / "summary.json", value)
    elif damage == "level_summary_seed":
        _edit_json(level_dir / "summary.json", seed=999)
    elif damage == "raw_config_mismatch":
        value = yaml.safe_load((directory / "effective_config.yaml").read_text(encoding="utf-8"))
        value["forgetting_tolerance"] = 0.2
        _yaml(directory / "effective_config.yaml", value)
    elif damage == "effective_config_mismatch":
        value = yaml.safe_load((level_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        value["metrics"]["collision_radius"] = 0.2
        _yaml(level_dir / "effective_config.yaml", value)
    elif damage == "target_test_split":
        value = json.loads((level_dir / "targets.json").read_text(encoding="utf-8"))
        value["validation_targets"][-1]["split"] = "test"
        _json(level_dir / "targets.json", value)
    elif damage == "cumulative_context":
        value = yaml.safe_load((level_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        value["curriculum"]["training_targets"].pop(0)
        _yaml(level_dir / "effective_config.yaml", value)
    elif damage == "checkpoint_hash":
        best = campaign / "level_3" / "gnn_3_rounds_id" / "seed_15" / "best.pt"
        best.write_bytes(b"checkpoint-changed")
    elif damage == "checkpoint_last":
        value = yaml.safe_load((level_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        value["initial_checkpoint"]["path"] = str(campaign / "level_3" / "gnn_3_rounds_id" / "seed_15" / "last.pt")
        _yaml(level_dir / "effective_config.yaml", value)
    elif damage in ("missing_transfer_provenance", "missing_loss_weights"):
        value = yaml.safe_load((level_dir / "effective_config.yaml").read_text(encoding="utf-8"))
        if damage == "missing_transfer_provenance":
            value.pop("initial_checkpoint")
        else:
            value["training"].pop("losses")
        _yaml(level_dir / "effective_config.yaml", value)
    elif damage == "summary_counter":
        _edit_json(directory / "summary.json", notable_forgetting_count=0)
    else:
        rows = _read_csv(csv_path)
        retained = next(row for row in rows if row["level"] == "4" and row["target"] == "geometries_a")
        introduced = next(row for row in rows if row["level"] == "4" and row["target"] == "numbers_b")
        if damage == "missing_level_row":
            rows.remove(retained)
        elif damage == "missing_validation_target":
            rows.remove(introduced)
        elif damage == "duplicate_row":
            rows.append(deepcopy(retained))
        elif damage == "unweighted_total":
            retained["total_loss"] = sum(float(retained[key]) for key in report.LOSS_TERMS)
        elif damage == "nonfinite":
            retained["final_distance"] = "nan"
        elif damage in ("previous_distance", "forgetting"):
            retained[damage] = 123
        elif damage == "notable_flag":
            retained["notable_forgetting"] = "False"
        elif damage == "new_target_forgetting":
            introduced["previous_distance"], introduced["forgetting"] = 1, 0
        elif damage == "validation_count":
            retained["validation_seed_count"] = 4
        else:
            pytest.fail(f"Cas de test inconnu : {damage}")
        _csv(csv_path, rows)
    # Un ancien rapport doit lui aussi rester intact si la vérification échoue.
    existing = campaign / "distance_matrix.csv"
    existing.write_bytes(b"previous-report")
    before = {path.relative_to(campaign): path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        CurriculumReport(campaign).run()
    after = {path.relative_to(campaign): path.read_bytes() for path in campaign.rglob("*") if path.is_file()}
    assert after == before
    assert existing.read_bytes() == b"previous-report"
    assert not (campaign / "summary.json").exists()
    assert not (campaign / "losses_by_target.csv").exists()
    assert not list(campaign.glob("heatmap_*.png"))


def test_float_rounding_of_weighted_total_is_tolerated(campaign):
    path = campaign / "gnn_3_rounds_id" / "seed_11" / "losses_by_target.csv"
    rows = _read_csv(path)
    rows[0]["total_loss"] = float(rows[0]["total_loss"]) + 1e-6
    _csv(path, rows)
    assert CurriculumReport(campaign).run()["row_count"] == 200


def test_smoke_requires_its_declared_seed_and_cannot_estimate_uncertainty(tmp_path):
    campaign = _build_campaign(tmp_path, "smoke")
    summary = CurriculumReport(campaign, "smoke").run()
    assert summary["training_seeds"] == [1]
    assert summary["forgetting_tolerance"] == 0.05
    assert summary["error_count"] == 0 and summary["retention_ok"]
    assert all(row["n"] == 1 and row["std_sample"] is None and row["ci95"] is None for row in summary["distance_matrix"])
    assert all(row["n"] == 1 and row["std_sample"] is None and row["ci95"] is None for row in summary["forgetting"])
    with pytest.raises(ValueError, match="graines manquantes"):
        CurriculumReport(campaign, "standard").run()


def test_no_source_does_not_create_output(tmp_path):
    root = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="aucune campagne"):
        CurriculumReport(root).run()
    assert not root.exists()
