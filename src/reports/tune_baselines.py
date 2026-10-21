"""Réglage des baselines sous budget fixe (phase 3).

Crosse la grille de config/baselines.yaml (budget = nombre d'essais fixé à l'avance,
vérifié), évalue chaque essai sur un épisode cercle avec la suite de métriques, et
journalise tout : le meilleur jeu de paramètres est copié dans `rules:` de la
configuration, l'effort de réglage reste documenté (leçon d'audit : une baseline
mal réglée flatte le GNN).

Usage : python -m src.reports.tune_baselines
"""

from __future__ import annotations

import csv
import itertools
import json
import subprocess
from pathlib import Path

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.flocking import FlockingController
from src.controllers.manual_shape import ManualShapeController
from src.metrics import MetricSuite
from src.shapes.index import generate_index
from src.simulator.episode import Episode

OUTPUT = Path("outputs/references/tuning")

RULES: dict[str, type] = {"flocking": FlockingController, "manual": ManualShapeController}


def _version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "inconnue"


def _trials(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[c] for c in keys))]


def run() -> dict:
    loader = ConfigLoader()
    cfg = loader.load_raw("baselines")
    cfg_sim = loader.load_simulator()
    cfg_metrics = loader.load_metrics()
    cfg_shapes = loader.load_shapes()

    index, _ = generate_index(cfg_shapes, Path("outputs/shapes"))
    target = index.point_cloud_by_key(cfg["tuning_target"])
    suite = MetricSuite(cfg_metrics, cfg_sim, cfg_shapes.target_radius)
    episode = Episode(cfg_sim)
    mission = Mission(shape_id=0, target=target)

    overall_summary: dict[str, object] = {"budget_trials": cfg["budget_trials"], "criterion": cfg["criterion"]}
    for rule_name, controller_class in RULES.items():
        trials = _trials(cfg["grids"][rule_name])
        if len(trials) > int(cfg["budget_trials"]):
            raise ValueError(f"budget de reglage depasse : {rule_name} compte {len(trials)} essais > {cfg['budget_trials']}")

        output_dir = OUTPUT / rule_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for trial in trials:
            controller = controller_class(**trial)
            result = episode.execute(
                mission=mission, controller=controller, n_robots=int(cfg["tuning_swarm_size"]), episode_seed=int(cfg["tuning_seed"])
            )
            report = suite.compute(result.positions, result.active, target).values
            results.append({**trial, "final_distance": float(report["final_distance"]), "success": bool(report["success"])})

        best = min(results, key=lambda r: r["final_distance"])
        with (output_dir / "log.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        summary = {
            "rule": rule_name,
            "n_robots": int(cfg["tuning_swarm_size"]),
            "tuning_seed": int(cfg["tuning_seed"]),
            "target": cfg["tuning_target"],
            "completed_trials": len(results),
            "best": best,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "effective_config.yaml").write_text(
            "source: config/baselines.yaml (section grilles)\n", encoding="utf-8"
        )
        (output_dir / "seeds.json").write_text(json.dumps({"tuning_seed": cfg["tuning_seed"]}), encoding="utf-8")
        (output_dir / "version.txt").write_text(_version() + "\n", encoding="utf-8")
        overall_summary[rule_name] = summary
        print(f"[tuning] {rule_name} : {len(results)} essais, meilleur = {best}")
    return overall_summary


if __name__ == "__main__":
    run()
