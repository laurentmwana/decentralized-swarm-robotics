"""Évaluateur : exécute la campagne et écrit une ligne brute par épisode.

Mêmes graines d'épisode pour tous les contrôleurs (positions initiales et pannes
identiques -> comparaison appariée possible). Les positions de quelques épisodes par
condition sont enregistrées (.npz) pour permettre le recalcul des métriques.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.config import ConfigLoader, MetricsConfig, SimulatorConfig, ShapesConfig
from src.common.mission import Mission
from src.controllers.base import Controller
from src.metrics import MetricSuite
from src.shapes.index import ShapeIndex

from .scenario import Scenario, ScenarioGrid

ROW_KEYS = [
    "controller", "training_seed", "episode_seed", "n_robots", "failure_fraction",
    "shape", "shape_id", "final_distance", "convergence_step", "success",
    "collision_rate", "normalized_regularity", "final_components", "step",
]


@dataclass
class Evaluator:
    grid: ScenarioGrid
    suite: MetricSuite
    index: ShapeIndex
    cfg_sim: SimulatorConfig
    output_dir: Path
    positions_per_condition: int = 0
    _seen: dict[tuple, int] = field(default_factory=dict, repr=False)

    @staticmethod
    def from_config(loader: ConfigLoader | None = None, output_dir: Path = Path("outputs/evaluation"), evaluation_config: dict | None = None) -> "Evaluator":
        loader = loader or ConfigLoader()
        cfg_eval = evaluation_config if evaluation_config is not None else loader.load_raw("evaluation")
        cfg_sim = loader.load_simulator()
        cfg_metrics = loader.load_metrics()
        cfg_shapes = loader.load_shapes()
        index = ShapeIndex.load(Path("outputs/shapes"))
        return Evaluator(
            grid=ScenarioGrid.from_config(cfg_eval),
            suite=MetricSuite(cfg_metrics, cfg_sim, cfg_shapes.target_radius),
            index=index,
            cfg_sim=cfg_sim,
            output_dir=Path(output_dir),
            positions_per_condition=int(cfg_eval["positions_per_condition"]),
        )

    def evaluate(self, name: str, controller: Controller, training_seed: int | None, scenarios: list[Scenario]) -> list[dict]:
        from src.simulator.episode import Episode

        rows = []
        for scenario in scenarios:
            target = self.index.point_cloud_by_key(scenario.shape_key)
            shape_id = self.index.shape_id(scenario.shape_key)
            cfg = self.grid.simulator_config(self.cfg_sim)
            cfg = _with_failure(cfg, scenario.failure_fraction)
            episode = Episode(cfg)
            controller.reset()
            result = episode.execute(
                mission=Mission(shape_id=shape_id, target=target),
                controller=controller,
                n_robots=scenario.n_robots,
                episode_seed=scenario.episode_seed,
            )
            values = self.suite.compute(result.positions, result.active, target).values
            rows.append(
                {
                    "controller": name,
                    "training_seed": training_seed,
                    "episode_seed": scenario.episode_seed,
                    "n_robots": scenario.n_robots,
                    "failure_fraction": scenario.failure_fraction,
                    "shape": scenario.shape_key,
                    "shape_id": shape_id,
                    "final_distance": float(values["final_distance"]),
                    "convergence_step": values["convergence_step"],
                    "success": bool(values["success"]),
                    "collision_rate": float(values["collision_rate"]),
                    "normalized_regularity": float(values["normalized_regularity"]),
                    "final_components": int(values["final_components"]),
                    "step": result.step,
                }
            )
            self._record_positions(name, training_seed, scenario, result, target)
        return rows

    def _record_positions(self, name, training_seed, scenario: Scenario, result, target) -> None:
        condition = (name, training_seed, scenario.n_robots, scenario.failure_fraction, scenario.shape_key)
        seen_count = self._seen.get(condition, 0)
        if seen_count >= self.positions_per_condition:
            return
        self._seen[condition] = seen_count + 1
        shape = scenario.shape_key
        condition_name = f"n{scenario.n_robots}_failure{int(round(scenario.failure_fraction * 100)):02d}_{shape}"
        base = self.output_dir / name / (f"seed_{training_seed}" if training_seed is not None else "")
        output_dir = base / "positions" / condition_name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"seed_{scenario.episode_seed}.npz",
            positions=result.positions,
            active=result.active,
            target=target,
            target_radius=self.suite.target_radius,
        )


def _with_failure(cfg: SimulatorConfig, fraction: float) -> SimulatorConfig:
    import dataclasses

    return dataclasses.replace(cfg, failure_fraction=fraction)
