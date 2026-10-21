"""Statistiques par condition, par graine et par épisode, à partir des lignes brutes."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AGGREGATABLE_METRICS = ["final_distance", "success", "collision_rate", "normalized_regularity", "final_components"]


@dataclass(frozen=True)
class ResultsTable:
    rows: list[dict]

    @classmethod
    def read_csv(cls, path: Path) -> "ResultsTable":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ("training_seed", "episode_seed", "n_robots", "shape_id", "convergence_step", "final_components", "step"):
                row[key] = int(row[key]) if row[key] else None
            for key in ("failure_fraction", "final_distance", "collision_rate", "normalized_regularity"):
                row[key] = float(row[key])
            if row["success"] not in ("True", "False"):
                raise ValueError(f"valeur de succès invalide : {row['success']!r}")
            row["success"] = row["success"] == "True"
        identities = [(row["controller"], row["training_seed"], row["episode_seed"], *cls.condition_of(row)) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError("le journal contient des épisodes dupliqués")
        return cls(rows)

    @staticmethod
    def condition_of(row: dict) -> tuple:
        return (row["n_robots"], row["failure_fraction"], row["shape"])

    @staticmethod
    def statistics(values) -> dict[str, float]:
        values = np.asarray(values, dtype=float)
        deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        return {"mean": float(values.mean()), "std": deviation, "ci95": 1.96 * deviation / math.sqrt(len(values)), "n": len(values)}

    def aggregates(self, metric: str = "final_distance") -> dict[tuple, dict[str, float]]:
        groups: dict[tuple, list[float]] = defaultdict(list)
        for row in self.rows:
            if row[metric] is not None:
                groups[(row["controller"], row["training_seed"], *self.condition_of(row))].append(float(row[metric]))
        return {key: self.statistics(values) for key, values in groups.items()}

    def variation_between_seeds(self, metric: str = "final_distance") -> dict[tuple, dict]:
        groups: dict[tuple, list[float]] = defaultdict(list)
        for (controller, seed, *condition), aggregate in self.aggregates(metric).items():
            if seed is not None:
                groups[(controller, *condition)].append(aggregate["mean"])
        return {key: self.statistics(values) for key, values in groups.items()}


@dataclass(frozen=True)
class PairedComparison:
    a: str
    b: str
    differences: dict[str, np.ndarray]
    training_seeds: tuple[int | None, ...]

    @staticmethod
    def build(rows: list[dict], controller_a: str, controller_b: str) -> "PairedComparison":
        groups: dict[str, dict[tuple, dict[int | None, dict]]] = {
            controller_a: defaultdict(dict), controller_b: defaultdict(dict),
        }
        for row in rows:
            if row["controller"] not in groups:
                continue
            key = (row["episode_seed"], *ResultsTable.condition_of(row))
            group = groups[row["controller"]][key]
            seed = row["training_seed"]
            if seed in group:
                raise ValueError("plusieurs lignes pour le même épisode et la même graine")
            group[seed] = row
        differences: dict[str, list[float]] = {metric: [] for metric in AGGREGATABLE_METRICS}
        training_seeds = []
        for key, group_a in groups[controller_a].items():
            group_b = groups[controller_b].get(key, {})
            for seed_a, row_a in group_a.items():
                candidates = group_b.items() if seed_a is None else [(seed_b, row_b) for seed_b, row_b in group_b.items() if seed_b in (None, seed_a)]
                for seed_b, row_b in candidates:
                    training_seeds.append(seed_a if seed_a is not None else seed_b)
                    for metric in AGGREGATABLE_METRICS:
                        differences[metric].append(float(row_a[metric]) - float(row_b[metric]))
        return PairedComparison(controller_a, controller_b, {metric: np.array(values) for metric, values in differences.items()}, tuple(training_seeds))

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            metric: {"mean_gain": stats["mean"], "std": stats["std"], "ci95": stats["ci95"], "paired_count": stats["n"]}
            for metric, values in self.differences.items() if len(values)
            for stats in [ResultsTable.statistics(values)]
        }

    def seed_summary(self) -> dict[str, dict]:
        output = {}
        seeds = tuple(dict.fromkeys(self.training_seeds))
        for metric, values in self.differences.items():
            if not len(values):
                continue
            means = [float(values[np.array([candidate == seed for candidate in self.training_seeds])].mean()) for seed in seeds]
            output[metric] = ResultsTable.statistics(means)
        return output
