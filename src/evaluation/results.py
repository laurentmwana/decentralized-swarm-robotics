"""Statistiques par condition, par graine et par épisode, à partir des lignes brutes."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AGGREGATABLE_METRICS = ["final_distance", "success", "collision_rate", "normalized_regularity", "final_components"]
CI95_METHOD = "student_t_0.975_sample_standard_error"

# Quantiles 0,975 de la loi de Student, ddl 1 à 30 (tables standard). Avec 5 graines
# d'entraînement (ddl 4), 1,96 sous-estimerait l'intervalle d'environ 30 %.
_T975 = (
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
)


def t_critical_95(degrees_of_freedom: int) -> float:
    """Quantile bilatéral 95 % de Student ; développement de Cornish-Fisher au-delà de 30 ddl."""
    if degrees_of_freedom < 1:
        raise ValueError(f"degrés de liberté invalides : {degrees_of_freedom}")
    if degrees_of_freedom <= len(_T975):
        return _T975[degrees_of_freedom - 1]
    z, df = 1.959964, float(degrees_of_freedom)
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


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
        if len(values) < 2:
            return {"mean": float(values.mean()), "std": 0.0, "ci95": 0.0, "n": len(values)}
        deviation = float(values.std(ddof=1))
        ci95 = t_critical_95(len(values) - 1) * deviation / math.sqrt(len(values))
        return {"mean": float(values.mean()), "std": deviation, "ci95": ci95, "n": len(values)}

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
