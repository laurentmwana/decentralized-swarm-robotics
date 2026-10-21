"""Suite de metriques : assemble les cinq metriques de la phase 2 en un rapport unique.

La suite est construite depuis les configurations versionnees (metrics.yaml +
simulator.yaml + shapes.yaml) — aucun parametre numerique n'est code en dur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.config import MetricsConfig, SimulatorConfig

from .base import Metric
from .chamfer import ChamferDistance
from .collisions import CollisionRate
from .connectivity import GraphConnectivity
from .regularity import SpreadRegularity
from .success import SuccessCriterion


@dataclass(frozen=True)
class MetricReport:
    values: dict[str, Any] = field(default_factory=dict)

    @property
    def distance_curve(self) -> np.ndarray:
        return self.values["distance_curve"]

    @property
    def success(self) -> bool:
        return bool(self.values["success"])

    @property
    def convergence_step(self) -> int | None:
        return self.values["convergence_step"]


class MetricSuite:
    def __init__(self, metrics: MetricsConfig, simulator: SimulatorConfig, target_radius: float) -> None:
        self.target_radius = target_radius
        self.success_threshold = metrics.success_threshold
        self.metrics = [
            ChamferDistance(target_radius),
            SuccessCriterion(metrics.success_threshold, target_radius),
            CollisionRate(metrics.collision_radius),
            SpreadRegularity(metrics.regularity_reference),
            GraphConnectivity(simulator.r_c),
        ]

    def compute(self, positions: np.ndarray, active: np.ndarray, target: np.ndarray) -> MetricReport:
        values: dict[str, Any] = {}
        for metric in self.metrics:
            values.update(metric.compute(positions, active, target))
        return MetricReport(values=values)
