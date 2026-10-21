"""Journaux bruts d'entraînement : pertes par itération et point de validation."""

from __future__ import annotations

import csv
from pathlib import Path


class TrainingLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._losses: list[dict[str, object]] = []
        self._validations: list[dict[str, object]] = []
        self._validation_targets: list[dict[str, object]] = []
        self._validation_episodes: list[dict[str, object]] = []
        self._training_targets: list[dict[str, object]] = []

    def loss(self, iteration: int, total: float, detail: dict[str, float]) -> None:
        self._losses.append({"iteration": iteration, "total": total, **detail})

    def target_loss(self, iteration: int, shape_id: int, total: float, detail: dict[str, float]) -> None:
        self._training_targets.append({"iteration": iteration, "shape_id": shape_id, "total": total, **detail})

    def validation(self, iteration: int, values: dict[str, object]) -> None:
        # Les détails restent dans des tableaux séparés, sans modifier les colonnes historiques.
        values = dict(values)
        for key, rows in (("validation_targets", self._validation_targets), ("validation_episodes", self._validation_episodes)):
            rows.extend({"iteration": iteration, **row} for row in values.pop(key, []))
        self._validations.append({"iteration": iteration, **values})

    @property
    def last_loss(self) -> dict[str, object]:
        return self._losses[-1]

    def write(self) -> None:
        self._write_csv(self.output_dir / "losses.csv", self._losses)
        self._write_csv(self.output_dir / "validation.csv", self._validations)
        if self._training_targets:
            self._write_csv(self.output_dir / "training_targets.csv", self._training_targets)
        if self._validation_targets:
            self._write_csv(self.output_dir / "validation_targets.csv", self._validation_targets)
        if self._validation_episodes:
            self._write_csv(self.output_dir / "validation_episodes.csv", self._validation_episodes)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if not rows:
            raise ValueError(f"journal vide : {path.name}")
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
