"""Campagne reproductible sur contrôleurs gelés, sans entraînement implicite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.config import ConfigLoader
from src.evaluation.builders import FigureBuilder, TableBuilder
from src.evaluation.evaluator import ROW_KEYS, Evaluator
from src.evaluation.factory import ControllerFactory
from src.evaluation.results import AGGREGATABLE_METRICS, PairedComparison, ResultsTable

OUTPUT = Path("outputs/evaluation")


class Campaign:
    def __init__(self, evaluator: Evaluator, factory: ControllerFactory, config: dict, output: Path, loader: ConfigLoader) -> None:
        self.evaluator = evaluator
        self.factory = factory
        self.config = config
        self.output = Path(output)
        self.loader = loader

    @classmethod
    def from_config(cls, loader: ConfigLoader | None = None, output: Path = OUTPUT, config_name: str = "evaluation", repetitions: int | None = None) -> "Campaign":
        loader = loader or ConfigLoader()
        config = loader.load_raw(config_name)
        if repetitions is not None:
            config = {**config, "repetitions": repetitions}
        evaluator = Evaluator.from_config(loader, output, config)
        factory = ControllerFactory(loader, config)
        return cls(evaluator, factory, config, output, loader)

    def run(self, controllers: list[str] | None = None) -> dict:
        names = list(self.config["controllers"]) if controllers is None else controllers
        if not names or len(names) != len(set(names)):
            raise ValueError("la sélection des contrôleurs doit être non vide et sans doublons")
        unknown = set(names) - self.config["controllers"].keys()
        if unknown:
            raise ValueError(f"contrôleurs inconnus : {sorted(unknown)}")
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"dossier de campagne déjà occupé : {self.output}")
        versions = {name: self.factory.versions(name) for name in names}
        prepared = {(name, seed): self.factory.build(name, seed) for name, seeds in versions.items() for seed in seeds}
        scenarios = self.evaluator.grid.scenarios()
        for scenario in scenarios:
            self.evaluator.index.point_cloud_by_key(scenario.shape_key)
            self.evaluator.index.shape_id(scenario.shape_key)
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_metadata(self.output, names, versions)
        all_rows = []
        for name in names:
            controller_rows = []
            for seed in versions[name]:
                directory = self.output / name / (f"seed_{seed}" if seed is not None else "")
                directory.mkdir(parents=True, exist_ok=True)
                self._write_metadata(directory, [name], {name: [seed]})
                rows = self.evaluator.evaluate(name, prepared[(name, seed)], seed, scenarios)
                self._write_rows(rows, directory / "episodes.csv")
                (directory / "summary.json").write_text(json.dumps({"rows": len(rows), "status": "complete"}), encoding="utf-8")
                controller_rows.extend(rows)
            self._write_rows(controller_rows, self.output / name / "episodes.csv")
            all_rows.extend(controller_rows)
        self._write_rows(all_rows, self.output / "episodes.csv")
        expected = len(scenarios) * sum(len(seeds) for seeds in versions.values())
        if len(all_rows) != expected:
            raise ValueError(f"nombre de lignes brutes inattendu : {len(all_rows)} != {expected}")
        table = ResultsTable.read_csv(self.output / "episodes.csv")
        summary = self.build_reports(table, names, versions)
        summary.update(rows=len(all_rows), expected_rows=expected, status="complete")
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def _write_metadata(self, directory: Path, names: list[str], versions: dict) -> None:
        snapshot = {
            "evaluation": {**self.config, "controllers": {name: self.config["controllers"][name] for name in names}},
            "simulator": asdict(self.loader.load_simulator()),
            "metrics": asdict(self.loader.load_metrics()),
            "shapes": asdict(self.loader.load_shapes()),
            "baselines": self.loader.load_raw("baselines"),
        }
        snapshot["training_runs"] = {}
        for name in names:
            if self.config["controllers"][name]["type"] != "trained":
                continue
            snapshot["training_runs"][name] = {}
            for seed in versions[name]:
                run_snapshot = self.factory.snapshot(name, seed)
                checkpoint = self.factory.run_directory(name, seed) / "best.pt"
                with checkpoint.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                snapshot["training_runs"][name][seed] = {"config": run_snapshot, "checkpoint_sha256": digest}
        (directory / "effective_config.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
        seeds = {"training_seeds": versions, "episode_seeds": [self.evaluator.grid.episode_seed(index) for index in range(self.evaluator.grid.repetitions)]}
        (directory / "seeds.json").write_text(json.dumps(seeds, indent=2), encoding="utf-8")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        version = f"commit={commit}\npython={platform.python_version()}\nnumpy={np.__version__}\ntorch={torch.__version__}\n"
        (directory / "version.txt").write_text(version, encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")

    @staticmethod
    def _write_rows(rows: list[dict], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=ROW_KEYS)
            writer.writeheader()
            writer.writerows(rows)

    def build_reports(self, table: ResultsTable, names: list[str], versions: dict) -> dict:
        tables = TableBuilder(self.output)
        tables.write_aggregates(table, AGGREGATABLE_METRICS)
        tables.write_markdown(table, "final_distance", self.output / "aggregates.md")
        figures = FigureBuilder(self.output)
        shape = self.evaluator.grid.shapes[0]
        size = self.evaluator.grid.swarm_sizes[0]
        fraction = self.evaluator.grid.failure_fractions[0]
        condition = f"n{size}_failure{int(round(fraction * 100)):02d}_{shape.replace('@', '_at_')}"
        position_dirs = []
        for name in names:
            for seed in versions[name]:
                directory = self.output / name / (f"seed_{seed}" if seed is not None else "") / "positions" / condition
                if directory.is_dir():
                    position_dirs.append((f"{name} / {seed}" if seed is not None else name, directory))
        figure_dir = self.output / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        threshold = self.evaluator.suite.success_threshold
        figures.distance_curves(position_dirs, figure_dir / "distance_curves.png", f"Distance — {shape}, {size} robots", threshold)
        figures.distance_boxes(table.rows, figure_dir / "final_distance_box.png", "Distribution des distances finales")
        figures.success_by_size(table, figure_dir / "success_by_size.png", f"Succès (seuil {threshold:g})")
        seed_variation = [
            {"controller": controller, "n_robots": size, "failure_fraction": fraction, "shape": shape, **statistics}
            for (controller, size, fraction, shape), statistics in table.variation_between_seeds().items()
        ]
        pairings = {f"{a}_minus_{b}": {"episodes": comparison.summary(), "seeds": comparison.seed_summary()}
                    for a, b in combinations(names, 2) for comparison in [PairedComparison.build(table.rows, a, b)]}
        return {"variation_between_seeds": seed_variation, "pairings": pairings, "success_threshold": threshold}


def run(controllers: list[str] | None = None, repetitions: int | None = None, output: Path = OUTPUT, config_name: str = "evaluation") -> dict:
    return Campaign.from_config(output=output, config_name=config_name, repetitions=repetitions).run(controllers)


def expected_row_count(config: dict, trained_seed_count: int | None = None) -> int:
    seed_count = len(config["training_seeds"]) if trained_seed_count is None else trained_seed_count
    conditions = len(config["swarm_sizes"]) * len(config["failure_fractions"]) * len(config["shapes"])
    controller_count = sum(1 if settings["type"] == "baseline" else seed_count for settings in config["controllers"].values())
    return conditions * config["repetitions"] * controller_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controllers", nargs="+", default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", default="evaluation")
    args = parser.parse_args()
    run(args.controllers, args.repetitions, args.output, args.config)


if __name__ == "__main__":
    main()
