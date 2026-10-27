"""Curriculum cumulatif, transfert du meilleur état et mesure explicite de l'oubli."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.config import ConfigLoader
from src.common.seeds import SeedManager
from src.metrics import MetricSuite
from src.shapes.index import ShapeIndex
from src.training.checkpoint import CheckpointManager
from src.training.losses import CompositeLoss
from src.training.rollout import DifferentiableSimulator
from src.training.train import run as train_run
from src.training.train import build_model, _write_json
from src.training.trainer import _version, validation_seeds


@dataclass(frozen=True)
class CurriculumLevel:
    number: int
    name: str
    training_targets: tuple[str, ...]
    validation_targets: tuple[str, ...]


class ForgettingMonitor:
    def __init__(self, tolerance: float) -> None:
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError("tolérance d'oubli invalide")
        self.tolerance = tolerance
        self.rows: list[dict] = []
        self.previous: dict[str, float] = {}

    def record(self, level: int, measurements: list[dict]) -> None:
        for measurement in measurements:
            target = measurement["target"]
            distance = float(measurement["final_distance"])
            previous = self.previous.get(target)
            change = None if previous is None else distance - previous
            self.rows.append({"level": level, **measurement, "previous_distance": previous,
                              "forgetting": change,
                              "notable_forgetting": change is not None and change > self.tolerance})
            self.previous[target] = distance

    def measure(self, directory: Path, index: ShapeIndex, keys: tuple[str, ...]) -> list[dict]:
        from src.common.config import ModelConfig, SimulatorConfig, MetricsConfig

        saved = yaml.safe_load((directory / "effective_config.yaml").read_text(encoding="utf-8"))
        cfg_sim = SimulatorConfig(**saved["simulator"])
        model = build_model(ModelConfig(**saved["model"]), {}, cfg_sim.v_max)
        CheckpointManager(directory).load(model, "best")
        model.eval()
        model.requires_grad_(False)
        simulator = DifferentiableSimulator(model, cfg_sim)
        loss = CompositeLoss.from_config(saved["training"], cfg_sim, saved["shapes"]["target_radius"],
                                         saved["metrics"]["collision_radius"])
        suite = MetricSuite(MetricsConfig(**saved["metrics"]), cfg_sim, saved["shapes"]["target_radius"])
        profile = saved["training"]["profiles"][saved["profile"]]
        seeds = validation_seeds(profile)
        measurements = []
        with torch.no_grad():
            for key in keys:
                target = torch.tensor(index.point_cloud_by_key(key), dtype=torch.float32)
                values = []
                for seed in seeds:
                    generator = torch.Generator().manual_seed(seed)
                    positions = simulator.initial_positions(profile["n_robots"], cfg_sim.spawn_radius, generator)
                    rollout = simulator.rollout(positions, index.shape_id(key), target, saved["communication"])
                    total, detail = loss(rollout, target)
                    history = rollout.positions.numpy()
                    metrics = suite.compute(history, np.ones(history.shape[:2], dtype=bool), target.numpy()).values
                    values.append({"total_loss": float(total), "final_distance": float(metrics["final_distance"]), **detail})
                measurements.append({"target": key, "shape_id": index.shape_id(key), "validation_seed_count": len(seeds),
                                     **{name: float(np.mean([row[name] for row in values])) for name in values[0]}})
        return measurements

    def write(self, output: Path) -> None:
        if not self.rows:
            raise ValueError("aucune mesure d'oubli disponible")
        with (output / "losses_by_target.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


class Curriculum:
    def __init__(self, config: dict, loader: ConfigLoader | None = None) -> None:
        self.config = config
        self.loader = loader or ConfigLoader()
        self.output = Path(config["output"])
        if config["target_sampling"] != "cyclic" or config["transfer_checkpoint"] != "best":
            raise ValueError("le curriculum exige un échantillonnage cyclique et le transfert du meilleur état")
        shapes = self.loader.load_shapes()
        self.levels = []
        training, validation, used_types = [], [], set()
        for number, definition in enumerate(config["levels"], start=1):
            kinds = definition["shape_types"]
            if not kinds or len(set(kinds)) != len(kinds) or used_types.intersection(kinds):
                raise ValueError("types de formes vides ou répétés dans le curriculum")
            for kind in kinds:
                if kind not in shapes.splits:
                    raise ValueError(f"type de forme inconnu : {kind}")
                training.extend(f"{kind}_{name}" for name in shapes.splits[kind]["train"])
                validation.extend(f"{kind}_{name}" for name in shapes.splits[kind]["val"])
            used_types.update(kinds)
            self.levels.append(CurriculumLevel(number, definition["name"], tuple(training), tuple(validation)))
        if not self.levels:
            raise ValueError("curriculum vide")
        for profile in config["profiles"].values():
            validation_seeds(profile)
            if not profile["seeds"] or len(set(profile["seeds"])) != len(profile["seeds"]):
                raise ValueError("graines d'entraînement vides ou répétées")
            if min(profile[key] for key in ("iterations", "n_robots", "horizon", "validation_interval")) <= 0:
                raise ValueError("budget du curriculum invalide")
            if set(profile["seeds"]) & set(profile["validation_seeds"]):
                raise ValueError("graines d'entraînement et de validation communes")
        ForgettingMonitor(float(config["forgetting_tolerance"]))

    @classmethod
    def from_config(cls, config_name: str = "curriculum", loader: ConfigLoader | None = None) -> "Curriculum":
        loader = loader or ConfigLoader()
        return cls(loader.load_raw(config_name), loader)

    def plan(self, variant: str, seed: int, profile: str) -> dict:
        if variant not in self.config["variants"]:
            raise ValueError(f"variante de curriculum inconnue : {variant}")
        budget = self.config["profiles"][profile]
        base = self.loader.load_raw(self.config["training_config"])
        levels = []
        for level in self.levels:
            validations = len(level.training_targets) + len(level.validation_targets)
            validation_count = (budget["iterations"] - 1) // budget["validation_interval"] + 1
            levels.append({"level": level.number, "name": level.name,
                           "training_targets": list(level.training_targets), "validation_targets": list(level.validation_targets),
                           "optimization_rollout_steps": budget["iterations"] * base["optimizer"]["batch_episodes"] * budget["horizon"],
                           "validation_rollout_steps": validation_count * validations * len(budget["validation_seeds"]) * budget["horizon"],
                           "output": str(self.output / f"level_{level.number}" / variant / f"seed_{seed}")})
        return {"variant": variant, "seed": seed, "profile": profile, "budget": budget,
                "levels": levels, "duration_seconds": None,
                "duration_note": "Coût algorithmique seulement ; aucune durée mesurée ni optimisation exécutée."}

    def run(self, variant: str, seed: int, profile: str) -> dict:
        plan = self.plan(variant, seed, profile)
        campaign_dir = self.output / variant / f"seed_{seed}"
        directories = [campaign_dir] + [Path(level["output"]) for level in plan["levels"]]
        for directory in directories:
            if directory.exists():
                raise FileExistsError(f"dossier de curriculum déjà occupé : {directory}")
        campaign_dir.mkdir(parents=True, exist_ok=False)
        (campaign_dir / "effective_config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        _write_json(campaign_dir / "seeds.json", {**SeedManager(seed).log(), "validation_seeds": self.config["profiles"][profile]["validation_seeds"]})
        (campaign_dir / "version.txt").write_text(_version() + "\n", encoding="utf-8")
        _write_json(campaign_dir / "status.json", {"status": "running", "variant": variant, "seed": seed, "profile": profile})
        monitor = ForgettingMonitor(float(self.config["forgetting_tolerance"]))
        checkpoint = None
        summaries = []
        try:
            for level in self.levels:
                cfg_training = self.loader.load_raw(self.config["training_config"])
                cfg_training["variants"] = self.config["variants"]
                cfg_training["profiles"] = self.config["profiles"]
                cfg_training["training_target"] = level.training_targets[0]
                cfg_training["validation_target"] = level.validation_targets[0]
                context = {"config": self.config, "level": level.number, "name": level.name,
                           "training_targets": list(level.training_targets), "validation_targets": list(level.validation_targets),
                           "optimizer_state_transfer": False}
                output = self.output / f"level_{level.number}"
                summary = train_run(variant, seed, profile, output, loader=self.loader, training_config=cfg_training,
                                    training_keys=list(level.training_targets), validation_keys=list(level.validation_targets),
                                    initial_checkpoint=checkpoint, curriculum_context=context)
                directory = output / variant / f"seed_{seed}"
                index = ShapeIndex.load(Path("outputs/shapes"))
                monitor.record(level.number, monitor.measure(directory, index, level.training_targets + level.validation_targets))
                monitor.write(campaign_dir)
                checkpoint = directory / "best.pt"
                summaries.append({"level": level.number, **summary})
            result = {"status": "completed", "variant": variant, "seed": seed, "profile": profile,
                      "levels": summaries, "forgetting_tolerance": monitor.tolerance,
                      "notable_forgetting_count": sum(row["notable_forgetting"] for row in monitor.rows),
                      "final_checkpoint": str(checkpoint)}
            _write_json(campaign_dir / "summary.json", result)
            _write_json(campaign_dir / "status.json", {"status": "completed", "variant": variant, "seed": seed, "profile": profile})
            return result
        except BaseException as error:
            _write_json(campaign_dir / "status.json", {"status": "failed", "error": str(error), "variant": variant, "seed": seed, "profile": profile})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="curriculum")
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    curriculum = Curriculum.from_config(args.config)
    if args.all and (args.variant is not None or args.seed is not None):
        parser.error("--all ne se combine pas avec --variant ou --seed")
    if not args.all and (args.variant is None or args.seed is None):
        parser.error("fournir --variant et --seed, ou --all")
    variants = list(curriculum.config["variants"]) if args.all else [args.variant]
    seeds = curriculum.config["profiles"][args.profile]["seeds"] if args.all else [args.seed]
    plans = [curriculum.plan(variant, seed, args.profile) for variant in variants for seed in seeds]
    if args.dry_run:
        print(json.dumps({"jobs": plans}, indent=2, ensure_ascii=False))
        return
    for plan in plans:
        directories = [curriculum.output / plan["variant"] / f"seed_{plan['seed']}"] + [Path(level["output"]) for level in plan["levels"]]
        if any(directory.exists() for directory in directories):
            raise FileExistsError("un dossier de curriculum existe déjà ; aucun entraînement lancé")
    for plan in plans:
        curriculum.run(plan["variant"], plan["seed"], args.profile)


if __name__ == "__main__":
    main()
