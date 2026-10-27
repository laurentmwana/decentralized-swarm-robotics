"""Étude de robustesse sur contrôleurs gelés ; aucune optimisation implicite.

Les conditions sont stratifiées par forme, taille et panne ordinaire. Les écarts
entre épisodes et entre graines d'entraînement sont rapportés séparément.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import yaml

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.evaluation.factory import ControllerFactory
from src.evaluation.results import CI95_METHOD, ResultsTable
from src.evaluation.scenario import Scenario, ScenarioGrid
from src.metrics import MetricSuite
from src.shapes.index import ShapeIndex
from src.simulator.episode import Episode
from src.simulator.graph import CommunicationGraph
from src.simulator.perturbations import CommunicationNoise, FaultyRobot, Perturbation, PositionNoise, RangeReduction


@dataclass(frozen=True)
class PerturbationLevel:
    name: str
    strength: float
    index: int

    def build(self) -> Perturbation:
        constructors = {
            "position_noise": lambda: PositionNoise(std=self.strength),
            "communication_noise": lambda: CommunicationNoise(probability=self.strength),
            "range_reduction": lambda: RangeReduction(scale=1.0 - self.strength),
            "faulty_robot": lambda: FaultyRobot(fraction=self.strength),
        }
        if self.name not in constructors:
            raise ValueError(f"perturbation inconnue : {self.name}")
        return constructors[self.name]()

    @property
    def identifier(self) -> str:
        # Représentation exacte : aucune collision due à un arrondi décimal.
        return f"level_{self.index:03d}_{self.strength.hex()}"


def reformation_step(distance_curve: np.ndarray, failure_step: int | None, threshold: float, stability_steps: int) -> int | None:
    """Durée après panne, jamais le pas de convergence antérieur à celle-ci.

    Précondition : les stability_steps observations juste avant la panne sont
    finies et sous le seuil. Après la panne, le premier k STRICTEMENT supérieur
    au pas de panne doit rester sous le seuil jusqu'à T, avec au moins autant
    d'observations. Formation maintenue : durée 1 ; absence de réussite préalable,
    panne absente ou suffixe trop court : None. Un NaN invalide la stabilité.
    """
    curve = np.asarray(distance_curve, dtype=float)
    if curve.ndim != 1 or stability_steps < 1 or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("paramètres de reformation invalides")
    if failure_step is None:
        return None
    if not 0 <= failure_step < len(curve):
        raise ValueError("pas de panne hors de la courbe")
    if failure_step < stability_steps:
        return None
    ok = np.isfinite(curve) & (curve <= threshold)
    if not ok[failure_step - stability_steps:failure_step].all():
        return None
    for step in range(failure_step + 1, len(curve) - stability_steps + 1):
        if ok[step:].all():
            return step - failure_step
    return None


def condition_identifier(scenario: Scenario) -> str:
    return f"n{scenario.n_robots}_failure_{float(scenario.failure_fraction).hex()}_{quote(scenario.shape_key, safe='')}"


def _json_value(value):
    """JSON strict : les métriques indéfinies restent NaN dans les NPZ, null ici."""
    if isinstance(value, dict):
        return {key: _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_value(value), indent=2, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RobustnessStudy:
    def __init__(self, config: dict, loader: ConfigLoader | None = None, output: Path | None = None,
                 factory: ControllerFactory | None = None, index: ShapeIndex | None = None) -> None:
        self.config = config
        self.loader = loader or ConfigLoader()
        self.output = Path(output if output is not None else config["output"])
        self.grid = ScenarioGrid.from_config(config)
        self.cfg_sim = self.grid.simulator_config(self.loader.load_simulator())
        self.cfg_metrics = self.loader.load_metrics()
        self.cfg_shapes = self.loader.load_shapes()
        self.suite = MetricSuite(self.cfg_metrics, self.cfg_sim, self.cfg_shapes.target_radius)
        self.index = index
        self.factory = factory
        self.levels = self._validate()

    @classmethod
    def from_config(cls, loader: ConfigLoader | None = None, config_name: str = "robustness", output: Path | None = None) -> RobustnessStudy:
        loader = loader or ConfigLoader()
        return cls(loader.load_raw(config_name), loader, output)

    def _validate(self) -> list[PerturbationLevel]:
        known = {"output", "training_root", "training_seeds", "evaluation_seed", "seed_stride", "repetitions",
                 "evaluation_horizon", "swarm_sizes", "failure_fractions", "shapes", "success_threshold",
                 "stability_steps", "controllers", "perturbations"}
        if set(self.config) != known:
            raise ValueError(f"clés de robustesse manquantes ou inconnues : {sorted(set(self.config) ^ known)}")
        if self.config["success_threshold"] != self.suite.success_threshold:
            raise ValueError("seuil de succès différent de metrics.yaml")
        stable = self.config["stability_steps"]
        if isinstance(stable, bool) or not isinstance(stable, int) or not 1 <= stable <= self.grid.horizon:
            raise ValueError("fenêtre de stabilité invalide")
        names = list(self.config["controllers"])
        if not names or any(re.fullmatch(r"[a-z][a-z0-9_]*", name) is None for name in names):
            raise ValueError("noms de contrôleurs invalides")
        for settings in self.config["controllers"].values():
            if settings.get("type") not in ("baseline", "trained") or set(settings) - {"type", "variant", "training_root"}:
                raise ValueError("configuration de contrôleur invalide")
            if settings["type"] == "trained" and not settings.get("variant"):
                raise ValueError("variante de checkpoint manquante")
        for values in (self.grid.swarm_sizes, self.grid.failure_fractions, self.grid.shapes):
            if len(values) != len(set(values)):
                raise ValueError("conditions de scénario dupliquées")
        levels = []
        for name, settings in self.config["perturbations"].items():
            if set(settings) != {"strengths"}:
                raise ValueError("seule la clé strengths est autorisée pour une perturbation")
            strengths = [float(value) for value in settings["strengths"]]
            if not strengths or strengths[0] != 0 or any(b <= a for a, b in zip(strengths, strengths[1:])):
                raise ValueError("niveaux croissants exigés, avec identité au niveau zéro")
            for level_index, strength in enumerate(strengths):
                level = PerturbationLevel(name, strength, level_index)
                level.build()
                levels.append(level)
        if not levels:
            raise ValueError("aucune perturbation configurée")
        return levels

    def _factories(self) -> dict[str, ControllerFactory]:
        # Adaptation locale à training_root, sans modifier le contrat de Factory.
        return {
            name: self.factory if self.factory is not None else ControllerFactory(
                self.loader, self.config,
                training_dir=Path(settings.get("training_root", self.config["training_root"])),
            )
            for name, settings in self.config["controllers"].items()
        }

    def _metadata(self, factories: dict, versions: dict) -> tuple[dict, dict, str]:
        snapshot = {
            "robustness": {**self.config, "output": str(self.output.resolve())},
            "simulator": asdict(self.cfg_sim),
            "metrics": asdict(self.cfg_metrics),
            "shapes": asdict(self.cfg_shapes),
            "baselines": self.loader.load_raw("baselines"),
            "training_runs": {},
            "protocol": {
                "communication_noise": "symmetric_edge_dropout",
                "position_noise": "active_local_observations_only_true_radius_graph",
                "range_reduction": "radius_times_one_minus_strength",
                "faulty_robot": "independent_mid_episode_withdrawals_union_with_ordinary_failures",
                "failure_count": "round_fraction_times_swarm_size",
                "reformation": "pre_failure_window_then_post_failure_stable_suffix_duration",
                "final_graph": "diagnostic_sample_without_controller_command",
                "breakpoint": "first_tested_strength_with_seed_mean_success_strictly_below_0.5_no_interpolation",
                "ci95_method": CI95_METHOD,
                "undefined_metrics": "nan_in_npz_null_in_json_excluded_from_metric_statistics",
            },
        }
        for name, factory in factories.items():
            if self.config["controllers"][name]["type"] != "trained":
                continue
            snapshot["training_runs"][name] = {}
            for seed in versions[name]:
                checkpoint = factory.run_directory(name, seed) / "best.pt"
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                snapshot["training_runs"][name][seed] = {
                    "config": factory.snapshot(name, seed), "checkpoint": str(checkpoint.resolve()),
                    "checkpoint_sha256": digest,
                }
        episode_seeds = [self.grid.episode_seed(rep) for rep in range(self.grid.repetitions)]
        streams = [f"perturbations/{name}/{hook}" for name in self.config["perturbations"]
                   for hook in ("observe", "graph", "commands", "failures")]
        seeds = {"training_seeds": versions, "episode_seeds": episode_seeds,
                 "root_seed": "episode_seed", "stream_keys": ["initial_positions", "failures", *streams],
                 "derivation": "SeedSequence(entropy=episode_seed,spawn_key=(crc32(utf8_key),))"}
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True)
        root = Path(__file__).resolve().parents[2]
        sources = {}
        for directory in (root / "src", self.loader.config_dir.resolve()):
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix in (".py", ".yaml"):
                    sources[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        snapshot["source_sha256"] = sources
        version = (f"commit={commit}\ndirty={bool(dirty.strip())}\npython={platform.python_version()}\n"
                   f"numpy={np.__version__}\ntorch={torch.__version__}\npyyaml={yaml.__version__}\n")
        return snapshot, seeds, version

    def run(self) -> dict:
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"dossier de robustesse déjà occupé : {self.output}")
        factories = self._factories()
        versions = {name: factory.versions(name) for name, factory in factories.items()}
        # Préflight complet AVANT toute création de sortie ; aucun entraînement.
        prepared = {(name, seed): factories[name].build(name, seed)
                    for name, seeds in versions.items() for seed in seeds}
        self.index = self.index if self.index is not None else ShapeIndex.load(Path("outputs/shapes"))
        scenarios = self.grid.scenarios()
        missions = {shape: Mission(shape_id=self.index.shape_id(shape), target=self.index.point_cloud_by_key(shape))
                    for shape in self.grid.shapes}
        snapshot, seeds, version = self._metadata(factories, versions)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "effective_config.yaml").write_text(yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8")
        _write_json(self.output / "seeds.json", seeds)
        (self.output / "version.txt").write_text(version, encoding="utf-8")
        _write_json(self.output / "summary.json", {"status": "running"})
        rows = []
        try:
            for level in self.levels:
                for (name, seed), controller in prepared.items():
                    for scenario in scenarios:
                        mission = missions[scenario.shape_key]
                        cfg = replace(self.cfg_sim, failure_fraction=scenario.failure_fraction)
                        controller.reset()
                        result = Episode(cfg, level.build()).execute(mission, controller, scenario.n_robots, scenario.episode_seed)
                        values = self.suite.compute(result.positions, result.active, mission.target).values
                        components = np.array([CommunicationGraph.connected_components(graph, active)
                                               for graph, active in zip(result.adjacency, result.active)], dtype=int)
                        elapsed = reformation_step(values["distance_curve"], result.failure_step,
                                                   self.suite.success_threshold, self.config["stability_steps"])
                        directory = self.output / level.name / level.identifier / name / (f"seed_{seed}" if seed is not None else "baseline") / condition_identifier(scenario)
                        directory.mkdir(parents=True, exist_ok=True)
                        archive = directory / f"episode_{scenario.episode_seed}.npz"
                        np.savez_compressed(
                            archive, positions=result.positions, active=result.active, commands=result.commands,
                            observed_positions=result.observed_positions, adjacency=result.adjacency,
                            target=mission.target, target_radius=self.suite.target_radius,
                            communication_radius=result.communication_radius, distance_curve=values["distance_curve"],
                            components_by_step=components, failure_step=-1 if result.failure_step is None else result.failure_step,
                            failure_indices=np.asarray(result.failure_indices, dtype=int),
                            episode_seed=scenario.episode_seed, training_seed=-1 if seed is None else seed,
                            shape_id=mission.shape_id, dt=cfg.dt, v_max=cfg.v_max, base_communication_radius=cfg.r_c,
                            strength=level.strength, failure_fraction=scenario.failure_fraction,
                            success_threshold=self.suite.success_threshold, stability_steps=self.config["stability_steps"],
                            reformation_step=-1 if elapsed is None else elapsed,
                        )
                        rows.append({
                            "perturbation": level.name, "strength": level.strength, "controller": name,
                            "training_seed": seed, "episode_seed": scenario.episode_seed,
                            "n_robots": scenario.n_robots, "failure_fraction": scenario.failure_fraction,
                            "shape": scenario.shape_key, "shape_id": mission.shape_id,
                            "final_distance": values["final_distance"], "success": bool(values["success"]),
                            "success_threshold": self.suite.success_threshold, "convergence_step": values["convergence_step"],
                            "collision_rate": values["collision_rate"], "normalized_regularity": values["normalized_regularity"],
                            "final_components": int(components[-1]), "mean_components": float(components.mean()),
                            "final_active_fraction": float(result.active[-1].mean()),
                            "failure_step": result.failure_step, "failure_count": len(result.failure_indices),
                            "reformation_step": elapsed, "step": result.step,
                            "communication_radius": result.communication_radius,
                            "archive": archive.relative_to(self.output).as_posix(),
                        })
            _write_csv(self.output / "episodes.csv", rows)
            expected = len(self.levels) * len(scenarios) * len(prepared)
            if len(rows) != expected:
                raise ValueError("nombre d'épisodes de robustesse inattendu")
            summary = self.build_report(rows)
            summary.update(status="complete", rows=len(rows), expected_rows=expected)
            _write_json(self.output / "summary.json", summary)
            return summary
        except Exception as error:
            _write_csv(self.output / "episodes.csv", rows)
            _write_json(self.output / "summary.json", {"status": "failed", "rows": len(rows), "error": str(error)})
            raise

    def build_report(self, rows: list[dict]) -> dict:
        metrics = ("final_distance", "success", "collision_rate", "normalized_regularity", "final_components",
                   "mean_components", "final_active_fraction", "reformation_step")
        groups = defaultdict(list)
        for row in rows:
            key = (row["perturbation"], row["strength"], row["controller"], row["n_robots"], row["failure_fraction"], row["shape"], row["training_seed"])
            groups[key].append(row)
        within, across_groups = [], defaultdict(list)
        labels = ("perturbation", "strength", "controller", "n_robots", "failure_fraction", "shape")
        for (*condition, seed), episodes in groups.items():
            for metric in metrics:
                values = [float(row[metric]) for row in episodes if row[metric] is not None and np.isfinite(row[metric])]
                stats = ResultsTable.statistics(values) if values else {"mean": None, "std": None, "ci95": None, "n": 0}
                aggregate = {**dict(zip(labels, condition)), "training_seed": seed, "metric": metric,
                             **stats, "episode_count": len(episodes), "success_threshold": self.suite.success_threshold}
                within.append(aggregate)
                across_groups[(*condition, metric)].append(aggregate)
        between = []
        for (*condition, metric), aggregates in across_groups.items():
            means = [row["mean"] for row in aggregates if row["mean"] is not None]
            stats = ResultsTable.statistics(means) if means else {"mean": None, "std": None, "ci95": None, "n": 0}
            trained = aggregates[0]["training_seed"] is not None
            between.append({**dict(zip(labels, condition)), "metric": metric, **stats,
                            "variation_level": "between_training_seeds" if trained else "baseline_episode_mean",
                            "success_threshold": self.suite.success_threshold})
        break_groups = defaultdict(list)
        for aggregate in between:
            if aggregate["metric"] == "success":
                key = tuple(aggregate[label] for label in labels if label != "strength")
                break_groups[key].append(aggregate)
        breakpoints = []
        for key, aggregates in break_groups.items():
            ordered = sorted(aggregates, key=lambda row: row["strength"])
            first = next((row["strength"] for row in ordered if row["mean"] is not None and row["mean"] < 0.5), None)
            breakpoints.append({**dict(zip((label for label in labels if label != "strength"), key)),
                                "first_tested_strength_below_half": first, "success_threshold": self.suite.success_threshold})
        self.output.mkdir(parents=True, exist_ok=True)
        _write_csv(self.output / "within_seed.csv", within)
        _write_csv(self.output / "between_seeds.csv", between)
        _write_csv(self.output / "breakpoints.csv", breakpoints)
        self._plot(between, within)
        return {"within_seed": within, "between_seeds": between, "breakpoints": breakpoints,
                "success_threshold": self.suite.success_threshold,
                "ci95_method": CI95_METHOD}

    def _plot(self, between: list[dict], within: list[dict]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directory = self.output / "figures"
        directory.mkdir(parents=True, exist_ok=True)
        conditions = sorted({(row["perturbation"], row["n_robots"], row["failure_fraction"], row["shape"]) for row in between})
        for perturbation, size, fraction, shape in conditions:
            figure, axes = plt.subplots(1, 2, figsize=(10, 4))
            for axis, metric in zip(axes, ("final_distance", "success")):
                selected = [row for row in between if (row["perturbation"], row["n_robots"], row["failure_fraction"], row["shape"], row["metric"]) == (perturbation, size, fraction, shape, metric)]
                for name in sorted({row["controller"] for row in selected}):
                    values = sorted((row for row in selected if row["controller"] == name and row["mean"] is not None), key=lambda row: row["strength"])
                    if not values:
                        continue
                    x = np.array([row["strength"] for row in values])
                    y = np.array([row["mean"] for row in values])
                    trained = values[0]["variation_level"] == "between_training_seeds"
                    if trained:
                        band = np.array([row["std"] for row in values])
                    else:
                        band = np.array([next(row["std"] for row in within if all(row[key] == value[key] for key in ("perturbation", "strength", "controller", "n_robots", "failure_fraction", "shape", "metric"))) for value in values])
                    axis.plot(x, y, marker="o", label=f"{name} ({'graines' if trained else 'épisodes'})")
                    axis.fill_between(x, y - band, y + band, alpha=0.15)
                axis.set_xlabel("Intensité de perturbation")
                axis.set_ylabel("Distance finale" if metric == "final_distance" else f"Succès (seuil {self.suite.success_threshold:g})")
                if metric == "success":
                    axis.axhline(0.5, color="black", linestyle=":")
                    axis.set_ylim(-0.05, 1.05)
                else:
                    axis.axhline(self.suite.success_threshold, color="black", linestyle=":")
                if axis.lines:
                    axis.legend(fontsize="small")
            figure.suptitle(f"{perturbation} — {shape}, N={size}, panne={fraction:g} ; bandes ±1 écart-type")
            figure.tight_layout()
            identifier = condition_identifier(Scenario(size, fraction, shape, 0))
            figure.savefig(directory / f"{perturbation}_{identifier}.png", dpi=150)
            plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="robustness")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    RobustnessStudy.from_config(config_name=args.config, output=args.output).run()


if __name__ == "__main__":
    main()
