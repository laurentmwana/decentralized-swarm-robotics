"""Phases 10–11 : orchestration de campagnes gelées et rapports depuis les archives."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.common.config import ConfigLoader
from src.evaluation.builders import TableBuilder
from src.evaluation.campaign import Campaign
from src.evaluation.evaluator import Evaluator
from src.evaluation.factory import ControllerFactory
from src.evaluation.results import AGGREGATABLE_METRICS, ResultsTable
from src.shapes.index import ShapeIndex, build_split


def write_records(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    """Écrit un tableau, y compris vide avec un schéma explicite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def archive_path(source: Path, row: dict) -> Path:
    condition = (f"n{row['n_robots']}_failure{round(row['failure_fraction'] * 100):02d}_"
                 f"{row['shape']}")
    directory = source / row["controller"]
    if row["training_seed"] is not None:
        directory /= f"seed_{row['training_seed']}"
    return directory / "positions" / condition / f"seed_{row['episode_seed']}.npz"


def require_empty(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"dossier de sortie déjà occupé : {path}")


def effective_targets(snapshot: dict) -> tuple[list[str], list[str]]:
    """Lit les cibles réellement utilisées, jamais le split théorique du checkpoint."""
    if "curriculum" in snapshot:
        section = snapshot["curriculum"]
        training = section.get("training_targets")
        validation = section.get("validation_targets")
    else:
        section = snapshot.get("training", {})
        training = [section["training_target"]] if "training_target" in section else None
        validation = [section["validation_target"]] if "validation_target" in section else None
    for targets in (training, validation):
        if not isinstance(targets, list) or not targets or any(
            not isinstance(key, str) or "_" not in key or "@" in key for key in targets
        ) or len(targets) != len(set(targets)):
            raise ValueError("listes effectives training/validation absentes ou invalides dans le snapshot")
    return list(training), list(validation)


class ShapeStudyReport:
    """Deux niveaux de variabilité ; aucun épisode ne devient une graine d'entraînement."""

    @staticmethod
    def grouped(table: ResultsTable, exposure: dict[str, str], by_type: bool = True) -> tuple[list[dict], list[dict]]:
        groups = defaultdict(list)
        for row in table.rows:
            shape_type = row["shape"].split("_", 1)[0] if by_type else "all"
            key = (row["controller"], row["training_seed"], row["n_robots"],
                   row["failure_fraction"], shape_type, exposure[row["shape"]])
            groups[key].append(row)
        within, between_groups = [], defaultdict(list)
        for (controller, seed, size, failure, kind, group), rows in sorted(groups.items(), key=lambda item: str(item[0])):
            identity = {"controller": controller, "training_seed": seed, "n_robots": size,
                        "failure_fraction": failure, "shape_type": kind, "exposure": group}
            for metric in AGGREGATABLE_METRICS:
                episode_groups = defaultdict(list)
                for row in rows:
                    episode_groups[row["episode_seed"]].append(float(row[metric]))
                stats = ResultsTable.statistics([float(np.mean(values)) for values in episode_groups.values()])
                if stats["n"] == 1:
                    stats["ci95"] = None
                record = {**identity, "metric": metric, "shape_count": len({row["shape"] for row in rows}), **stats}
                within.append(record)
                between_groups[(controller, size, failure, kind, group, metric)].append(record)
        between = []
        for (controller, size, failure, kind, group, metric), records in between_groups.items():
            trained = records[0]["training_seed"] is not None
            stats = ResultsTable.statistics([record["mean"] for record in records]) if trained else {
                key: records[0][key] for key in ("mean", "std", "ci95", "n")}
            if trained and stats["n"] == 1:
                stats["ci95"] = None
            between.append({"controller": controller, "n_robots": size, "failure_fraction": failure,
                            "shape_type": kind, "exposure": group, "metric": metric,
                            "uncertainty_unit": "training_seed" if trained else "episode", **stats})
        return within, between

    @staticmethod
    def fixed_examples(table: ResultsTable, config: dict) -> list[dict]:
        groups = defaultdict(list)
        for row in table.rows:
            if (row["training_seed"] in (None, config["training_seed"])
                    and row["episode_seed"] in config["episode_seeds"]
                    and row["failure_fraction"] == config["failure_fraction"]):
                groups[(row["controller"], row["n_robots"], row["shape"])].append(row)
        expected = {(row["controller"], row["n_robots"], row["shape"]) for row in table.rows}
        if set(groups) != expected or any(len(rows) < 2 for rows in groups.values()):
            raise ValueError("exemples fixes incomplets : au moins deux épisodes par forme et contrôleur")
        examples = []
        for key in sorted(groups):
            ordered = sorted(groups[key], key=lambda row: (row["final_distance"], row["episode_seed"]))
            for label, row in (("favorable", ordered[0]), ("unfavorable", ordered[-1])):
                examples.append({**row, "example_kind": label})
        return examples

    @staticmethod
    def plot_examples(source: Path, examples: list[dict], columns: int) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Les extrêmes sont relatifs au sous-ensemble fixé, pas une preuve de lisibilité.
        for offset in range(0, len(examples), columns * 2):
            page = examples[offset:offset + columns * 2]
            figure, axes = plt.subplots(2, columns, figsize=(3 * columns, 6), squeeze=False)
            for axis, row in zip(axes.flat, page):
                with np.load(archive_path(source, row), allow_pickle=False) as data:
                    final = data["positions"][-1][data["active"][-1].astype(bool)]
                    target = data["target"]
                axis.scatter(target[:, 0], target[:, 1], s=12, marker="x", label="cible")
                axis.scatter(final[:, 0], final[:, 1], s=12, label="essaim")
                axis.set_aspect("equal")
                axis.set_title(f"{row['shape']} / {row['controller']}\n{row['example_kind']}, "
                               f"graine {row['episode_seed']}, d={row['final_distance']:.3f}", fontsize=8)
            for axis in list(axes.flat)[len(page):]:
                axis.set_visible(False)
            figure.tight_layout()
            figure.savefig(source / "figures" / f"shape_examples_{offset // (columns * 2):03d}.png")
            plt.close(figure)


class _StudyCampaign(Campaign):
    """Conserve le moteur et ses snapshots, remplace seulement les rapports Markdown."""

    def build_reports(self, table: ResultsTable, names: list[str], versions: dict) -> dict:
        TableBuilder(self.output).write_aggregates(table, AGGREGATABLE_METRICS)
        return {"success_threshold": self.evaluator.suite.success_threshold}


class _FrozenFactory:
    """Prépare tous les poids avant écriture et conserve les snapshots vérifiés."""

    def __init__(self, factory: ControllerFactory, names: list[str]) -> None:
        self.factory = factory
        self.version_map = {name: factory.versions(name) for name in names}
        self.snapshots = {}
        self.digests = {}
        self.controllers = {}
        for name, seeds in self.version_map.items():
            for seed in seeds:
                if seed is not None:
                    self.snapshots[(name, seed)] = copy.deepcopy(factory.snapshot(name, seed))
                    self.digests[(name, seed)] = self._digest(name, seed)
                self.controllers[(name, seed)] = factory.build(name, seed)
        self.verify()

    def _digest(self, name: str, seed: int) -> str:
        with (self.run_directory(name, seed) / "best.pt").open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def verify(self) -> None:
        for (name, seed), digest in self.digests.items():
            if self._digest(name, seed) != digest or self.factory.snapshot(name, seed) != self.snapshots[(name, seed)]:
                raise ValueError("checkpoint ou snapshot modifié pendant l'évaluation indépendante")

    def versions(self, name: str) -> list[int | None]:
        return self.version_map[name]

    def run_directory(self, name: str, seed: int) -> Path:
        return self.factory.run_directory(name, seed)

    def snapshot(self, name: str, seed: int) -> dict:
        return copy.deepcopy(self.snapshots[(name, seed)])

    def build(self, name: str, seed: int | None):
        return self.controllers[(name, seed)]


class ComplexShapeStudy:
    def __init__(self, config: dict, loader: ConfigLoader | None = None) -> None:
        self.config = copy.deepcopy(config)
        self.loader = loader or ConfigLoader()
        self.output = Path(config["output"]).resolve()

    def _selection(self, index: ShapeIndex) -> dict[str, list[str]]:
        cfg = self.loader.load_shapes()
        split = build_split(cfg)
        expected_ids = [f"{kind}_{name}" for kind in ("geometries", "digits", "letters", "numbers")
                        for name in getattr(cfg, kind)]
        if index.identifiers != expected_ids:
            raise ValueError("identifiants de formes différents de l'ordre stable configuré")
        request = self.config["target_selection"]
        if (not request["types"] or len(request["types"]) != len(set(request["types"]))
                or not set(request["types"]) <= {"geometries", "digits", "letters", "numbers"}
                or not request["seen_splits"] or len(request["seen_splits"]) != len(set(request["seen_splits"]))
                or not set(request["seen_splits"]) <= {"train", "val"}):
            raise ValueError("types ou splits vus invalides ; test réservé au mode heldout")
        selection = {part: [] for part in ("train", "val", "test")}
        indexed = {}
        for row in index.rows:
            key = f"{row['type']}_{row['name']}"
            if key in indexed or split.mapping.get(key) != row["split"] or index.shape_id(key) != row["shape_id"]:
                raise ValueError("index des formes incompatible avec le split figé")
            indexed[key] = row
        for key, part in split.mapping.items():
            if key.split("_", 1)[0] in self.config["target_selection"]["types"]:
                if key not in indexed:
                    raise ValueError(f"cible absente de l'index : {key}")
                cloud = index.point_cloud_by_key(key)
                if (cloud.shape != (cfg.reference_points, 2) or not np.isfinite(cloud).all()
                        or len(np.unique(cloud, axis=0)) < max(self.config["evaluation"]["swarm_sizes"])
                        or indexed[key]["n_points"] != len(cloud)):
                    raise ValueError(f"cible insuffisante, non finie ou dupliquée : {key}")
                selection[part].append(key)
        if any(not selection[part] for part in selection):
            raise ValueError("sélection de formes vide dans un split")
        return selection

    def _evaluation(self, selection: dict, mode: str) -> tuple[dict, dict[str, str]]:
        evaluation = copy.deepcopy(self.config["evaluation"])
        seen = [key for part in self.config["target_selection"]["seen_splits"] for key in selection[part]]
        if mode == "compare":
            comparison = self.config["comparison"]
            evaluation["controllers"].update(copy.deepcopy(comparison["controllers"]))
            shapes = comparison["shapes"]
            if not set(shapes) <= set(seen):
                raise ValueError("comparaison spécialisée limitée aux formes train/val configurées")
            seen = shapes
        shapes = seen + selection["test"] if mode == "heldout" else seen
        if not shapes or len(shapes) != len(set(shapes)):
            raise ValueError("sélection des formes vide ou dupliquée")
        evaluation.update(shapes=shapes, shape_split=selection, seen_shapes=seen,
                          heldout_shapes=selection["test"] if mode == "heldout" else [])
        return evaluation, {key: "heldout" if key in selection["test"] else "seen" for key in shapes}

    def _validate_snapshots(self, frozen: _FrozenFactory, evaluation: dict, exposure: dict, heldout: bool) -> list[dict]:
        records = []
        episode_seeds = set(range(evaluation["evaluation_seed"] * evaluation["seed_stride"],
                                  evaluation["evaluation_seed"] * evaluation["seed_stride"] + evaluation["repetitions"]))
        if episode_seeds & set(evaluation["training_seeds"]):
            raise ValueError("graines d'évaluation et d'entraînement non disjointes")
        for (name, seed), snapshot in frozen.snapshots.items():
            settings = evaluation["controllers"][name]
            expected = self.config["expected_training"]
            profile = snapshot["training"]["profiles"][snapshot["profile"]]
            if snapshot["profile"] != expected["profile"] or profile["n_robots"] != settings.get("expected_n_robots", expected["n_robots"]):
                raise ValueError("profil ou taille d'entraînement incompatible avec l'étude")
            model = snapshot["model"]
            shapes_cfg = self.loader.load_shapes()
            identifiers = [f"{kind}_{label}" for kind in ("geometries", "digits", "letters", "numbers")
                           for label in getattr(shapes_cfg, kind)]
            if (model["hidden_dim"] != expected["hidden_dim"] or model["message_rounds"] != expected["message_rounds"]
                    or model["use_target_geometry"] != settings["expected_use_target_geometry"]
                    or model["n_shapes"] <= max(identifiers.index(key) for key in exposure)
                    or not snapshot["communication"]):
                raise ValueError("architecture ou information du checkpoint incompatible")
            training, validation = effective_targets(snapshot)
            used = set(training + validation)
            if heldout:
                leaked = [key for key, group in exposure.items() if group == "heldout" and key in used]
                unseen = [key for key, group in exposure.items() if group == "seen" and key not in used]
                if leaked:
                    raise ValueError(f"fuite de cibles test dans training/validation : {name}/{seed} : {leaked}")
                if unseen:
                    raise ValueError(f"formes annoncées vues mais absentes du snapshot : {name}/{seed} : {unseen}")
            records.append({"controller": name, "training_seed": seed, "training_targets": training,
                            "validation_targets": validation, "checkpoint_sha256": frozen.digests[(name, seed)],
                            "information": "nearest_unit_direction_and_shape_id" if model["use_target_geometry"] else "shape_id_only"})
        return records

    def run(self, mode: str = "complex") -> dict:
        if mode not in ("complex", "heldout", "compare"):
            raise ValueError(f"mode d'étude inconnu : {mode}")
        output = Path(self.config["comparison"]["output"]).resolve() if mode == "compare" else self.output
        require_empty(output)
        index = ShapeIndex.load(Path("outputs/shapes"))
        selection = self._selection(index)
        evaluation, exposure = self._evaluation(selection, mode)
        factory = ControllerFactory(self.loader, evaluation, Path(evaluation["training_root"]))
        for name, settings in evaluation["controllers"].items():
            if settings["type"] == "trained":
                for seed in evaluation["training_seeds"]:
                    expected_path = Path(settings.get("training_root", evaluation["training_root"])) / settings["variant"] / f"seed_{seed}"
                    if factory.run_directory(name, seed).resolve() != expected_path.resolve():
                        raise ValueError("ControllerFactory.run_directory/versions doivent respecter settings['training_root']")
        frozen = _FrozenFactory(factory, list(evaluation["controllers"]))
        provenance = self._validate_snapshots(frozen, evaluation, exposure, mode == "heldout")
        evaluator = Evaluator.from_config(self.loader, output, evaluation)
        evaluator.index = index
        examples_cfg = self.config["examples"]
        episode_seeds = [evaluator.grid.episode_seed(rep) for rep in range(evaluator.grid.repetitions)]
        if (evaluation["positions_per_condition"] < evaluation["repetitions"]
                or len(set(examples_cfg["episode_seeds"])) < 2
                or not set(examples_cfg["episode_seeds"]) <= set(episode_seeds)
                or examples_cfg["training_seed"] not in evaluation["training_seeds"]
                or examples_cfg["failure_fraction"] not in evaluation["failure_fractions"]
                or examples_cfg["columns"] < 1):
            raise ValueError("archives complètes et exemples fixes valides requis")
        frozen.verify()
        campaign = _StudyCampaign(evaluator, frozen, evaluation, output, self.loader)
        summary = campaign.run()
        frozen.verify()
        table = ResultsTable.read_csv(output / "episodes.csv")
        within, between = ShapeStudyReport.grouped(table, exposure)
        write_records(output / "by_type_per_seed.csv", within)
        write_records(output / "by_type_between_seeds.csv", between)
        _, seen_heldout = ShapeStudyReport.grouped(table, exposure, by_type=False)
        write_records(output / "seen_vs_heldout.csv", seen_heldout)
        failures = [{"controller": name, "training_seed": seed, "n_robots": size,
                     "failure_fraction": failure, "shape": shape, "failure_rate": 1 - stats["mean"],
                     "episode_count": stats["n"], "exposure": exposure[shape]}
                    for (name, seed, size, failure, shape), stats in table.aggregates("success").items()
                    if stats["mean"] < 1]
        write_records(output / "failing_shapes.csv", failures,
                      ["controller", "training_seed", "n_robots", "failure_fraction", "shape", "failure_rate", "episode_count", "exposure"])
        examples = ShapeStudyReport.fixed_examples(table, examples_cfg)
        for row in examples:
            if not archive_path(output, row).is_file():
                raise FileNotFoundError(f"archive requise : {archive_path(output, row)}")
        write_records(output / "examples.csv", examples)
        (output / "figures").mkdir(exist_ok=True)
        ShapeStudyReport.plot_examples(output, examples, examples_cfg["columns"])
        summary.update(mode=mode, by_type=between, seen_vs_heldout=seen_heldout, provenance=provenance,
                       shape_split=selection, readability_status="pending", information_equivalent=False)
        (output / "study_config.yaml").write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        (output / "limitations.txt").write_text(
            "ID seul et géométrie ne reçoivent pas une information équivalente ; aucun gain n'isole leur effet causal.\n"
            "La règle manuelle reçoit un déplacement au contour ; le GNN géométrique une direction unitaire.\n"
            "Un embedding ID jamais entraîné est une limitation explicite, pas une preuve d'apprentissage zéro-shot.\n"
            "Les poids sont gelés. En mode heldout, les cibles test sont exclues des listes effectives training/validation de tous les snapshots.\n"
            "IC 95 % : approximation normale entre graines pour les modèles ; entre épisodes pour la règle. IC absent si n=1.\n"
            "Exemples : minima et maxima de distance parmi les épisodes fixés, sans exclusion des échecs ; égalités possibles.\n"
            "La distance géométrique ne mesure pas la lisibilité. Concavités, traits fins et formes proches sont des hypothèses, pas des causes établies.\n",
            encoding="utf-8")
        return summary


class HeldOutEvaluator(ComplexShapeStudy):
    """Réserve le test et vérifie tous les snapshots, y compris les validations."""

    def run(self, mode: str = "heldout") -> dict:
        if mode != "heldout":
            raise ValueError("HeldOutEvaluator accepte uniquement le mode heldout")
        return super().run(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="complex-shapes")
    parser.add_argument("--mode", choices=("complex", "heldout", "compare"), default="complex")
    args = parser.parse_args()
    path = Path(args.config)
    loader = ConfigLoader(path.resolve().parent) if path.suffix in (".yaml", ".yml") else ConfigLoader()
    name = path.stem if path.suffix else args.config
    study_class = HeldOutEvaluator if args.mode == "heldout" else ComplexShapeStudy
    study_class(loader.load_raw(name), loader).run(args.mode)


if __name__ == "__main__":
    main()
