"""Rapport du curriculum sauvegardé, sans chargement de modèle ni nouvel épisode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from src.common.config import ConfigLoader
from src.evaluation.results import ResultsTable

LOSS_TERMS = ("shape", "coverage", "collisions", "motion")
ROW_KEYS = (
    "level", "target", "shape_id", "validation_seed_count", "total_loss",
    "final_distance", *LOSS_TERMS, "previous_distance", "forgetting", "notable_forgetting",
)
# Tolérances de lecture des moyennes issues de calculs en virgule flottante.
REL_TOL = 1e-5
ABS_TOL = 1e-7
CAMPAIGN_FILES = (
    "effective_config.yaml", "seeds.json", "version.txt", "status.json",
    "summary.json", "losses_by_target.csv",
)
LEVEL_FILES = (
    "effective_config.yaml", "seeds.json", "version.txt", "status.json",
    "summary.json", "targets.json", "best.pt", "last.pt", "losses.csv", "validation.csv",
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if path.suffix == ".yaml" else json.loads(text)
    _check(isinstance(value, dict), f"dictionnaire attendu : {path}")
    return value


def _required(directory: Path, names: tuple[str, ...]) -> None:
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"artefact manquant : {path}")
        _check(path.stat().st_size > 0, f"artefact vide : {path}")


def _integer(value, label: str) -> int:
    _check(not isinstance(value, bool), f"entier invalide : {label}")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"entier invalide : {label}") from error
    _check(str(result) == str(value), f"entier invalide : {label}")
    return result


def _number(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"nombre invalide : {label}") from error
    _check(not isinstance(value, bool) and math.isfinite(result), f"nombre non fini : {label}")
    return result


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def _identity(value: dict, variant: str, seed: int, profile: str, origin: Path,
              *, summary: bool = False) -> None:
    expected = {"variant": variant, "seed": seed, "profile": profile, "status": "completed"}
    # Les résumés de niveaux portent seulement la graine, contrairement aux statuts.
    for key, wanted in expected.items():
        if summary and key != "seed" and key not in value:
            continue
        _check(value.get(key) == wanted, f"identité ou statut incohérent ({key}) : {origin}")


def _stats(values: list[float]) -> dict:
    stats = ResultsTable.statistics(values)
    _check(all(math.isfinite(stats[key]) for key in ("mean", "std", "ci95")),
           "statistiques non finies : valeurs numériques trop grandes")
    return {"mean": stats["mean"], "std_sample": stats["std"] if stats["n"] > 1 else None,
            "ci95": stats["ci95"] if stats["n"] > 1 else None, "n": stats["n"]}


def _write_csv(path: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class CurriculumReport:
    """Vérifie tous les runs en lecture seule avant de régénérer les sorties racine."""

    def __init__(self, root: Path | str, profile: str = "standard") -> None:
        self.root = Path(root).resolve()
        self.profile = profile

    @classmethod
    def from_config(cls, config: str = "curriculum", profile: str = "standard",
                    output: Path | None = None, loader: ConfigLoader | None = None) -> "CurriculumReport":
        # La configuration courante sert uniquement à localiser la campagne.
        raw = (loader or ConfigLoader()).load_raw(config)
        return cls(output if output is not None else raw["output"], profile)

    def _checkpoint_path(self, recorded: str, level: int, variant: str, seed: int, raw: dict) -> None:
        relative = Path(f"level_{level}") / variant / f"seed_{seed}" / "best.pt"
        actual = self.root / relative
        original = Path(raw["output"]) / relative
        path = Path(str(recorded).replace("\\", "/"))
        _check(path.resolve() in (actual.resolve(), original.resolve()),
               f"provenance du meilleur checkpoint incohérente : {recorded}")

    @staticmethod
    def _targets(raw: dict, shapes: dict) -> list[tuple[list[str], list[str]]]:
        training, validation, used = [], [], set()
        result = []
        for level in raw["levels"]:
            kinds = level["shape_types"]
            _check(kinds and len(set(kinds)) == len(kinds) and not used.intersection(kinds),
                   "types de formes répétés ou absents du curriculum")
            for kind in kinds:
                parts = shapes.get("splits", {}).get(kind, {})
                _check(parts.get("train") and parts.get("val"), f"découpage sauvegardé incomplet : {kind}")
                training.extend(f"{kind}_{name}@{raw['resolution']}" for name in parts["train"])
                validation.extend(f"{kind}_{name}@{raw['resolution']}" for name in parts["val"])
            used.update(kinds)
            _check(len(set(training + validation)) == len(training + validation),
                   "cibles sauvegardées dupliquées ou splits communs")
            result.append((training.copy(), validation.copy()))
        return result

    @staticmethod
    def _comparable(saved: dict, raw: dict, *, across_variants: bool = False) -> dict:
        result = deepcopy(saved)
        result["model"].pop("init_seed", None)
        result.pop("initial_checkpoint", None)
        if across_variants:
            result.pop("variant", None)
            result.pop("override", None)
            result.pop("communication", None)
            result.pop("curriculum", None)
            result["training"].pop("training_target", None)
            result["training"].pop("validation_target", None)
            for override in raw["variants"].values():
                for key in override:
                    result["model"].pop(key, None)
        return result

    def _level(self, raw: dict, variant: str, seed: int, number: int, budget: dict,
               shape_ids: dict[str, int]) -> tuple[dict, dict, list[str], dict]:
        directory = self.root / f"level_{number}" / variant / f"seed_{seed}"
        _required(directory, LEVEL_FILES)
        saved = _mapping(directory / "effective_config.yaml")
        summary = _mapping(directory / "summary.json")
        _identity(_mapping(directory / "status.json"), variant, seed, self.profile, directory)
        _identity(summary, variant, seed, self.profile, directory, summary=True)
        _check(saved.get("variant") == variant and saved.get("profile") == self.profile,
               f"identité de configuration incohérente : {directory}")
        for section in ("model", "training", "shapes", "simulator", "metrics", "curriculum"):
            _check(isinstance(saved.get(section), dict), f"provenance absente ({section}) : {directory}")
        override = raw["variants"][variant]
        _check(saved.get("override") == override and saved.get("communication") == bool(override.get("communication", True)),
               f"surcharge de variante incohérente : {directory}")
        _check(saved["model"].get("init_seed") == seed, f"graine du modèle incohérente : {directory}")
        for key, value in override.items():
            if key != "communication":
                _check(saved["model"].get(key) == value, f"architecture de variante incohérente : {directory}")
        training = saved["training"]
        _check(training.get("profiles") == raw["profiles"] and training.get("variants") == raw["variants"],
               f"budgets ou variantes incohérents : {directory}")
        _check(summary.get("iterations") == budget["iterations"] and saved["simulator"].get("horizon") == budget["horizon"],
               f"budget de niveau incohérent : {directory}")
        best_iteration = _integer(summary.get("best_iteration"), "best_iteration")
        _check(1 <= best_iteration <= budget["iterations"], f"meilleure itération invalide : {directory}")
        _number(summary.get("best_validation_score"), "best_validation_score")
        provenance = _mapping(directory / "seeds.json")
        _check(provenance.get("seed") == seed and provenance.get("model_init_seed") == seed
               and provenance.get("training_seeds") == sorted(budget["seeds"])
               and provenance.get("validation_seeds") == budget["validation_seeds"]
               and provenance.get("sampler_seed") == saved["shapes"].get("sampler_seed")
               and "sampler_seed" in saved["shapes"], f"provenance des graines incohérente : {directory}")
        targets = self._targets(raw, saved["shapes"])
        train_keys, val_keys = targets[number - 1]
        context = {"config": raw, "level": number, "name": raw["levels"][number - 1]["name"],
                   "training_targets": train_keys, "validation_targets": val_keys, "optimizer_state_transfer": False}
        _check(saved["curriculum"] == context, f"contexte cumulatif incohérent : {directory}")
        _check(training.get("training_target") == train_keys[0] and training.get("validation_target") == val_keys[0],
               f"cibles principales incohérentes : {directory}")
        records = _mapping(directory / "targets.json")
        for key, expected in (("training_targets", train_keys), ("validation_targets", train_keys + val_keys)):
            entries = records.get(key)
            _check(isinstance(entries, list) and all(isinstance(row, dict) for row in entries),
                   f"provenance des cibles absente : {directory}")
            _check([row.get("key") for row in entries] == expected, f"cibles cumulatives incomplètes : {directory}")
            for row in entries:
                target = row["key"]
                shape_id = _integer(row.get("shape_id"), "shape_id")
                _check(shape_id >= 0 and row.get("split") == ("train" if target in train_keys else "val"),
                       f"identité ou split de cible incohérent : {directory}")
                _check(shape_ids.setdefault(target, shape_id) == shape_id, f"identifiant de cible variable : {target}")
        _check(records.get("training_target") == records["training_targets"][0],
               f"provenance de la cible principale incohérente : {directory}")
        if number == 1:
            _check("initial_checkpoint" not in saved, f"transfert inattendu au premier niveau : {directory}")
        else:
            checkpoint = saved.get("initial_checkpoint", {})
            _check(isinstance(checkpoint, dict) and "path" in checkpoint and "sha256" in checkpoint,
                   f"provenance du transfert absente : {directory}")
            self._checkpoint_path(checkpoint["path"], number - 1, variant, seed, raw)
            previous = self.root / f"level_{number - 1}" / variant / f"seed_{seed}" / "best.pt"
            with previous.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            _check(checkpoint["sha256"] == digest, f"empreinte du meilleur checkpoint incohérente : {directory}")
        weights = training.get("losses", {}).get("weights", {})
        _check(isinstance(weights, dict) and set(weights) == set(LOSS_TERMS), f"poids de perte absents ou inconnus : {directory}")
        weights = {key: _number(weights[key], key) for key in LOSS_TERMS}
        _check(all(value >= 0 for value in weights.values()), f"poids de perte négatifs : {directory}")
        return saved, summary, train_keys + val_keys, weights

    def _rows(self, directory: Path, variant: str, seed: int, expected: dict[int, list[str]],
              weights: dict[int, dict], shape_ids: dict[str, int], budget: dict, tolerance: float) -> list[dict]:
        with (directory / "losses_by_target.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            _check(reader.fieldnames is not None and len(reader.fieldnames) == len(ROW_KEYS)
                   and set(reader.fieldnames) == set(ROW_KEYS), f"colonnes de pertes invalides : {directory}")
            raw_rows = list(reader)
        rows, seen = [], set()
        for raw_row in raw_rows:
            _check(set(raw_row) == set(ROW_KEYS) and all(value is not None for value in raw_row.values()),
                   f"ligne CSV invalide : {directory}")
            row = {"variant": variant, "seed": seed, **raw_row}
            for key in ("level", "shape_id", "validation_seed_count"):
                row[key] = _integer(row[key], key)
            pair = (row["level"], row["target"])
            _check(pair not in seen, f"mesure dupliquée : {directory}, {pair}")
            seen.add(pair)
            _check(row["level"] in expected and row["target"] in expected[row["level"]],
                   f"cible ou niveau inattendu : {directory}, {pair}")
            _check(row["shape_id"] == shape_ids[row["target"]]
                   and row["validation_seed_count"] == len(budget["validation_seeds"]),
                   f"provenance de mesure incohérente : {directory}, {pair}")
            for key in ("total_loss", "final_distance", *LOSS_TERMS):
                row[key] = _number(row[key], key)
                _check(row[key] >= 0, f"mesure négative ({key}) : {directory}")
            total = sum(weights[row["level"]][key] * row[key] for key in LOSS_TERMS)
            _check(_close(row["total_loss"], total), f"total_loss incompatible avec les poids sauvegardés : {directory}, {pair}")
            for key in ("previous_distance", "forgetting"):
                row[key] = None if row[key] == "" else _number(row[key], key)
            _check(row["notable_forgetting"] in ("True", "False"), f"indicateur d'oubli invalide : {directory}")
            row["notable_forgetting"] = row["notable_forgetting"] == "True"
            rows.append(row)
        requested = {(level, target) for level, targets in expected.items() for target in targets}
        _check(seen == requested, f"mesures de niveaux ou cibles manquantes : {directory}, {sorted(requested - seen)}")
        previous = {}
        for row in sorted(rows, key=lambda value: (value["level"], value["target"])):
            target = row["target"]
            if target not in previous:
                _check(row["previous_distance"] is None and row["forgetting"] is None
                       and not row["notable_forgetting"], f"oubli renseigné pour une nouvelle cible : {directory}, {target}")
            else:
                distance = previous[target]
                change = row["final_distance"] - distance
                _check(row["previous_distance"] is not None and _close(row["previous_distance"], distance)
                       and row["forgetting"] is not None and _close(row["forgetting"], change),
                       f"différence d'oubli non appariée : {directory}, {target}")
                _check(row["notable_forgetting"] == (change > tolerance), f"seuil d'oubli incohérent : {directory}, {target}")
                # Les statistiques utilisent la différence recomputée, pas un arrondi du CSV.
                row["previous_distance"], row["forgetting"] = distance, change
            previous[target] = row["final_distance"]
        return rows

    def _load(self) -> tuple[dict, list[dict]]:
        candidates = sorted(self.root.glob("*/seed_*/effective_config.yaml"))
        if not candidates:
            raise FileNotFoundError(f"aucune campagne curriculum sauvegardée : {self.root}")
        raw = _mapping(candidates[0])
        _check(isinstance(raw.get("variants"), dict) and raw["variants"], "variantes sauvegardées absentes")
        _check(isinstance(raw.get("levels"), list) and len(raw["levels"]) == 4, "quatre niveaux sauvegardés sont requis")
        _check(raw.get("transfer_checkpoint") == "best" and raw.get("target_sampling") == "cyclic",
               "contrat de mesure du curriculum incompatible")
        _check(self.profile in raw.get("profiles", {}), f"profil absent de la configuration sauvegardée : {self.profile}")
        budget = raw["profiles"][self.profile]
        for key in ("seeds", "validation_seeds"):
            values = budget.get(key)
            _check(isinstance(values, list) and values, f"provenance absente : {key}")
            values = [_integer(value, key) for value in values]
            _check(len(values) == len(set(values)), f"graines dupliquées : {key}")
        _check(not set(budget["seeds"]).intersection(budget["validation_seeds"]), "graines d'entraînement et validation communes")
        tolerance = _number(raw.get("forgetting_tolerance"), "forgetting_tolerance")
        _check(tolerance >= 0, "tolérance d'oubli négative")
        variants = set(raw["variants"])
        _check(all(isinstance(name, str) and name.isidentifier() for name in variants), "nom de variante invalide")
        present = {path.name for path in self.root.iterdir() if path.is_dir() and not path.name.startswith("level_")}
        _check(present == variants, f"variantes manquantes ou inattendues : {sorted(present ^ variants)}")
        rows, shape_ids, references = [], {}, {}
        common, version = None, None
        for variant in sorted(variants):
            directory = self.root / variant
            actual_seeds = {path.name for path in directory.iterdir() if path.is_dir()}
            expected_seeds = {f"seed_{seed}" for seed in budget["seeds"]}
            _check(actual_seeds == expected_seeds, f"graines manquantes ou inattendues pour {variant} : {sorted(actual_seeds ^ expected_seeds)}")
            for seed in sorted(budget["seeds"]):
                campaign = directory / f"seed_{seed}"
                _required(campaign, CAMPAIGN_FILES)
                _check(_mapping(campaign / "effective_config.yaml") == raw, f"configurations curriculum différentes : {campaign}")
                _identity(_mapping(campaign / "status.json"), variant, seed, self.profile, campaign)
                summary = _mapping(campaign / "summary.json")
                _identity(summary, variant, seed, self.profile, campaign)
                provenance = _mapping(campaign / "seeds.json")
                _check(provenance.get("root_seed") == seed and provenance.get("validation_seeds") == budget["validation_seeds"],
                       f"provenance des graines incohérente : {campaign}")
                saved_version = (campaign / "version.txt").read_text(encoding="utf-8").strip()
                if version is None:
                    version = saved_version
                _check(bool(saved_version) and saved_version == version, f"versions différentes : {campaign}")
                levels = summary.get("levels")
                _check(isinstance(levels, list) and len(levels) == 4 and all(isinstance(level, dict) for level in levels),
                       f"résumé de niveaux incomplet : {campaign}")
                _check([level.get("level") for level in levels] == [1, 2, 3, 4], f"identité des niveaux incohérente : {campaign}")
                expected, weights = {}, {}
                for number in range(1, 5):
                    saved, level_summary, expected[number], weights[number] = self._level(raw, variant, seed, number, budget, shape_ids)
                    _check(levels[number - 1] == {"level": number, **level_summary}, f"résumés de niveau incohérents : {campaign}")
                    level_dir = self.root / f"level_{number}" / variant / f"seed_{seed}"
                    _check((level_dir / "version.txt").read_text(encoding="utf-8").strip() == version,
                           f"version de niveau différente : {level_dir}")
                    comparable = self._comparable(saved, raw)
                    key = (variant, number)
                    _check(references.setdefault(key, comparable) == comparable, f"configurations différentes entre graines : {level_dir}")
                    comparable = self._comparable(saved, raw, across_variants=True)
                    if common is None:
                        common = comparable
                    _check(common == comparable, f"configurations différentes hors variantes et graines : {level_dir}")
                campaign_rows = self._rows(campaign, variant, seed, expected, weights, shape_ids, budget, tolerance)
                _check(_close(_number(summary.get("forgetting_tolerance"), "forgetting_tolerance"), tolerance),
                       f"tolérance du résumé incohérente : {campaign}")
                _check(summary.get("notable_forgetting_count") == sum(row["notable_forgetting"] for row in campaign_rows),
                       f"compteur d'oubli incohérent : {campaign}")
                self._checkpoint_path(summary.get("final_checkpoint", ""), 4, variant, seed, raw)
                rows.extend(campaign_rows)
        return raw, sorted(rows, key=lambda row: (row["variant"], row["seed"], row["level"], row["target"]))

    def run(self) -> dict:
        raw, rows = self._load()  # Aucune sortie n'est écrite avant cette vérification complète.
        distances, changes = defaultdict(list), defaultdict(list)
        for row in rows:
            key = (row["variant"], row["level"], row["target"])
            distances[key].append(row["final_distance"])
            if row["forgetting"] is not None:
                changes[key].append(row["forgetting"])
        matrix = [{"variant": variant, "level": level, "target": target, **_stats(values)}
                  for (variant, level, target), values in sorted(distances.items())]
        tolerance = float(raw["forgetting_tolerance"])
        forgetting = []
        for (variant, level, target), values in sorted(changes.items()):
            stats = _stats(values)
            forgetting.append({"variant": variant, "level": level, "previous_level": level - 1,
                               "target": target, **stats, "forgetting_tolerance": tolerance,
                               "notable_forgetting": stats["mean"] > tolerance,
                               "notable_forgetting_count": sum(value > tolerance for value in values)})
        count = sum(row["notable_forgetting"] for row in rows)
        summary = {"status": "completed", "profile": self.profile, "variants": sorted(raw["variants"]),
                   "training_seeds": sorted(raw["profiles"][self.profile]["seeds"]),
                   "validation_seeds": raw["profiles"][self.profile]["validation_seeds"],
                   "level_count": 4, "row_count": len(rows), "measurement_checkpoint": "best",
                   "aggregation_unit": "training_seed", "forgetting_tolerance": tolerance,
                   "notable_forgetting_count": count, "error_count": count, "retention_ok": count == 0,
                   "notable_forgetting_group_count": sum(row["notable_forgetting"] for row in forgetting),
                   "distance_matrix": matrix, "forgetting": forgetting,
                   "notes": "Moyennes par cible déjà évaluées sur les mêmes graines de validation ; "
                            "dispersion et IC normal à 95 % entre graines d'entraînement, sans mélange d'épisodes. "
                            "L'oubli est la différence appariée avec le niveau précédent ; les nouvelles cibles "
                            "n'ont pas d'oubli. Le compteur d'erreurs compte les dépassements par graine, niveau "
                            "et cible. Le meilleur état gelé est le contrat du producteur : aucun tenseur "
                            "n'est chargé pour refaire ou certifier ces mesures."}
        _write_csv(self.root / "losses_by_target.csv", rows, ("variant", "seed", *ROW_KEYS))
        _write_csv(self.root / "distance_matrix.csv", matrix, ("variant", "level", "target", "mean", "std_sample", "ci95", "n"))
        _write_csv(self.root / "forgetting.csv", forgetting,
                   ("variant", "level", "previous_level", "target", "mean", "std_sample", "ci95", "n",
                    "forgetting_tolerance", "notable_forgetting", "notable_forgetting_count"))
        self._heatmaps(matrix, sorted(raw["variants"]))
        (self.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return summary

    def _heatmaps(self, matrix: list[dict], variants: list[str]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        minimum = min(row["mean"] for row in matrix)
        maximum = max(row["mean"] for row in matrix)
        for variant in variants:
            entries = [row for row in matrix if row["variant"] == variant]
            targets = sorted({row["target"] for row in entries})
            indices = {target: index for index, target in enumerate(targets)}
            values = np.full((len(targets), 4), np.nan)
            for row in entries:
                values[indices[row["target"]], row["level"] - 1] = row["mean"]
            fig, axis = plt.subplots(figsize=(7.5, max(3.5, len(targets) * 0.25)))
            try:
                palette = plt.get_cmap("viridis").with_extremes(bad="lightgray")
                image = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=palette, vmin=minimum, vmax=maximum)
                axis.set_xticks(range(4), [1, 2, 3, 4])
                axis.set_yticks(range(len(targets)), targets, fontsize=7)
                axis.set_xlabel("Niveau")
                axis.set_ylabel("Cible")
                axis.set_title(f"Distance finale moyenne — {variant}\nGris : cible non encore introduite")
                fig.colorbar(image, ax=axis, label="Distance finale")
                fig.tight_layout()
                fig.savefig(self.root / f"heatmap_{variant}.png", dpi=150)
            finally:
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="curriculum")
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--output", type=Path, help="Racine des artefacts sauvegardés, si déplacée.")
    args = parser.parse_args()
    try:
        summary = CurriculumReport.from_config(args.config, args.profile, args.output).run()
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Rapport refusé : {error}\n")
    print(json.dumps({"status": summary["status"], "forgetting_tolerance": summary["forgetting_tolerance"],
                      "notable_forgetting_count": summary["notable_forgetting_count"],
                      "retention_ok": summary["retention_ok"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
