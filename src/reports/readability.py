"""Protocole humain aveugle : archives seulement, aucune réponse ni note inventée."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import yaml

from src.common.config import ConfigLoader
from src.evaluation.results import ResultsTable
from src.evaluation.shape_studies import archive_path, require_empty, write_records


class ReadabilityProtocol(ABC):
    @abstractmethod
    def prepare(self) -> dict:
        """Prépare les stimuli aveugles et une vérité séparée."""
        raise NotImplementedError

    @abstractmethod
    def score(self) -> dict:
        """Analyse uniquement les réponses réellement reçues."""
        raise NotImplementedError


class HumanReadabilityProtocol(ReadabilityProtocol):
    def __init__(self, config: dict) -> None:
        self.config = copy.deepcopy(config)
        self.source = Path(config["source"]).resolve()
        self.output = Path(config["output"]).resolve()
        self.score_output = Path(config["score_output"]).resolve()

    @staticmethod
    def sample_id(positions: np.ndarray, active: np.ndarray, ordinal: int, blind_seed: int) -> str:
        # L'identité opaque dépend du rendu, jamais de la vérité, du contrôleur ou de son URL.
        digest = hashlib.sha256()
        digest.update(np.asarray(positions, dtype="<f8").tobytes())
        digest.update(np.asarray(active, dtype=np.uint8).tobytes())
        digest.update(f"{blind_seed}:{ordinal}".encode("ascii"))
        return "sample_" + digest.hexdigest()[:24]

    def _select(self) -> list[dict]:
        with (self.source / "effective_config.yaml").open(encoding="utf-8") as stream:
            snapshot = yaml.safe_load(stream)
        if json.loads((self.source / "summary.json").read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError("la campagne source n'est pas complète")
        rows = ResultsTable.read_csv(self.source / "episodes.csv").rows
        selected = [row for row in rows if row["controller"] in self.config["controllers"]
                    and row["training_seed"] in (None, self.config["training_seed"])
                    and row["episode_seed"] in self.config["episode_seeds"]
                    and row["n_robots"] == self.config["n_robots"]
                    and row["failure_fraction"] == self.config["failure_fraction"]]
        shapes = snapshot["evaluation"]["shapes"]
        expected = {(name, shape, seed) for name in self.config["controllers"]
                    for shape in shapes for seed in self.config["episode_seeds"]}
        actual = {(row["controller"], row["shape"], row["episode_seed"]) for row in selected}
        if not expected or actual != expected or len(selected) != len(expected):
            raise ValueError("échantillonnage fixe incomplet ou dupliqué dans la campagne source")
        return sorted(selected, key=lambda row: (row["controller"], row["shape"], row["episode_seed"]))

    def prepare(self) -> dict:
        require_empty(self.output)
        if self.output == self.source or self.source in self.output.parents:
            raise ValueError("les planches doivent rester séparées de la campagne source")
        if (self.config["render_extent"] <= 0 or self.config["point_size"] <= 0
                or self.config["image_pixels"] < 1 or self.config["image_dpi"] < 1):
            raise ValueError("paramètres de rendu invalides")
        selected = self._select()
        stimuli = []
        for ordinal, row in enumerate(selected):
            path = archive_path(self.source, row)
            with np.load(path, allow_pickle=False) as data:
                positions = np.asarray(data["positions"])
                active = np.asarray(data["active"])
            if (positions.ndim != 3 or positions.shape[-1] != 2 or len(positions) == 0
                    or active.shape != positions.shape[:2] or not np.isfinite(positions).all()
                    or not np.isin(active, [0, 1]).all()):
                raise ValueError(f"archive de positions invalide : {path}")
            final, mask = positions[-1], active[-1].astype(bool)
            if not mask.any() or np.abs(final[mask]).max() > self.config["render_extent"]:
                raise ValueError("rendu vide ou hors cadre : ne pas masquer des robots par recadrage")
            identifier = self.sample_id(final, mask, ordinal, self.config["blind_seed"])
            stimuli.append((identifier, final[mask].copy(), row, path))
        order = np.random.default_rng(self.config["blind_seed"]).permutation(len(stimuli))
        stimuli = [stimuli[index] for index in order]
        # Toutes les archives ont été contrôlées avant la première écriture.
        (self.output / "images").mkdir(parents=True, exist_ok=True)
        (self.output / "private").mkdir(exist_ok=True)
        questionnaire, truth = [], []
        for identifier, final, row, path in stimuli:
            image = f"images/{identifier}.png"
            self._render(final, self.output / image)
            questionnaire.append({"sample_id": identifier, "image_path": image})
            truth.append({"sample_id": identifier, "true_shape": row["shape"],
                          "controller": row["controller"], "training_seed": row["training_seed"],
                          "episode_seed": row["episode_seed"], "source_archive": str(path),
                          "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        write_records(self.output / "questionnaire.csv", questionnaire)
        write_records(self.output / "private" / "truth.csv", truth)
        write_records(self.output / "responses_template.csv", [], ["participant_id", "sample_id", "predicted_shape"])
        cards = [f'<figure><img src="{row["image_path"]}" alt="{row["sample_id"]}">'
                 f'<figcaption>{row["sample_id"]}</figcaption></figure>' for row in questionnaire]
        instruction = html.escape(self.config["instructions"])
        page = ('<!doctype html><html lang="fr"><meta charset="utf-8"><title>Questionnaire aveugle</title>'
                '<style>body{font-family:sans-serif}figure{display:inline-block}img{width:320px}</style>'
                f'<h1>Reconnaissance de formes</h1><p>{instruction}</p>' + "\n".join(cards) + '</html>')
        (self.output / "questionnaire.html").write_text(page, encoding="utf-8")
        (self.output / "private" / "effective_config.yaml").write_text(
            yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")
        summary = self._pending(len(stimuli))
        summary["method"] = "human_blind_questionnaire"
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output / "private" / "protocol.txt").write_text(
            "Ne distribuer que questionnaire.html, questionnaire.csv et images/. Garder private/ hors du serveur public.\n"
            "Aucune cible, aucun contrôleur, aucune note géométrique dans les images ni leurs URL.\n"
            "Les participants saisissent une réponse libre ; les noms canoniques sont geometries_circle, digits_8, letters_Y, numbers_314, etc.\n"
            "Le recrutement et le nombre de participants ne sont pas présumés. Le CSV représente les seules observations reçues.\n"
            "Chaque participant doit répondre à tous les stimuli si require_complete est activé. Les réponses inconnues sont incorrectes.\n"
            "Les stimuli sont fixés avant collecte, sans sélection selon distance ou réussite.\n",
            encoding="utf-8")
        return summary

    def _render(self, final: np.ndarray, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        size = self.config["image_pixels"] / self.config["image_dpi"]
        figure, axis = plt.subplots(figsize=(size, size), dpi=self.config["image_dpi"])
        axis.scatter(final[:, 0], final[:, 1], s=self.config["point_size"], color="black")
        extent = self.config["render_extent"]
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_aspect("equal")
        axis.axis("off")
        figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        figure.savefig(path, metadata={"Software": "swarm-gnn"})
        plt.close(figure)

    @staticmethod
    def _pending(sample_count: int) -> dict:
        return {"status": "pending", "sample_count": sample_count, "participant_count": 0,
                "response_count": 0, "accuracy": None, "std_between_participants": None,
                "confidence_interval": None}

    def analyze(self, truth: dict[str, str], responses: list[dict]) -> tuple[dict, list[dict]]:
        if not truth:
            raise ValueError("vérité de référence vide")
        if not responses:
            return self._pending(len(truth)), []
        groups, identities = {}, set()
        for response in responses:
            if set(response) != {"participant_id", "sample_id", "predicted_shape"} or any(
                not isinstance(value, str) or not value.strip() for value in response.values()
            ):
                raise ValueError("réponse invalide : participant_id,sample_id,predicted_shape requis")
            participant, sample, prediction = (response[key].strip() for key in ("participant_id", "sample_id", "predicted_shape"))
            if sample not in truth or (participant, sample) in identities:
                raise ValueError("sample_id inconnu ou réponse dupliquée")
            identities.add((participant, sample))
            groups.setdefault(participant, []).append(prediction == truth[sample])
        records = []
        for participant, correct in sorted(groups.items()):
            if self.config["require_complete"] and len(correct) != len(truth):
                raise ValueError(f"réponses incomplètes pour le participant {participant}")
            records.append({"participant_id": participant, "correct_count": int(sum(correct)),
                            "response_count": len(correct), "sample_count": len(truth),
                            "coverage": len(correct) / len(truth), "accuracy": float(np.mean(correct))})
        values = np.array([row["accuracy"] for row in records])
        level = self.config["confidence_level"]
        repetitions = self.config["bootstrap_repetitions"]
        if not 0 < level < 1 or not isinstance(repetitions, int) or repetitions < 2:
            raise ValueError("paramètres d'intervalle de confiance invalides")
        interval = None
        if len(values) > 1:
            rng = np.random.default_rng(self.config["bootstrap_seed"])
            # Rééchantillonnage des participants, jamais des réponses comme observations indépendantes.
            means = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(repetitions)])
            interval = np.quantile(means, [(1 - level) / 2, (1 + level) / 2]).tolist()
        summary = {"status": "observed", "sample_count": len(truth), "participant_count": len(records),
                   "response_count": len(responses), "accuracy": float(values.mean()),
                   "std_between_participants": float(values.std(ddof=1)) if len(values) > 1 else None,
                   "confidence_interval": interval, "confidence_level": level,
                   "interval_method": "participant_percentile_bootstrap", "uncertainty_unit": "participant",
                   "limitations": ["IC non estimable avec un seul participant ; petit effectif fragile.",
                                   "Inférence conditionnelle aux stimuli fixes, sans extrapolation à un panel représentatif.",
                                   "Une exactitude géométrique ne prouve pas la lisibilité ; aucun classifieur entraîné ici."]}
        return summary, records

    def score(self) -> dict:
        require_empty(self.score_output)
        if self.score_output == self.output or self.score_output == self.source or self.source in self.score_output.parents:
            raise ValueError("dossier de scores distinct requis")
        path = self.output / "private" / "truth.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            truth_rows = list(csv.DictReader(stream))
        truth = {row["sample_id"]: row["true_shape"] for row in truth_rows}
        if len(truth) != len(truth_rows):
            raise ValueError("vérité de référence dupliquée")
        response_path = Path(self.config["responses"]).resolve() if self.config["responses"] else None
        responses, raw = [], None
        if response_path is not None and response_path.exists():
            raw = response_path.read_bytes()
            with response_path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != ["participant_id", "sample_id", "predicted_shape"]:
                    raise ValueError("colonnes CSV attendues : participant_id,sample_id,predicted_shape")
                responses = list(reader)
        summary, records = self.analyze(truth, responses)
        summary.update(method="human_blind_questionnaire", truth_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       responses_sha256=hashlib.sha256(raw).hexdigest() if raw is not None else None)
        self.score_output.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            (self.score_output / "responses.csv").write_bytes(raw)
        write_records(self.score_output / "per_participant.csv", records,
                      ["participant_id", "correct_count", "response_count", "sample_count", "coverage", "accuracy"])
        (self.score_output / "effective_config.yaml").write_text(yaml.safe_dump(self.config, allow_unicode=True), encoding="utf-8")
        (self.score_output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    def run(self, mode: str | None = None) -> dict:
        mode = mode or self.config["mode"]
        if mode not in ("prepare", "score"):
            raise ValueError(f"mode de lisibilité inconnu : {mode}")
        return self.prepare() if mode == "prepare" else self.score()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="readability")
    parser.add_argument("--mode", choices=("prepare", "score"), default=None)
    args = parser.parse_args()
    path = Path(args.config)
    loader = ConfigLoader(path.resolve().parent) if path.suffix in (".yaml", ".yml") else ConfigLoader()
    config = loader.load_raw(path.stem if path.suffix else args.config)
    HumanReadabilityProtocol(config).run(args.mode)


if __name__ == "__main__":
    main()
