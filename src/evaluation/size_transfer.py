"""Étude du transfert de taille : campagnes sur checkpoints gelés, sans optimisation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.common.config import ConfigLoader
from src.evaluation.campaign import Campaign
from src.evaluation.evaluator import Evaluator
from src.evaluation.factory import ControllerFactory
from src.evaluation.results import ResultsTable


class SizeTransferStudy:
    def __init__(self, config: dict, loader: ConfigLoader | None = None) -> None:
        self.config = config
        self.loader = loader or ConfigLoader()
        self.output = Path(config["output"])

    def run(self) -> dict:
        evaluation = self.loader.load_raw(self.config["evaluation_config"])
        evaluation = {**evaluation, "swarm_sizes": self.config["evaluation_sizes"],
                      "controllers": {self.config["controller"]: evaluation["controllers"][self.config["controller"]]}}
        factories = {}
        for size in self.config["training_sizes"]:
            factory = ControllerFactory(self.loader, evaluation, Path(self.config["checkpoint_roots"][size]))
            for seed in factory.versions(self.config["controller"]):
                snapshot = factory.snapshot(self.config["controller"], seed)
                trained_size = snapshot["training"]["profiles"][snapshot["profile"]]["n_robots"]
                if trained_size != size:
                    raise ValueError(f"checkpoint annoncé pour {size} robots mais entraîné sur {trained_size}")
            factories[size] = factory
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError(f"étude déjà présente : {self.output}")
        tables = {}
        for size, factory in factories.items():
            directory = self.output / f"trained_on_{size}"
            evaluator = Evaluator.from_config(self.loader, directory, evaluation)
            Campaign(evaluator, factory, evaluation, directory, self.loader).run([self.config["controller"]])
            tables[size] = ResultsTable.read_csv(directory / "episodes.csv")
        return self.build_report(tables)

    def build_report(self, tables: dict[int, ResultsTable]) -> dict:
        records = []
        for training_size, table in tables.items():
            for (_, evaluation_size, failure, shape), stats in table.variation_between_seeds().items():
                records.append({"training_size": training_size, "evaluation_size": evaluation_size,
                                "failure_fraction": failure, "shape": shape, **stats})
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "cross_size.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        source_size = self.config["training_sizes"][0]
        degradation = []
        direct = {(row["evaluation_size"], row["failure_fraction"], row["shape"]): row for row in records if row["training_size"] == row["evaluation_size"]}
        for row in records:
            if row["training_size"] != source_size or row["evaluation_size"] == source_size:
                continue
            key = (row["evaluation_size"], row["failure_fraction"], row["shape"])
            specialized = direct[key]
            degradation.append({"source_size": source_size, "evaluation_size": row["evaluation_size"],
                                "failure_fraction": row["failure_fraction"], "shape": row["shape"],
                                "transfer_distance": row["mean"], "direct_distance": specialized["mean"],
                                "relative_degradation": (row["mean"] - specialized["mean"]) / specialized["mean"] if specialized["mean"] else None})
        self._plot(records)
        summary = {"cross_size": records, "degradation": degradation}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.output / "limitations.txt").write_text(
            "Les poids sont gelés lors du transfert. La portée reste identique : la densité et la connectivité changent avec la taille.\n"
            "Les dispersions du tableau proviennent des graines d'entraînement ; les agrégats des épisodes restent dans chaque campagne.\n"
            "Une distance directe nulle rend la dégradation relative indéfinie.\n", encoding="utf-8")
        return summary

    def _plot(self, records: list[dict]) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conditions = sorted({(row["failure_fraction"], row["shape"]) for row in records})
        for failure, shape in conditions:
            fig, axis = plt.subplots()
            for training_size in self.config["training_sizes"]:
                selected = sorted((row for row in records if row["training_size"] == training_size and row["failure_fraction"] == failure and row["shape"] == shape), key=lambda row: row["evaluation_size"])
                axis.errorbar([row["evaluation_size"] for row in selected], [row["mean"] for row in selected],
                              yerr=[row["ci95"] for row in selected], label=f"entraîné sur {training_size}", marker="o")
            axis.set_xlabel("taille d'évaluation")
            axis.set_ylabel("distance finale (IC entre graines)")
            axis.legend()
            fig.tight_layout()
            fig.savefig(self.output / f"transfer_failure{round(failure * 100):02d}_{shape.replace('@', '_at_')}.png")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="size-transfer")
    args = parser.parse_args()
    SizeTransferStudy(ConfigLoader().load_raw(args.config)).run()


if __name__ == "__main__":
    main()
