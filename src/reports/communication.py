"""Comparaison de la communication à consigne égale depuis une campagne figée."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.common.config import ConfigLoader
from src.evaluation.results import AGGREGATABLE_METRICS, PairedComparison, ResultsTable


class CommunicationComparison:
    def __init__(self, source: Path, output: Path, reference: str, controllers: list[str]) -> None:
        self.source = Path(source)
        self.output = Path(output)
        self.reference = reference
        self.controllers = controllers

    def _validate_protocol(self, snapshot: dict) -> list[dict]:
        information = []
        reference_runs = snapshot["training_runs"][self.reference]
        for name, settings in snapshot["evaluation"]["controllers"].items():
            if settings["type"] == "baseline":
                information.append({"controller": name, "training_seed": None, "shape_id": False,
                                    "target_geometry": "nearest_displacement" if name == "manual" else "none",
                                    "neighbors": True, "velocity": False, "message_rounds": 0})
                continue
            runs = snapshot["training_runs"][name]
            for seed, artifact in runs.items():
                config = artifact["config"]
                communication = config["override"].get("communication", True)
                if name in self.controllers or name == self.reference:
                    reference = reference_runs[seed]["config"]
                    for key in ("training_target", "validation_target", "optimizer", "losses"):
                        if config["training"][key] != reference["training"][key]:
                            raise ValueError(f"budget ou cibles différents : {name}, {key}")
                    if config["training"]["profiles"][config["profile"]] != reference["training"]["profiles"][reference["profile"]]:
                        raise ValueError(f"profils différents : {name}")
                    for key in ("use_target_geometry", "hidden_dim", "embedding_dim", "n_shapes", "aggregator", "activation"):
                        if config["model"][key] != reference["model"][key]:
                            raise ValueError(f"information ou architecture différente : {name}, {key}")
                    if config["simulator"] != reference["simulator"]:
                        raise ValueError(f"dynamique d'entraînement différente : {name}")
                    if communication != (name != self.reference):
                        raise ValueError(f"mode de communication incohérent : {name}")
                information.append({"controller": name, "training_seed": seed, "shape_id": True,
                                    "target_geometry": "nearest_unit_direction" if config["model"]["use_target_geometry"] else "none",
                                    "neighbors": communication, "velocity": True,
                                    "message_rounds": config["model"]["message_rounds"] if communication else 0})
        for name in self.controllers:
            if set(snapshot["training_runs"][name]) != set(reference_runs):
                raise ValueError(f"graines d'entraînement différentes : {name}")
        return information

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def run(self) -> dict:
        table = ResultsTable.read_csv(self.source / "episodes.csv")
        with (self.source / "effective_config.yaml").open(encoding="utf-8") as stream:
            snapshot = yaml.safe_load(stream)
        information = self._validate_protocol(snapshot)
        conditions = sorted({table.condition_of(row) for row in table.rows})
        self.output.mkdir(parents=True, exist_ok=True)
        self._write_csv(self.output / "information.csv", information)
        gains = []
        for size, failure, shape in conditions:
            rows = [row for row in table.rows if table.condition_of(row) == (size, failure, shape)]
            condition_table = ResultsTable(rows)
            records = []
            for metric in AGGREGATABLE_METRICS:
                for (name, seed, *_), stats in condition_table.aggregates(metric).items():
                    records.append({"controller": name, "training_seed": seed, "metric": metric, **stats,
                                    "success_threshold": snapshot["metrics"]["success_threshold"]})
            condition_name = f"n{size}_failure{round(failure * 100):02d}_{shape}"
            self._write_csv(self.output / f"table_{condition_name}.csv", records)
            condition_gains = []
            for name in self.controllers:
                reference_keys = {(row["training_seed"], row["episode_seed"]) for row in rows if row["controller"] == self.reference}
                candidate_keys = {(row["training_seed"], row["episode_seed"]) for row in rows if row["controller"] == name}
                if not reference_keys or reference_keys != candidate_keys:
                    raise ValueError(f"épisodes non appariés : {name}, {condition_name}")
                comparison = PairedComparison.build(rows, name, self.reference)
                stats = comparison.seed_summary()["final_distance"]
                record = {"controller": name, "reference": self.reference, "n_robots": size,
                          "failure_fraction": failure, "shape": shape, "mean_difference": stats["mean"],
                          "std_between_seeds": stats["std"], "ci95_between_seeds": stats["ci95"],
                          "seed_count": stats["n"], "paired_count": comparison.summary()["final_distance"]["paired_count"]}
                condition_gains.append(record)
                gains.append(record)
            self._plot(condition_gains, self.output / f"gain_{condition_name}.png")
        self._write_csv(self.output / "paired_gains.csv", gains)
        summary = {"success_threshold": snapshot["metrics"]["success_threshold"], "gains": gains}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        conclusions = ["Une différence négative signifie une distance plus faible avec communication.",
                       "Les intervalles sont calculés entre graines, pas en traitant leurs épisodes comme indépendants.",
                       "La règle manuelle reçoit une distance au contour ; le GNN reçoit une direction unitaire."]
        for row in gains:
            mean, interval = row["mean_difference"], row["ci95_between_seeds"]
            conclusion = "amélioration" if mean + interval < 0 else "dégradation" if mean - interval > 0 else "écart non concluant"
            conclusions.append(f"{row['controller']}, {row['shape']}, {row['n_robots']} robots : {conclusion}, différence {mean:.4f} ± {interval:.4f} (IC 95 %).")
        (self.output / "conclusion.txt").write_text("\n".join(conclusions) + "\n", encoding="utf-8")
        return summary

    @staticmethod
    def _plot(rows: list[dict], path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots()
        axis.errorbar([row["controller"] for row in rows], [-row["mean_difference"] for row in rows],
                      yerr=[row["ci95_between_seeds"] for row in rows], fmt="o", capsize=4)
        axis.axhline(0, color="black", linestyle="--")
        axis.set_ylabel("Gain de distance vs sans communication (IC entre graines)")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="comparison")
    args = parser.parse_args()
    config = ConfigLoader().load_raw(args.config)
    CommunicationComparison(Path(config["source"]), Path(config["output"]), config["reference_controller"], config["communication_controllers"]).run()


if __name__ == "__main__":
    main()
