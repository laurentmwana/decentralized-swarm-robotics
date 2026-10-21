"""Constructeurs de tableaux et figures standard à partir des fichiers bruts."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .results import ResultsTable


class TableBuilder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_aggregates(self, table: ResultsTable, metrics: list[str]) -> Path:
        path = self.output_dir / "aggregates.csv"
        rows = []
        for metric in metrics:
            for (controller, seed, swarm_size, failure, shape), agg in table.aggregates(metric).items():
                rows.append(
                    {"metric": metric, "controller": controller, "training_seed": seed,
                     "n_robots": swarm_size, "failure_fraction": failure, "shape": shape, **agg}
                )
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_markdown(self, table: ResultsTable, metric: str, path: Path) -> Path:
        rows = []
        for (controller, seed, swarm_size, failure, shape), agg in sorted(table.aggregates(metric).items()):
            rows.append(
                f"| {controller} | {seed if seed is not None else '-'} | {swarm_size} | {failure:.2f} | {shape} "
                f"| {agg['mean']:.3f} ± {agg['std']:.3f} | [{agg['mean'] - agg['ci95']:.3f}, {agg['mean'] + agg['ci95']:.3f}] | {agg['n']} |"
            )
        text = [
            f"# Agrégats — {metric}",
            "",
            "| contrôleur | graine | robots | retraits | forme | moyenne ± écart-type | IC 95 % | n |",
            "|---|---|---|---|---|---|---|---|",
            *rows,
            "",
        ]
        path.write_text("\n".join(text), encoding="utf-8")
        return path


class FigureBuilder:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)

    def distance_curves(self, position_dirs: list[tuple[str, Path]], path: Path, title: str, success_threshold: float) -> Path | None:
        """Courbes recalculées depuis les positions, avec dispersion entre épisodes."""
        from src.metrics.chamfer import chamfer_step

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(7.5, 4.5))
        for name, output_dir in position_dirs:
            curves = []
            for npz in sorted(output_dir.glob("*.npz")):
                with np.load(npz) as data:
                    positions, active, target = data["positions"], data["active"], data["target"]
                    radius = float(data["target_radius"])
                step = np.array(
                    [chamfer_step(positions[t][active[t]], target) / radius for t in range(len(positions))]
                )
                curves.append(step)
            if not curves:
                continue
            matrix = np.vstack(curves)
            mean = matrix.mean(axis=0)
            axis.plot(mean, label=f"{name} (moyenne)")
            axis.fill_between(range(len(mean)), matrix.min(axis=0), matrix.max(axis=0), alpha=0.15)
        axis.axhline(success_threshold, color="black", linestyle="--", label=f"seuil {success_threshold:g}")
        axis.set_xlabel("pas")
        axis.set_ylabel("Chamfer normalisée")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def distance_boxes(self, rows: list[dict], path: Path, title: str) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            groups[row["controller"]].append(float(row["final_distance"]))
        names = sorted(groups)
        fig, axis = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.2))
        axis.boxplot([groups[n] for n in names], tick_labels=names, showfliers=False)
        axis.set_ylabel("distance finale")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def success_by_size(self, table: ResultsTable, path: Path, title: str) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        aggregates = table.aggregates("success")
        controllers = sorted({key[0] for key in aggregates})
        swarm_sizes = sorted({key[2] for key in aggregates})
        width = 0.8 / max(1, len(controllers))
        fig, axis = plt.subplots(figsize=(7.5, 4.2))
        for i, controller in enumerate(controllers):
            rate = []
            for swarm_size in swarm_sizes:
                values = [agg["mean"] for key, agg in aggregates.items() if key[0] == controller and key[2] == swarm_size]
                rate.append(float(np.mean(values)) if values else 0.0)
            axis.bar(np.arange(len(swarm_sizes)) + i * width, rate, width, label=controller)
        axis.set_xticks(np.arange(len(swarm_sizes)) + width * (len(controllers) - 1) / 2)
        axis.set_xticklabels(swarm_sizes)
        axis.set_xlabel("taille d'essaim")
        axis.set_ylabel("taux de succès")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
