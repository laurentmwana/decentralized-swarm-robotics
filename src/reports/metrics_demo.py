"""Livrables phase 2 : courbe d'exemple + table de comparabilite 20/40/100.

Meme episode (meme graine, meme cible, meme controleur trivial) rejoue a trois tailles :
les metriques normalisees (Chamfer / rayon cible, taux, regularite rapportee) doivent
rester comparables d'une taille a l'autre. Aucun parametre : tout vient de config/.

Usage : python -m src.reports.metrics_demo
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.trivial import TrivialController
from src.metrics import MetricSuite
from src.shapes.index import generate_index
from src.simulator.episode import Episode

OUTPUT = Path("outputs/metrics")
FIGURE = Path("figures/example_distance_curve.png")

TABLE_KEYS = [
    "final_distance",
    "convergence_step",
    "success",
    "collision_rate",
    "regularity_value",
    "normalized_regularity",
    "final_components",
]


def run() -> dict:
    loader = ConfigLoader()
    cfg_sim = loader.load_simulator()
    cfg_metrics = loader.load_metrics()
    cfg_shapes = loader.load_shapes()
    cfg_demo = loader.load_raw("demo-trivial")

    index, _ = generate_index(cfg_shapes, Path("outputs/shapes"))
    target = index.point_cloud_by_key(cfg_demo["target"])
    seed = int(cfg_demo["episode_seed"])

    episode = Episode(cfg_sim)
    suite = MetricSuite(cfg_metrics, cfg_sim, cfg_shapes.target_radius)
    mission = Mission(shape_id=0, target=target)

    rows: list[dict[str, object]] = []
    curves: dict[int, np.ndarray] = {}
    for swarm_size in cfg_demo["swarm_sizes"]:
        n = int(swarm_size)
        controller = TrivialController(**cfg_demo["controller"])
        result = episode.execute(mission=mission, controller=controller, n_robots=n, episode_seed=seed)
        report = suite.compute(result.positions, result.active, target)
        curves[n] = np.asarray(report.values["distance_curve"])
        rows.append({"n_robots": n, "episode_seed": seed, **{key: report.values[key] for key in TABLE_KEYS}})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, OUTPUT / "comparability.csv")
    summary = {
        "target": cfg_demo["target"],
        "episode_seed": seed,
        "success_threshold": cfg_metrics.success_threshold,
        "rows": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _figure(curves, cfg_metrics.success_threshold)
    for row in rows:
        print(f"[metrics] {row['n_robots']} robots : { {key: row[key] for key in TABLE_KEYS[:4]} }")
    return summary


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = ["n_robots", "episode_seed"] + TABLE_KEYS
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")


def _figure(curves: dict[int, np.ndarray], threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7, 4.2))
    for n, curve in curves.items():
        axis.plot(curve, label=f"{n} robots")
    axis.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2, label=f"seuil = {threshold}")
    axis.set_xlabel("pas")
    axis.set_ylabel("Chamfer normalisee (rayon cible)")
    axis.set_title("Courbe de distance — meme episode (graine unique), 20 / 40 / 100 robots")
    axis.legend()
    axis.grid(alpha=0.3)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    print(f"[metrics] figure : {FIGURE}")


if __name__ == "__main__":
    run()
