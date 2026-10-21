"""Demonstration du critere d'acceptation phase 1 : le controleur trivial tourne sur
20, 40 et 100 robots, et deux executions donnent les memes positions.

Usage : `python -m src.reports.demo_phase1`
Sorties : outputs/phase1_demo/ (dossier de run standard, plan 0.4) et
figures/demo_trivial_sizes.png (generee depuis outputs/, jamais retouchee a la main).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.common.config import ConfigLoader
from src.controllers.trivial import TrivialController
from src.shapes.index import ShapeIndex, generate_index
from src.simulator.episode import Episode, EpisodeResult
from src.simulator.state import SwarmState
from src.simulator.graph import CommunicationGraph

OUTPUT = Path("outputs/phase1_demo")
FIGURES = Path("figures/demo_trivial_sizes.png")


def _version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "inconnue"


def _load_index(cfg_shapes) -> tuple[ShapeIndex, dict]:
    index, split = generate_index(cfg_shapes, Path("outputs/shapes"))
    return index, split


def _log(res: EpisodeResult, graph: CommunicationGraph, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "mean_displacement", "connected_components"])
        for t in range(res.step):
            displacement = np.linalg.norm(res.commands[t], axis=1)[res.active[t]].mean() if res.active[t].any() else 0.0
            adj = graph.adjacency(SwarmState(positions=res.positions[t], active=res.active[t], step=t))
            writer.writerow([t, round(float(displacement), 6), graph.connected_components(adj, res.active[t])])


def run() -> dict:
    loader = ConfigLoader()
    cfg_demo = loader.load_raw("demo-trivial")
    cfg_sim = loader.load_simulator()
    cfg_shapes = loader.load_shapes()
    index, _ = _load_index(cfg_shapes)

    target_key = cfg_demo["target"]
    target = index.point_cloud_by_key(target_key)

    controller = TrivialController(**cfg_demo["controller"])
    episode = Episode(cfg_sim)
    graph = CommunicationGraph(cfg_sim.r_c)
    seed = int(cfg_demo["episode_seed"])

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"episode_seed": seed, "target": target_key, "swarm_sizes": {}}
    panels: list[tuple[int, np.ndarray, np.ndarray]] = []

    for swarm_size in cfg_demo["swarm_sizes"]:
        output_dir = OUTPUT / f"trivial_{swarm_size}"
        output_dir.mkdir(parents=True, exist_ok=True)
        mission = _mission(target)
        result = episode.execute(mission=mission, controller=controller, n_robots=int(swarm_size), episode_seed=seed)
        repeat = episode.execute(mission=mission, controller=controller, n_robots=int(swarm_size), episode_seed=seed)
        identical = np.array_equal(result.positions, repeat.positions)

        np.save(output_dir / "positions.npy", result.positions)
        (output_dir / "effective_config.yaml").write_text(
            yaml.safe_dump({"simulator": cfg_sim.__dict__, "shapes": cfg_shapes.__dict__, "demo": cfg_demo},
                           allow_unicode=True, sort_keys=False), encoding="utf-8")
        (output_dir / "seeds.json").write_text(json.dumps({"episode_seed": seed, "repetition": seed}), encoding="utf-8")
        (output_dir / "version.txt").write_text(_version() + "\n", encoding="utf-8")
        _log(result, graph, output_dir / "log.csv")
        size_summary = {"reproducible": identical, "step": result.step, "n_robots": int(swarm_size), "episode_seed": seed}
        (output_dir / "summary.json").write_text(json.dumps(size_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["swarm_sizes"][str(swarm_size)] = size_summary
        panels.append((int(swarm_size), target, result.positions[-1]))
        print(f"[demo] {swarm_size} robots : reproductible={identical}")

    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _figure(panels)
    return summary


def _mission(target: np.ndarray):
    from src.common.mission import Mission

    return Mission(shape_id=0, target=target)


def _figure(panels: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for axis, (swarm_size, target, final_positions) in zip(axes, panels):
        axis.axis("equal")
        axis.scatter(target[:, 0], target[:, 1], s=6, c="lightgray", label="cible")
        axis.scatter(final_positions[:, 0], final_positions[:, 1], s=12, c="tab:blue", label="robots (pas final)")
        axis.set_title(f"{swarm_size} robots")
        axis.legend(fontsize=7)
    fig.suptitle("Controleur trivial - etat final, graine 142 (demo phase 1)")
    FIGURES.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES, dpi=110)
    plt.close(fig)
    print(f"[demo] figure : {FIGURES}")


if __name__ == "__main__":
    result = run()
    if not all(v["reproducible"] for v in result["swarm_sizes"].values()):
        sys.exit(1)
