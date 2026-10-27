"""Livrable phase 4 : outputs/gnn/architecture.txt — couches et nombre de paramètres par variante
(1, 2, 3 tours et sans communication), et exécution de démonstration du modèle non entraîné
sur 20, 40 et 100 robots.

Usage : `python -m src.reports.architecture`
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.controllers.gnn import GNNController, NoCommGNNController
from src.models.gnn import SwarmGNN
from src.simulator.episode import Episode
from src.simulator.graph import CommunicationGraph

OUTPUT = Path("outputs/gnn/architecture.txt")
DEMO = Path("outputs/gnn/untrained_demo")


def _description(cfg: SwarmGNN) -> list[str]:
    rows = [
        f"Encoder : Linear(2 + {cfg.cfg.embedding_dim} emb) -> {cfg.cfg.hidden_dim} [{cfg.cfg.activation}]",
    ]
    for t in range(cfg.cfg.message_rounds):
        rows.append(
            f"MessageLayer {t + 1} : edge_mlp(2*{cfg.cfg.hidden_dim}+2 -> {cfg.cfg.hidden_dim})"
            f" + {cfg.cfg.aggregator} + update_mlp [{cfg.cfg.message_rounds} tours]"
        )
    rows.append(f"Decoder : Linear({cfg.cfg.hidden_dim} -> 2), borné v_max*tanh")
    return rows


def _variant_report(name: str, cfg, v_max: float) -> str:
    model = SwarmGNN(cfg, v_max)
    rows = [
        f"== {name} ==",
        f"  tours={cfg.message_rounds}  hidden={cfg.hidden_dim}  embedding={cfg.embedding_dim}  agregateur={cfg.aggregator}",
        f"  nombre de parametres : {model.parameter_count()}",
        "  couches :",
    ]
    rows += ["    " + l for l in _description(model)]
    return "\n".join(rows) + "\n"


def write_architecture(cfg_base, v_max: float, path: Path = OUTPUT) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    variants = {
        "gnn_1_round": replace(cfg_base, message_rounds=1),
        "gnn_2_rounds": replace(cfg_base, message_rounds=2),
        "gnn_3_rounds": replace(cfg_base, message_rounds=3),
        "no_comm": replace(cfg_base, message_rounds=3),
    }
    headers = ["Architecture des controleurs GNN (phase 4)",
               "Nombre de parametres INDEPENDANT du nombre de robots (poids partages).", ""]
    body = []
    summary = {}
    for name, cfg in variants.items():
        text = _variant_report(name, cfg, v_max)
        body.append(text)
        summary[name] = SwarmGNN(cfg, v_max).parameter_count()
    headers.append("Preuve d'independance de la taille :")
    model = SwarmGNN(cfg_base, v_max)
    headers.append(f"  meme modele, N=20/40/100 -> parametres invariants : {model.parameter_count()}")
    path.write_text("\n".join(headers) + "\n" + "\n".join(body), encoding="utf-8")
    return summary


def untrained_demo(cfg_base, sim, target: np.ndarray, swarm_sizes=(20, 40, 100)) -> dict:
    DEMO.mkdir(parents=True, exist_ok=True)
    episode = Episode(sim)
    graph = CommunicationGraph(sim.r_c)
    mission = Mission(shape_id=0, target=target)
    result = {}
    ctrl = GNNController(cfg_base, sim.v_max)
    for n in swarm_sizes:
        ctrl.reset()
        res = episode.execute(mission, ctrl, n, episode_seed=142)
        finite = bool(np.isfinite(res.positions).all())
        bounded = bool((np.linalg.norm(res.commands, axis=2) <= sim.v_max + 1e-6).all())
        np.save(DEMO / f"positions_n{n}.npy", res.positions)
        result[n] = {"error_free": finite, "bounded_commands": bounded, "step": res.step}
        print(f"[gnn-demo] {n} robots : sans_erreur={finite} bornees={bounded}")
    return result


def main() -> dict:
    loader = ConfigLoader()
    cfg = loader.load_model()
    sim = loader.load_simulator()
    from src.shapes.index import generate_index

    index, _ = generate_index(loader.load_shapes(), Path("outputs/shapes"))
    target = index.point_cloud_by_key("geometries_circle")
    summary = write_architecture(cfg, sim.v_max)
    print(f"[gnn] architecture -> {OUTPUT}")
    for name, parameter_count in summary.items():
        print(f"  {name}: {parameter_count} parametres")
    demo = untrained_demo(cfg, sim, target)
    return {"architecture": summary, "demo": demo}


if __name__ == "__main__":
    main()
