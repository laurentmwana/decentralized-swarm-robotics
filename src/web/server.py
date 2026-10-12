"""Expose en HTTP les trajectoires du modèle GNN entraîné.

Lancer avec ``python -m src.web.server`` puis appeler, par exemple,
``http://127.0.0.1:8000/api/simulate?target=alphabet_A``.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import yaml

from src.models import GNNController
from src.train.training import TrainingConfig, adjacency_from_positions, local_target_features


DEFAULT_CONFIG = "configs/curriculum.yaml"
DEFAULT_MODEL = "results/curriculum/numbers/model.pt"
DEFAULT_STAGE = "numbers"


@lru_cache(maxsize=4)
def _load_experiment(config_path: str, model_path: str, stage_name: str) -> tuple[TrainingConfig, list[str], list[np.ndarray], GNNController]:
    """Charge une campagne de curriculum et son modèle une seule fois."""
    with Path(config_path).open(encoding="utf-8") as stream:
        curriculum = yaml.safe_load(stream)
    stage = next((item for item in curriculum["stages"] if item["name"] == stage_name), None)
    if stage is None:
        raise ValueError(f"Niveau inconnu : {stage_name}")
    values = dict(curriculum["base"])
    values["arena_size"] = tuple(values["arena_size"])
    config = TrainingConfig(iterations=1, **values)
    target_paths = list(stage["targets"])
    targets = [np.load(path) * config.target_radius + np.asarray(config.arena_size) / 2 for path in target_paths]
    model = GNNController(config.hidden_dim, config.message_passing_steps, len(targets), config.max_speed)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    return config, target_paths, targets, model.eval()


def available_targets(config_path: str = DEFAULT_CONFIG, model_path: str = DEFAULT_MODEL, stage_name: str = DEFAULT_STAGE) -> list[dict[str, str]]:
    """Retourne les identifiants des formes que le modèle de ce niveau connaît."""
    _, paths, _, _ = _load_experiment(config_path, model_path, stage_name)
    return [{"id": Path(path).stem, "source": path} for path in paths]


def simulate_target(
    target_name: str,
    seed: int = 142,
    config_path: str = DEFAULT_CONFIG,
    model_path: str = DEFAULT_MODEL,
    stage_name: str = DEFAULT_STAGE,
) -> dict[str, object]:
    """Exécute un épisode GNN et renvoie toutes les positions pour une animation Web."""
    config, paths, targets, model = _load_experiment(config_path, model_path, stage_name)
    target_ids = {Path(path).stem: index for index, path in enumerate(paths)}
    if target_name not in target_ids:
        choices = ", ".join(sorted(target_ids))
        raise ValueError(f"Forme inconnue : {target_name}. Choix possibles : {choices}")
    target_id = target_ids[target_name]
    target = targets[target_id]
    generator = torch.Generator().manual_seed(seed)
    arena = torch.tensor(config.arena_size, dtype=torch.float32)
    positions = torch.rand((1, config.robot_count, 2), generator=generator) * arena
    velocities = torch.zeros_like(positions)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    center = arena / 2
    history = [positions[0].tolist()]
    with torch.no_grad():
        for _ in range(config.episode_steps):
            adjacency = adjacency_from_positions(positions, config.communication_radius)
            geometry = local_target_features(positions, target_tensor, config.target_radius) if config.use_target_geometry else None
            velocities = model(positions, velocities, adjacency, torch.tensor([target_id]), positions - center, geometry)
            positions = torch.minimum(torch.maximum(positions + velocities * config.time_step, torch.zeros_like(arena)), arena)
            history.append(positions[0].tolist())
    return {
        "model": {"path": model_path, "stage": stage_name, "type": "GNNController"},
        "target": {"id": target_name, "source": paths[target_id], "points": target.tolist()},
        "simulation": {
            "seed": seed,
            "robot_count": config.robot_count,
            "episode_steps": config.episode_steps,
            "arena_size": list(config.arena_size),
            "communication_radius": config.communication_radius,
            "time_step": config.time_step,
        },
        "frames": history,
    }


class _Handler(BaseHTTPRequestHandler):
    """Gestionnaire minimal, uniquement pour les données de démonstration locales."""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                payload: object = {"status": "ok", "model": DEFAULT_MODEL}
            elif parsed.path == "/api/targets":
                payload = {"targets": available_targets()}
            elif parsed.path == "/api/simulate":
                target = query.get("target", [""])[0]
                if not target:
                    raise ValueError("Le paramètre ?target= est obligatoire.")
                seed = int(query.get("seed", ["142"])[0])
                payload = simulate_target(target, seed)
            else:
                self._send_json({"error": "Route inconnue."}, HTTPStatus.NOT_FOUND)
                return
        except (FileNotFoundError, ValueError, StopIteration) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(payload, HTTPStatus.OK)

    def _send_json(self, payload: object, status: HTTPStatus) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Évite d'encombrer la sortie pendant une démonstration."""


def main() -> None:
    """Démarre l'API locale de visualisation."""
    parser = argparse.ArgumentParser(description="API locale pour la démo GNN.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    arguments = parser.parse_args()
    server = ThreadingHTTPServer((arguments.host, arguments.port), _Handler)
    print(f"API de démonstration : http://{arguments.host}:{arguments.port}/api/targets")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAPI arrêtée.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
