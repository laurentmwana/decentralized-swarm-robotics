"""Expose en HTTP les trajectoires et les mesures du modèle GNN entraîné.

Lancer avec ``python -m src.web.server`` puis appeler, par exemple,
``http://127.0.0.1:8000/api/simulate?target=alphabet_A``.

L'API reproduit exactement le protocole expérimental du mémoire :
- les commandes proviennent de ``src.eval.run`` (même graphe recalculé à
  chaque pas, même règle de regroupement, même bornage de vitesse) ;
- les mesures suivent la définition du chapitre 3 : distance de Chamfer
  symétrique non carrée sur les robots actifs, taux de collisions normalisé
  par les paires et les pas, régularité du plus proche voisin, succès
  ``t_conv <= T-1`` pour un seuil ``tau = 0,5`` ;
- les scénarios de panne retirent des robots à mi-épisode : ils sont figés,
  exclus du graphe, du comptage des collisions et de la mesure finale ;
- le passage à l'échelle duplique les points de la cible avec ``np.resize``,
  comme dans le protocole pilote (20, 40 ou 100 robots).
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

from src.eval.metrics import collision_count, nearest_neighbor_std, symmetric_chamfer_distance
from src.eval.run import EvaluationConfig, _commands_gathering, _commands_gnn
from src.models import GNNController
from src.train.training import TrainingConfig


DEFAULT_CONFIG = "configs/curriculum.yaml"
DEFAULT_MODEL = "results/curriculum-v2/numbers/model.pt"
DEFAULT_STAGE = "numbers"

SUCCESS_THRESHOLD = 0.5
FAILURE_FRACTIONS = (0.0, 0.05, 0.10, 0.20)
SCALE_ROBOT_COUNTS = (20, 40, 100)
METHODS = ("gnn", "gathering", "manual", "gnn_no_comm")


@lru_cache(maxsize=4)
def _load_experiment(config_path: str, model_path: str, stage_name: str) -> tuple[TrainingConfig, EvaluationConfig, list[str], list[np.ndarray], GNNController]:
    """Charge une campagne de curriculum, son modèle et la config d'évaluation."""
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
    evaluation = EvaluationConfig(
        seed=config.seed,
        episodes=1,
        episode_steps=config.episode_steps,
        robot_count=config.robot_count,
        arena_size=config.arena_size,
        communication_radius=config.communication_radius,
        max_speed=config.max_speed,
        time_step=config.time_step,
        safety_distance=config.safety_distance,
        target_radius=config.target_radius,
        failure_fractions=FAILURE_FRACTIONS,
        scale_factors=tuple(count // config.robot_count for count in SCALE_ROBOT_COUNTS),
        success_threshold=SUCCESS_THRESHOLD,
        model_path=model_path,
        target_path="",
        use_target_geometry=config.use_target_geometry,
    )
    return config, evaluation, target_paths, targets, model.eval()


def available_targets(config_path: str = DEFAULT_CONFIG, model_path: str = DEFAULT_MODEL, stage_name: str = DEFAULT_STAGE) -> list[dict[str, str]]:
    """Retourne les identifiants des formes que le modèle de ce niveau connaît."""
    _, _, paths, _, _ = _load_experiment(config_path, model_path, stage_name)
    return [{"id": Path(path).stem, "source": path} for path in paths]


def _validate(method: str, failure: float, robots: int) -> None:
    if method not in METHODS:
        raise ValueError(f"Méthode inconnue : {method}. Choix possibles : {', '.join(METHODS)}.")
    if failure not in FAILURE_FRACTIONS:
        choices = ", ".join(str(value) for value in FAILURE_FRACTIONS)
        raise ValueError(f"Fraction de panne invalide : {failure}. Choix possibles : {choices}.")
    if robots not in SCALE_ROBOT_COUNTS:
        choices = ", ".join(str(count) for count in SCALE_ROBOT_COUNTS)
        raise ValueError(f"Nombre de robots invalide : {robots}. Choix possibles : {choices} (protocole pilote).")


def simulate_target(
    target_name: str,
    seed: int = 142,
    method: str = "gnn",
    failure: float = 0.0,
    robots: int = 20,
    config_path: str = DEFAULT_CONFIG,
    model_path: str = DEFAULT_MODEL,
    stage_name: str = DEFAULT_STAGE,
) -> dict[str, object]:
    """Exécute un épisode du protocole pilote et renvoie positions et mesures."""
    _validate(method, failure, robots)
    config, evaluation, paths, targets, model = _load_experiment(config_path, model_path, stage_name)
    target_ids = {Path(path).stem: index for index, path in enumerate(paths)}
    if target_name not in target_ids:
        choices = ", ".join(sorted(target_ids))
        raise ValueError(f"Forme inconnue : {target_name}. Choix possibles : {choices}")
    target_id = target_ids[target_name]
    base_target = targets[target_id]
    target = base_target if robots == config.robot_count else np.resize(base_target, (robots, 2))

    rng = np.random.default_rng(seed)
    arena = np.asarray(config.arena_size)
    positions = rng.uniform((0, 0), arena, size=(robots, 2))
    velocities = np.zeros_like(positions)
    active = np.ones(robots, dtype=bool)
    retired: list[int] = []
    failure_step = config.episode_steps // 2
    convergence_step = config.episode_steps
    total_collisions = 0
    history = [positions.tolist()]
    with torch.no_grad():
        for step in range(config.episode_steps):
            if step == failure_step and failure:
                count = max(1, round(robots * failure))
                chosen = rng.choice(np.flatnonzero(active), size=count, replace=False)
                active[chosen] = False
                retired.extend(int(robot) for robot in chosen)
            if method == "gnn":
                commands = _commands_gnn(model, positions, velocities, active, target, evaluation, target_id)
            else:
                commands = _commands_gathering(positions, active, evaluation)
            commands[~active] = 0
            velocities = commands
            positions = np.clip(positions + velocities * config.time_step, (0, 0), arena)
            total_collisions += collision_count(positions[active], config.safety_distance)
            quality = symmetric_chamfer_distance(positions[active], target)
            if quality <= SUCCESS_THRESHOLD and convergence_step == config.episode_steps:
                convergence_step = step + 1
            history.append(positions.tolist())

    final = positions[active]
    possible_pairs = robots * (robots - 1) / 2
    metrics = {
        "shape_distance": symmetric_chamfer_distance(final, target),
        "regularity_std": nearest_neighbor_std(final),
        "collisions": total_collisions,
        "collision_rate": total_collisions / (config.episode_steps * possible_pairs),
        "convergence_step": convergence_step,
        "success": int(convergence_step < config.episode_steps),
    }
    return {
        "model": {"path": model_path, "stage": stage_name, "type": "GNNController"},
        "target": {"id": target_name, "source": paths[target_id], "points": target.tolist()},
        "simulation": {
            "seed": seed,
            "method": method,
            "robot_count": robots,
            "failure_fraction": failure,
            "failure_step": failure_step if failure else None,
            "retired": retired,
            "episode_steps": config.episode_steps,
            "arena_size": list(config.arena_size),
            "communication_radius": config.communication_radius,
            "time_step": config.time_step,
            "max_speed": config.max_speed,
            "safety_distance": config.safety_distance,
            "success_threshold": SUCCESS_THRESHOLD,
        },
        "metrics": metrics,
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
                method = query.get("method", ["gnn"])[0]
                failure = float(query.get("failure", ["0"])[0])
                robots = int(query.get("robots", ["20"])[0])
                payload = simulate_target(target, seed, method, failure, robots)
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
