"""FastAPI pour la visualisation et simulation du modèle GNN entraîné.

Lancer avec ``python -m src.web.server`` puis accéder à :
- Swagger UI (auto) : http://127.0.0.1:8000/docs
- ReDoc (auto)     : http://127.0.0.1:8000/redoc
- Health check     : http://127.0.0.1:8000/api/health
- Liste des cibles : http://127.0.0.1:8000/api/targets
- Simulation       : http://127.0.0.1:8000/api/simulate

L'API reproduit exactement le protocole expérimental du mémoire :
- Les commandes proviennent de ``src.eval.run`` (même graphe recalculé à
  chaque pas, même règle de regroupement, même bornage de vitesse)
- Les mesures suivent la définition du chapitre 3 : distance de Chamfer
  symétrique non carrée sur les robots actifs, taux de collisions normalisé
  par les paires et les pas, régularité du plus proche voisin, succès
  ``t_conv <= T-1`` pour un seuil ``tau = 0,5``
- Les scénarios de panne retirent des robots à mi-épisode : ils sont figés,
  exclus du graphe, du comptage des collisions et de la mesure finale
- Le passage à l'échelle duplique les points de la cible avec ``np.resize``,
  comme dans le protocole pilote (20, 40 ou 100 robots)
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.eval.metrics import collision_count, nearest_neighbor_std, symmetric_chamfer_distance
from src.eval.run import EvaluationConfig, _commands_gathering, _commands_gnn
from src.models import GNNController
from src.train.training import TrainingConfig


# Configuration par défaut
DEFAULT_CONFIG = "configs/curriculum.yaml"
DEFAULT_MODEL = "results/curriculum-v2/numbers/model.pt"
DEFAULT_STAGE = "numbers"

SUCCESS_THRESHOLD = 0.5
FAILURE_FRACTIONS = (0.0, 0.05, 0.10, 0.20)
SCALE_ROBOT_COUNTS = (20, 40, 100)
METHODS = ("gnn", "gathering", "manual", "gnn_no_comm")


# --- Modèles Pydantic pour la documentation OpenAPI ---

class ModelInfo(BaseModel):
    path: str
    stage: str
    type: str


class TargetInfo(BaseModel):
    id: str
    source: str
    points: List[List[float]]


class SimulationInfo(BaseModel):
    seed: int
    method: str
    robot_count: int
    failure_fraction: float
    failure_step: Optional[int]
    retired: List[int]
    episode_steps: int
    arena_size: List[float]
    communication_radius: float
    time_step: float
    max_speed: float
    safety_distance: float
    success_threshold: float


class MetricsInfo(BaseModel):
    shape_distance: float
    regularity_std: float
    collisions: int
    collision_rate: float
    convergence_step: int
    success: int


class SimulateResponse(BaseModel):
    model: ModelInfo
    target: TargetInfo
    simulation: SimulationInfo
    metrics: MetricsInfo
    frames: List[List[List[float]]]


class TargetsResponse(BaseModel):
    targets: List[Dict[str, str]]


class HealthResponse(BaseModel):
    status: str
    model: str


class ErrorResponse(BaseModel):
    error: str


# --- Instance FastAPI ---

app = FastAPI(
    title="Swarm GNN API",
    version="1.0.0",
    description=(
        "API pour la simulation et visualisation d'essaims de robots "
        "contrôlés par un réseau de neurones graphiques (GNN). "
        "L'API reproduit le protocole expérimental du mémoire."
    ),
    contact={"name": "Swarm GNN Project"},
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS pour permettre l'accès depuis un navigateur
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=4)
def _load_experiment(
    config_path: str, model_path: str, stage_name: str
) -> Tuple[TrainingConfig, EvaluationConfig, List[str], List[np.ndarray], GNNController]:
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
    targets = [
        np.load(path) * config.target_radius + np.asarray(config.arena_size) / 2
        for path in target_paths
    ]
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


def available_targets(
    config_path: str = DEFAULT_CONFIG, model_path: str = DEFAULT_MODEL, stage_name: str = DEFAULT_STAGE
) -> List[Dict[str, str]]:
    """Retourne les identifiants des formes que le modèle de ce niveau connaît."""
    _, _, paths, _, _ = _load_experiment(config_path, model_path, stage_name)
    return [{"id": Path(path).stem, "source": path} for path in paths]


def _validate(method: str, failure: float, robots: int) -> None:
    if method not in METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Méthode inconnue : {method}. Choix possibles : {', '.join(METHODS)}."
        )
    if failure not in FAILURE_FRACTIONS:
        choices = ", ".join(str(value) for value in FAILURE_FRACTIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Fraction de panne invalide : {failure}. Choix possibles : {choices}."
        )
    if robots not in SCALE_ROBOT_COUNTS:
        choices = ", ".join(str(count) for count in SCALE_ROBOT_COUNTS)
        raise HTTPException(
            status_code=400,
            detail=f"Nombre de robots invalide : {robots}. Choix possibles : {choices} (protocole pilote)."
        )


def simulate_target(
    target_name: str,
    seed: int = 142,
    method: str = "gnn",
    failure: float = 0.0,
    robots: int = 20,
    config_path: str = DEFAULT_CONFIG,
    model_path: str = DEFAULT_MODEL,
    stage_name: str = DEFAULT_STAGE,
) -> Dict[str, Any]:
    """Exécute un épisode du protocole pilote et renvoie positions et mesures."""
    _validate(method, failure, robots)
    config, evaluation, paths, targets, model = _load_experiment(config_path, model_path, stage_name)
    target_ids = {Path(path).stem: index for index, path in enumerate(paths)}
    if target_name not in target_ids:
        choices = ", ".join(sorted(target_ids))
        raise HTTPException(
            status_code=400,
            detail=f"Forme inconnue : {target_name}. Choix possibles : {choices}"
        )
    target_id = target_ids[target_name]
    base_target = targets[target_id]
    target = base_target if robots == config.robot_count else np.resize(base_target, (robots, 2))

    rng = np.random.default_rng(seed)
    arena = np.asarray(config.arena_size)
    positions = rng.uniform((0, 0), arena, size=(robots, 2))
    velocities = np.zeros_like(positions)
    active = np.ones(robots, dtype=bool)
    retired: List[int] = []
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
            elif method == "gnn_no_comm":
                null_adjacency = torch.zeros((1, robots, robots), dtype=torch.bool)
                commands = _commands_gnn(model, positions, velocities, active, target, evaluation, target_id, null_adjacency)
            elif method == "manual":
                from src.sim import local_manual_with_shape_consensus
                from src.sim.core import SimulationConfig, SwarmSimulator
                sim_config = SimulationConfig(
                    seed=config.seed,
                    robot_count=robots,
                    arena_size=config.arena_size,
                    communication_radius=config.communication_radius,
                    max_speed=config.max_speed,
                    time_step=config.time_step,
                    episode_steps=config.episode_steps,
                    safety_distance=config.safety_distance,
                    measurement_noise_std=0.0,
                )
                simulator = SwarmSimulator(sim_config, positions=positions)
                simulator.active = active
                commands = local_manual_with_shape_consensus(simulator, target, attraction_gain=1.0, repulsion_gain=1.0)
            else:  # gathering
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
        "shape_distance": float(symmetric_chamfer_distance(final, target)),
        "regularity_std": float(nearest_neighbor_std(final)),
        "collisions": int(total_collisions),
        "collision_rate": float(total_collisions / (config.episode_steps * possible_pairs)),
        "convergence_step": int(convergence_step),
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


# --- Routes API ---

@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Vérification de santé de l'API",
    tags=["Monitoring"],
)
def health() -> HealthResponse:
    """Vérifie que l'API est opérationnelle et le modèle chargé."""
    return HealthResponse(status="ok", model=DEFAULT_MODEL)


@app.get(
    "/api/targets",
    response_model=TargetsResponse,
    summary="Liste des formes cibles disponibles",
    tags=["Targets"],
)
def targets() -> TargetsResponse:
    """Retourne toutes les formes que le modèle GNN courant peut former."""
    try:
        return TargetsResponse(targets=available_targets())
    except (FileNotFoundError, ValueError, StopIteration) as error:
        raise HTTPException(status_code=400, detail=str(error))


# Type aliases for Literal enums (query params come as strings)
MethodType = Literal["gnn", "gathering", "manual", "gnn_no_comm"]
FailureType = Literal["0", "0.0", "0.05", "0.1", "0.10", "0.2", "0.20"]
RobotsType = Literal["20", "40", "100"]


@app.get(
    "/api/simulate",
    response_model=SimulateResponse,
    summary="Lance une simulation d'essaim",
    tags=["Simulation"],
    responses={
        200: {"model": SimulateResponse, "description": "Résultat de la simulation"},
        400: {"model": ErrorResponse, "description": "Paramètres invalides ou cible inconnue"},
    },
)
def simulate(
    target: str = Query(
        ...,
        description="Identifiant de la forme cible (ex: alphabet_A, geometry_circle, digit_0, number_123)",
        examples=["alphabet_A", "geometry_circle", "digit_0", "number_123"],
    ),
    seed: int = Query(142, ge=0, description="Graine aléatoire pour la reproductibilité"),
    method: MethodType = Query("gnn", description="Méthode de contrôle"),
    failure: FailureType = Query(0.0, description="Fraction de robots mis en panne à mi-épisode"),
    robots: RobotsType = Query(20, description="Nombre de robots (passage à l'échelle)"),
) -> SimulateResponse:
    """
    Exécute un épisode complet du protocole pilote et retourne la trajectoire complète avec les métriques.
    
    **Paramètres :**
    - **target** (requis) : identifiant de la cible (ex: `alphabet_A`)
    - **seed** (optionnel, défaut: 142) : graine aléatoire
    - **method** (optionnel, défaut: `gnn`) : méthode de contrôle (`gnn`, `gathering`, `manual`, `gnn_no_comm`)
    - **failure** (optionnel, défaut: 0.0) : fraction de robots en panne (`0.0`, `0.05`, `0.10`, `0.20`)
    - **robots** (optionnel, défaut: 20) : nombre de robots (`20`, `40`, `100`)
    
    **Exemples :**
    - `GET /api/simulate?target=geometry_circle` — GNN standard, 20 robots
    - `GET /api/simulate?target=alphabet_A&failure=0.10` — 10% de pannes
    - `GET /api/simulate?target=digit_2&method=gathering` — Baseline gathering
    - `GET /api/simulate?target=number_123&robots=100` — Passage à l'échelle 100 robots
    - `GET /api/simulate?target=geometry_square&method=manual` — Contrôle manuel
    """
    try:
        # Convert string query params to numeric types
        failure_num = float(failure)
        robots_num = int(robots)
        result = simulate_target(target, seed, method, failure_num, robots_num)
        return SimulateResponse(**result)
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, StopIteration) as error:
        raise HTTPException(status_code=400, detail=str(error))


def main() -> None:
    """Démarre le serveur FastAPI avec Uvicorn."""
    parser = argparse.ArgumentParser(description="API FastAPI pour la démo GNN avec Swagger/ReDoc.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true", help="Rechargement automatique en développement")
    arguments = parser.parse_args()

    import uvicorn
    print(f"API Swarm GNN : http://{arguments.host}:{arguments.port}")
    print(f"Swagger UI    : http://{arguments.host}:{arguments.port}/docs")
    print(f"ReDoc         : http://{arguments.host}:{arguments.port}/redoc")
    print(f"Health check  : http://{arguments.host}:{arguments.port}/api/health")
    uvicorn.run(
        "src.web.server:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )


if __name__ == "__main__":
    main()