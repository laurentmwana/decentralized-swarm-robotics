"""Expériences pilotes : baselines, pannes et passage à l'échelle."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.models import GNNController, NoCommunicationController
from src.sim import SwarmSimulator
from src.sim.rules import local_gathering_and_repulsion, local_manual_with_shape_consensus
from src.train.training import adjacency_from_positions, local_target_features

from .metrics import collision_count, nearest_neighbor_std, symmetric_chamfer_distance, graph_connectivity


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int
    episodes: int
    episode_steps: int
    robot_count: int
    arena_size: tuple[float, float]
    communication_radius: float
    max_speed: float
    time_step: float
    safety_distance: float
    target_radius: float
    failure_fractions: tuple[float, ...]
    scale_factors: tuple[int, ...]
    success_threshold: float
    model_path: str
    target_path: str
    use_target_geometry: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluationConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        for key in ("arena_size", "failure_fractions", "scale_factors"):
            data[key] = tuple(data[key])
        return cls(**data)


def _load_model(config: EvaluationConfig) -> GNNController:
    # Architecture must match the trained model (from curriculum config)
    model = GNNController(hidden_dim=32, message_passing_steps=2, target_count=11, max_speed=config.max_speed)
    model.load_state_dict(torch.load(config.model_path, map_location="cpu", weights_only=True))
    return model.eval()


def _commands_gnn(
    model: GNNController,
    positions: np.ndarray,
    velocities: np.ndarray,
    active: np.ndarray,
    target: np.ndarray,
    config: EvaluationConfig,
    target_id: int = 0,
    adjacency: np.ndarray | None = None,
) -> np.ndarray:
    pos = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
    vel = torch.tensor(velocities, dtype=torch.float32).unsqueeze(0)
    if adjacency is None:
        adjacency = adjacency_from_positions(pos, config.communication_radius) & torch.tensor(active).view(1, -1, 1) & torch.tensor(active).view(1, 1, -1)
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        geometry = local_target_features(pos, target_tensor, config.target_radius) if config.use_target_geometry else None
        result = model(pos, vel, adjacency, torch.tensor([target_id]), pos - torch.tensor(config.arena_size, dtype=torch.float32).view(1, 1, 2) / 2, geometry)
    return result[0].numpy()


def _commands_gathering(positions: np.ndarray, active: np.ndarray, config: EvaluationConfig) -> np.ndarray:
    commands = np.zeros_like(positions)
    for i in np.flatnonzero(active):
        offsets = positions[active] - positions[i]
        distances = np.linalg.norm(offsets, axis=1)
        neighbors = (distances <= config.communication_radius) & (distances > 0)
        if neighbors.any():
            commands[i] = offsets[neighbors].mean(axis=0)
    norms = np.linalg.norm(commands, axis=1, keepdims=True)
    return commands * np.minimum(1.0, config.max_speed / np.maximum(norms, 1e-12))




def simulate_episode(model: GNNController, method: str, target: np.ndarray, robot_count: int, failure_fraction: float, seed: int, config: EvaluationConfig, target_id: int = 0) -> dict[str, float | int | str]:
    """Exécute un épisode fixe, avec retrait de robots à mi-parcours."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform((0, 0), config.arena_size, size=(robot_count, 2))
    velocities = np.zeros_like(positions)
    active = np.ones(robot_count, dtype=bool)
    convergence_step = config.episode_steps
    total_collisions = 0
    
    # Create SimulationConfig for manual method (SwarmSimulator)
    from src.sim.core import SimulationConfig
    sim_config = SimulationConfig(
        seed=config.seed,
        robot_count=robot_count,
        arena_size=config.arena_size,
        communication_radius=config.communication_radius,
        max_speed=config.max_speed,
        time_step=config.time_step,
        episode_steps=config.episode_steps,
        safety_distance=config.safety_distance,
        measurement_noise_std=0.0
    )

    for step in range(config.episode_steps):
        if step == config.episode_steps // 2 and failure_fraction:
            count = max(1, round(robot_count * failure_fraction))
            active[rng.choice(np.flatnonzero(active), size=count, replace=False)] = False

        # Sélectionner la méthode de calcul des commandes
        if method == "gnn":
            commands = _commands_gnn(model, positions, velocities, active, target, config, target_id)
        elif method == "gnn_no_comm":
            # GNN sans communication : même modèle mais adjacence nulle (pas de messages)
            null_adjacency = torch.zeros((1, robot_count, robot_count), dtype=torch.bool)
            commands = _commands_gnn(model, positions, velocities, active, target, config, target_id, null_adjacency)
        elif method == "manual":
            # Utiliser la règle manuelle avec consigne de forme
            from src.sim import local_manual_with_shape_consensus
            simulator = SwarmSimulator(sim_config, positions=positions)
            simulator.active = active
            commands = local_manual_with_shape_consensus(simulator, target, attraction_gain=1.0, repulsion_gain=1.0)
        else:  # method == "gathering" ou tout autre valeur par défaut
            commands = _commands_gathering(positions, active, config)

        commands[~active] = 0
        velocities = commands
        positions = np.clip(positions + velocities * config.time_step, (0, 0), config.arena_size)
        total_collisions += collision_count(positions[active], config.safety_distance)
        quality = symmetric_chamfer_distance(positions[active], target)
        if quality <= config.success_threshold and convergence_step == config.episode_steps:
            convergence_step = step + 1
    final = positions[active]
    possible_pairs = robot_count * (robot_count - 1) / 2
    final_connectivity = graph_connectivity(final, config.communication_radius)
    return {"method": method, "robots": robot_count, "failure_fraction": failure_fraction, "seed": seed, "shape_distance": symmetric_chamfer_distance(final, target), "regularity_std": nearest_neighbor_std(final), "collisions": total_collisions, "collision_rate": total_collisions / (config.episode_steps * possible_pairs), "convergence_step": convergence_step, "success": int(convergence_step < config.episode_steps), "final_connectivity": final_connectivity}


def main() -> None:
    """Exécute le protocole pilote et produit le CSV et deux graphiques."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/evaluation-pilot.yaml")
    parser.add_argument("--output", default="results/evaluation-pilot")
    arguments = parser.parse_args()
    config = EvaluationConfig.from_yaml(arguments.config)
    model = _load_model(config)
    base_target = np.load(config.target_path) * config.target_radius + np.asarray(config.arena_size) / 2
    rows: list[dict[str, float | int | str]] = []
    methods = ("gnn", "gathering", "manual", "gnn_no_comm")
    for scale in config.scale_factors:
        count = config.robot_count * scale
        target = np.resize(base_target, (count, 2))
        for failure in config.failure_fractions:
            for episode in range(config.episodes):
                for method in methods:
                    rows.append(simulate_episode(model, method, target, count, failure, config.seed + episode, config))
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary: list[dict[str, float | int | str]] = []
    for method in ("gnn", "gathering", "manual", "gnn_no_comm"):
        for robots in sorted({int(row["robots"]) for row in rows}):
            for failure in config.failure_fractions:
                group = [row for row in rows if row["method"] == method and row["robots"] == robots and row["failure_fraction"] == failure]
                shape_distances = [row["shape_distance"] for row in group]
                collision_rates = [row["collision_rate"] for row in group]
                successes = [row["success"] for row in group]

                # Calculer la moyenne et l'écart-type
                shape_mean = float(np.mean(shape_distances))
                shape_std = float(np.std(shape_distances))
                collision_mean = float(np.mean(collision_rates))
                success_mean = float(np.mean(successes))

                # Calculer l'intervalle de confiance approximé à 95%
                # Pour des échantillons suffisamment grands (n >= 30), on utilise l'approximation normale : mean ± 1.96 * std/sqrt(n)
                # Pour des échantillons plus petits, on pourrait utiliser la loi de t de Student, mais pour simplifier
                # on utilise 2.0 comme facteur conservateur (légèrement plus large que l'intervalle exact)
                n = len(group)
                if n > 1:
                    std_err_shape = shape_std / np.sqrt(n)
                    std_err_collision = np.std(collision_rates) / np.sqrt(n)
                    std_err_success = np.std(successes) / np.sqrt(n)

                    # Facteur d'échelle pour intervalle de confiance à 95% (approximation)
                    # Pour n>=30, 1.96 est précis ; pour n<30, 2.0 est une approximation conservatrice
                    ci_factor = 2.0 if n < 30 else 1.96

                    shape_ci_low = shape_mean - ci_factor * std_err_shape
                    shape_ci_high = shape_mean + ci_factor * std_err_shape
                    collision_ci_low = collision_mean - ci_factor * std_err_collision
                    collision_ci_high = collision_mean + ci_factor * std_err_collision
                    success_ci_low = success_mean - ci_factor * std_err_success
                    success_ci_high = success_mean + ci_factor * std_err_success
                else:
                    # Pour n=1, pas d'erreur standard, l'IC est juste la moyenne
                    shape_ci_low = shape_ci_high = shape_mean
                    collision_ci_low = collision_ci_high = collision_mean
                    success_ci_low = success_ci_high = success_mean

                summary.append({"method": method, "robots": robots, "failure_fraction": failure, "episodes": len(group), "shape_distance_mean": shape_mean, "shape_distance_std": shape_std, "shape_distance_ci_low": shape_ci_low, "shape_distance_ci_high": shape_ci_high, "collision_rate_mean": collision_mean, "collision_rate_std": float(np.std(collision_rates)), "collision_rate_ci_low": collision_ci_low, "collision_rate_ci_high": collision_ci_high, "success_rate_mean": success_mean, "success_rate_std": float(np.std(successes)), "success_rate_ci_low": success_ci_low, "success_rate_ci_high": success_ci_high})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    conditions = (("failures", "failure_fraction", "Pannes (20 robots)", lambda row: row["robots"] == config.robot_count and row["failure_fraction"]), ("scale", "robots", "Passage à l'échelle (sans panne)", lambda row: row["failure_fraction"] == 0.0 and row["robots"]))
    for axis_name, group_key, title, include in conditions:
        figure, axis = plt.subplots()
        for method in ("gnn", "gathering", "manual", "gnn_no_comm"):
            values = sorted({row[group_key] for row in rows if row["method"] == method and include(row)})
            means = [np.mean([row["shape_distance"] for row in rows if row["method"] == method and include(row) and row[group_key] == value]) for value in values]
            axis.plot(values, means, marker="o", label=method)
        axis.set(xlabel=group_key, ylabel="distance de Chamfer", title=title)
        axis.legend()
        figure.savefig(output / f"{axis_name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
    print(f"episodes={len(rows)}; output={output}")


if __name__ == "__main__":
    main()
