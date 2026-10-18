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

from src.models import GNNController
from src.sim.rules import local_gathering_and_repulsion
from src.train.training import adjacency_from_positions, local_target_features

from .metrics import collision_count, nearest_neighbor_std, symmetric_chamfer_distance


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
) -> np.ndarray:
    pos = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
    vel = torch.tensor(velocities, dtype=torch.float32).unsqueeze(0)
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
    for step in range(config.episode_steps):
        if step == config.episode_steps // 2 and failure_fraction:
            count = max(1, round(robot_count * failure_fraction))
            active[rng.choice(np.flatnonzero(active), size=count, replace=False)] = False
        commands = _commands_gnn(model, positions, velocities, active, target, config, target_id) if method == "gnn" else _commands_gathering(positions, active, config)
        commands[~active] = 0
        velocities = commands
        positions = np.clip(positions + velocities * config.time_step, (0, 0), config.arena_size)
        total_collisions += collision_count(positions[active], config.safety_distance)
        quality = symmetric_chamfer_distance(positions[active], target)
        if quality <= config.success_threshold and convergence_step == config.episode_steps:
            convergence_step = step + 1
    final = positions[active]
    possible_pairs = robot_count * (robot_count - 1) / 2
    return {"method": method, "robots": robot_count, "failure_fraction": failure_fraction, "seed": seed, "shape_distance": symmetric_chamfer_distance(final, target), "regularity_std": nearest_neighbor_std(final), "collisions": total_collisions, "collision_rate": total_collisions / (config.episode_steps * possible_pairs), "convergence_step": convergence_step, "success": int(convergence_step < config.episode_steps)}


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
    for scale in config.scale_factors:
        count = config.robot_count * scale
        target = np.resize(base_target, (count, 2))
        for failure in config.failure_fractions:
            for episode in range(config.episodes):
                for method in ("gnn", "gathering"):
                    rows.append(simulate_episode(model, method, target, count, failure, config.seed + episode, config))
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary: list[dict[str, float | int | str]] = []
    for method in ("gnn", "gathering"):
        for robots in sorted({int(row["robots"]) for row in rows}):
            for failure in config.failure_fractions:
                group = [row for row in rows if row["method"] == method and row["robots"] == robots and row["failure_fraction"] == failure]
                summary.append({"method": method, "robots": robots, "failure_fraction": failure, "episodes": len(group), "shape_distance_mean": float(np.mean([row["shape_distance"] for row in group])), "shape_distance_std": float(np.std([row["shape_distance"] for row in group])), "collision_rate_mean": float(np.mean([row["collision_rate"] for row in group])), "success_rate": float(np.mean([row["success"] for row in group]))})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    conditions = (("failures", "failure_fraction", "Pannes (20 robots)", lambda row: row["robots"] == config.robot_count and row["failure_fraction"]), ("scale", "robots", "Passage à l'échelle (sans panne)", lambda row: row["failure_fraction"] == 0.0 and row["robots"]))
    for axis_name, group_key, title, include in conditions:
        figure, axis = plt.subplots()
        for method in ("gnn", "gathering"):
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
