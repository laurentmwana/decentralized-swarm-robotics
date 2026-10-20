"""Evaluation metrics and episode simulation for pilot protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
import torch
import yaml

from src.swarm_gnn.config import EvaluationConfig, SimulatorConfig
from src.swarm_gnn.models import GNNController
from src.swarm_gnn.simulator import SwarmSimulator
from src.swarm_gnn.simulator.rules import local_gathering_and_repulsion, local_manual_with_shape_consensus
from src.swarm_gnn.training import adjacency_from_positions, local_target_features


def collision_count(positions: np.ndarray, safety_distance: float) -> int:
    """Count pairs of robots closer than safety_distance."""
    if len(positions) < 2:
        return 0
    offsets = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    mask = np.triu(distances < safety_distance, k=1)
    return int(mask.sum())


def symmetric_chamfer_distance(positions: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point sets."""
    if len(positions) == 0 or len(target) == 0:
        return float('inf')
    distances = np.linalg.norm(positions[:, None, :] - target[None, :, :], axis=-1)
    return distances.min(axis=1).mean() + distances.min(axis=0).mean()


def nearest_neighbor_std(positions: np.ndarray) -> float:
    """Standard deviation of nearest neighbor distances."""
    if len(positions) < 2:
        return 0.0
    offsets = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    return float(nearest.std())


def graph_connectivity(positions: np.ndarray, radius: float) -> float:
    """Fraction of robots in largest connected component."""
    if len(positions) == 0:
        return 0.0
    n = len(positions)
    offsets = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    adj = distances <= radius
    np.fill_diagonal(adj, False)

    # BFS for connected components
    visited = np.zeros(n, dtype=bool)
    max_component = 0
    for i in range(n):
        if not visited[i]:
            component = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                component.append(node)
                for j in range(n):
                    if adj[node, j] and not visited[j]:
                        visited[j] = True
                        stack.append(j)
            max_component = max(max_component, len(component))
    return max_component / n


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
    """Generate commands using GNN controller."""
    pos = torch.tensor(positions, dtype=torch.float32).unsqueeze(0)
    vel = torch.tensor(velocities, dtype=torch.float32).unsqueeze(0)

    if adjacency is None:
        adj = adjacency_from_positions(pos, config.communication_radius)
        active_t = torch.tensor(active, dtype=torch.bool).view(1, -1, 1) & torch.tensor(active, dtype=torch.bool).view(1, 1, -1)
        adjacency = adj & active_t

    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        geometry = local_target_features(pos, target_tensor, config.target_radius) if config.use_target_geometry else None
        center = torch.tensor(config.arena_size, dtype=torch.float32).view(1, 1, 2) / 2
        result = model(pos, vel, adjacency, torch.tensor([target_id]), pos - center, geometry)

    commands = result[0].numpy()
    commands[~active] = 0
    return commands


def _commands_gathering(positions: np.ndarray, active: np.ndarray, config: EvaluationConfig) -> np.ndarray:
    """Gathering baseline: mean neighbor position."""
    commands = np.zeros_like(positions)
    for i in np.flatnonzero(active):
        offsets = positions[active] - positions[i]
        distances = np.linalg.norm(offsets, axis=1)
        neighbors = (distances <= config.communication_radius) & (distances > 0)
        if neighbors.any():
            commands[i] = offsets[neighbors].mean(axis=0)

    norms = np.linalg.norm(commands, axis=1, keepdims=True)
    return commands * np.minimum(1.0, config.max_speed / np.maximum(norms, 1e-12))


def simulate_episode(
    model: GNNController,
    method: str,
    target: np.ndarray,
    robot_count: int,
    failure_fraction: float,
    seed: int,
    config: EvaluationConfig,
    target_id: int = 0,
) -> dict:
    """Run a single episode with given method and parameters."""
    rng = np.random.default_rng(seed)
    arena = np.asarray(config.arena_size)
    positions = rng.uniform((0, 0), arena, size=(robot_count, 2))
    velocities = np.zeros_like(positions)
    active = np.ones(robot_count, dtype=bool)
    retired: List[int] = []
    failure_step = config.episode_steps // 2
    convergence_step = config.episode_steps
    total_collisions = 0

    # Create SimulatorConfig for manual method
    sim_config = SimulatorConfig(
        seed=config.seed,
        robot_count=robot_count,
        arena_size=config.arena_size,
        communication_radius=config.communication_radius,
        max_speed=config.max_speed,
        time_step=config.time_step,
        episode_steps=config.episode_steps,
        safety_distance=config.safety_distance,
        measurement_noise_std=0.0,
    )

    for step in range(config.episode_steps):
        if step == failure_step and failure_fraction > 0:
            count = max(1, round(robot_count * failure_fraction))
            choices = rng.choice(np.flatnonzero(active), size=count, replace=False)
            active[choices] = False
            retired.extend(int(r) for r in choices)

        if method == "gnn":
            commands = _commands_gnn(model, positions, velocities, active, target, config, target_id)
        elif method == "gnn_no_comm":
            null_adj = torch.zeros((1, robot_count, robot_count), dtype=torch.bool)
            commands = _commands_gnn(model, positions, velocities, active, target, config, target_id, null_adj)
        elif method == "manual":
            sim = SwarmSimulator(sim_config, positions=positions)
            sim.active = active
            commands = local_manual_with_shape_consensus(sim, target, attraction_gain=1.0, repulsion_gain=1.0)
        else:  # gathering
            commands = _commands_gathering(positions, active, config)

        commands[~active] = 0
        velocities = commands
        positions = np.clip(positions + velocities * config.time_step, (0, 0), arena)
        total_collisions += collision_count(positions[active], config.safety_distance)

        quality = symmetric_chamfer_distance(positions[active], target)
        if quality <= config.success_threshold and convergence_step == config.episode_steps:
            convergence_step = step + 1

    final = positions[active]
    possible_pairs = robot_count * (robot_count - 1) / 2

    return {
        "method": method,
        "robots": robot_count,
        "failure_fraction": failure_fraction,
        "seed": seed,
        "shape_distance": float(symmetric_chamfer_distance(final, target)),
        "regularity_std": float(nearest_neighbor_std(final)),
        "collisions": int(total_collisions),
        "collision_rate": float(total_collisions / (config.episode_steps * possible_pairs)),
        "convergence_step": int(convergence_step),
        "success": int(convergence_step < config.episode_steps),
        "final_connectivity": float(graph_connectivity(final, config.communication_radius)),
        "retired": retired,
    }