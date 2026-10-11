"""Exécute un contrôleur GNN non entraîné dans le simulateur."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.sim.core import SimulationConfig, SwarmSimulator

from .gnn_controller import GNNController


def main() -> None:
    """Vérifie l'intégration modèle-simulateur pour une taille quelconque."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulator.yaml")
    parser.add_argument("--steps", type=int, default=10)
    arguments = parser.parse_args()
    config = SimulationConfig.from_yaml(arguments.config)
    torch.manual_seed(config.seed)
    simulator = SwarmSimulator(config)
    model = GNNController(hidden_dim=64, message_passing_steps=2, target_count=4, max_speed=config.max_speed).eval()
    for _ in range(arguments.steps):
        positions = torch.tensor(simulator.positions, dtype=torch.float32).unsqueeze(0)
        velocities = torch.tensor(simulator.velocities, dtype=torch.float32).unsqueeze(0)
        adjacency = torch.tensor(simulator.neighbor_matrix(), dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            commands = model(positions, velocities, adjacency, torch.tensor([0]))[0].numpy()
        simulator.step(commands)
    maximum = np.linalg.norm(simulator.velocities, axis=1).max()
    print(f"robots={config.robot_count}; steps={arguments.steps}; max_speed_observed={maximum:.6f}")


if __name__ == "__main__":
    main()
