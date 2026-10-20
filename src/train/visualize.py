"""Visualise un épisode avec le modèle entraîné sur le cercle."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from src.models import GNNController

from .training import TrainingConfig, adjacency_from_positions, local_target_features


def main() -> None:
    """Exporte un GIF et l'état final du contrôleur entraîné."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training-circle.yaml")
    parser.add_argument("--model", default="results/training-circle/circle-best.pt")
    parser.add_argument("--target", default="data/shapes/geometry/geometry_circle.npy")
    parser.add_argument("--output", default="results/training-circle")
    arguments = parser.parse_args()
    config = TrainingConfig.from_yaml(arguments.config)
    target = np.load(arguments.target) * config.target_radius + np.asarray(config.arena_size) / 2
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    torch.manual_seed(config.seed)
    model = GNNController(config.hidden_dim, config.message_passing_steps, 1, config.max_speed)
    model.load_state_dict(torch.load(arguments.model, map_location="cpu", weights_only=True))
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1)
    positions = torch.rand((1, config.robot_count, 2), generator=generator) * torch.tensor(config.arena_size)
    velocities = torch.zeros_like(positions)
    history = [positions[0].numpy().copy()]
    center = torch.tensor(config.arena_size) / 2
    with torch.no_grad():
        for _ in range(config.episode_steps):
            adjacency = adjacency_from_positions(positions, config.communication_radius)
            geometry = local_target_features(positions, target_tensor, config.target_radius) if config.use_target_geometry else None
            velocities = model(positions, velocities, adjacency, torch.tensor([0]), positions - center, geometry)
            positions = torch.minimum(torch.maximum(positions + velocities * config.time_step, torch.zeros_like(center)), center * 2)
            history.append(positions[0].numpy().copy())
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots()
    axis.scatter(target[:, 0], target[:, 1], color="black", marker="x", label="cible")
    robots = axis.scatter([], [], color="tab:blue", label="robots")
    axis.set(xlim=(0, config.arena_size[0]), ylim=(0, config.arena_size[1]), aspect="equal", title="Formation du cercle")
    axis.legend(loc="upper right")

    def update(frame: int):
        robots.set_offsets(history[frame])
        return (robots,)

    animation = FuncAnimation(figure, update, frames=len(history), interval=120, blit=True)
    animation.save(output / "circle-formation.gif", writer="pillow", fps=8)
    robots.set_offsets(history[-1])
    figure.savefig(output / "circle-final.png", dpi=150, bbox_inches="tight")
    print(f"frames={len(history)}; output={output}")


if __name__ == "__main__":
    main()
