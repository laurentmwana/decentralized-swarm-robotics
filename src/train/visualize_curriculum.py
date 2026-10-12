"""Planche de validation visuelle du dernier niveau du curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.models import GNNController

from .training import TrainingConfig, adjacency_from_positions, local_target_features


def main() -> None:
    """Exporte une planche cible/robots après le dernier niveau du curriculum."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curriculum.yaml")
    parser.add_argument("--stage", default="numbers")
    parser.add_argument("--model", default="results/curriculum/numbers/model.pt")
    parser.add_argument("--output", default="results/curriculum/numbers/contact-sheet.png")
    arguments = parser.parse_args()
    with Path(arguments.config).open(encoding="utf-8") as stream:
        curriculum = yaml.safe_load(stream)
    stage = next(item for item in curriculum["stages"] if item["name"] == arguments.stage)
    base = dict(curriculum["base"])
    base["arena_size"] = tuple(base["arena_size"])
    config = TrainingConfig(iterations=1, **base)
    target_paths = stage["targets"]
    targets = [np.load(path) * config.target_radius + np.asarray(config.arena_size) / 2 for path in target_paths]
    model = GNNController(config.hidden_dim, config.message_passing_steps, len(targets), config.max_speed)
    model.load_state_dict(torch.load(arguments.model, map_location="cpu", weights_only=True))
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 99)
    columns = 4
    figure, axes = plt.subplots((len(targets) + columns - 1) // columns, columns, figsize=(12, 9))
    for target_id, (axis, target) in enumerate(zip(axes.flat, targets)):
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        positions = torch.rand((1, config.robot_count, 2), generator=generator) * torch.tensor(config.arena_size)
        velocities = torch.zeros_like(positions)
        center = torch.tensor(config.arena_size) / 2
        with torch.no_grad():
            for _ in range(config.episode_steps):
                adjacency = adjacency_from_positions(positions, config.communication_radius)
                geometry = local_target_features(positions, target_tensor, config.target_radius) if config.use_target_geometry else None
                velocities = model(positions, velocities, adjacency, torch.tensor([target_id]), positions - center, geometry)
                positions = torch.minimum(torch.maximum(positions + velocities * config.time_step, torch.zeros_like(center)), center * 2)
        axis.scatter(target[:, 0], target[:, 1], color="black", marker="x", s=25)
        axis.scatter(positions[0, :, 0], positions[0, :, 1], color="tab:blue", s=25)
        axis.set(title=Path(target_paths[target_id]).stem, xlim=(0, 10), ylim=(0, 10), aspect="equal")
    for axis in axes.flat[len(targets):]:
        axis.axis("off")
    figure.tight_layout()
    Path(arguments.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=150)
    print(arguments.output)


if __name__ == "__main__":
    main()
