"""Differentiable training pipeline for GNN controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor

from src.swarm_gnn.config import TrainingConfig
from src.swarm_gnn.models import GNNController


def adjacency_from_positions(positions: Tensor, radius: float) -> Tensor:
    """Build communication graph from positions."""
    distances = torch.cdist(positions, positions)
    adjacency = distances <= radius
    diagonal = torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
    return adjacency & ~diagonal


def chamfer_loss(positions: Tensor, target: Tensor) -> Tensor:
    """Symmetric Chamfer distance between robots and target points."""
    distances = torch.cdist(positions, target).square()
    return distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()


def collision_loss(positions: Tensor, safety_distance: float) -> Tensor:
    """Penalize robot pairs closer than safety_distance."""
    distances = torch.cdist(positions, positions)
    mask = ~torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
    penalties = torch.relu(safety_distance - distances).square()
    return penalties.masked_select(mask.expand_as(penalties)).mean()


def rollout(
    model: GNNController,
    positions: Tensor,
    target_ids: Tensor,
    config: TrainingConfig,
    target_points: Tensor | None = None,
) -> Tuple[Tensor, Tensor]:
    """
    Roll out simulator for episode_steps.

    Returns:
        final_positions: [batch, robots, 2]
        all_commands: [batch, steps, robots, 2]
    """
    velocities = torch.zeros_like(positions)
    arena = torch.tensor(config.arena_size, dtype=positions.dtype, device=positions.device)
    all_commands: List[Tensor] = []

    for _ in range(config.episode_steps):
        adjacency = adjacency_from_positions(positions, config.communication_radius)
        commands = model(positions, velocities, adjacency, target_ids)
        velocities = commands
        positions = torch.clamp(
            positions + velocities * config.time_step,
            torch.zeros_like(arena),
            arena,
        )
        all_commands.append(commands)

    return positions, torch.stack(all_commands, dim=1)


def train_circle(
    config: TrainingConfig,
    target_points: np.ndarray,
    output_directory: Union[str, Path],
) -> Tuple[GNNController, List[float]]:
    """Train on single circle target."""
    model, losses, _, _ = train_targets(config, [target_points], output_directory)
    torch.save(model.state_dict(), Path(output_directory) / "circle-best.pt")
    return model, losses


def train_targets(
    config: TrainingConfig,
    target_points: List[np.ndarray],
    output_directory: Union[str, Path],
) -> Tuple[GNNController, List[float], dict, dict]:
    """
    Train controller on multiple targets with per-target loss tracking.

    Returns:
        model, loss_history, final_metrics, per_target_history
    """
    if not target_points or any(p.shape != (config.robot_count, 2) for p in target_points):
        raise ValueError("Chaque forme cible doit contenir robot_count points 2D.")

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device("cpu")

    # Prepare targets: normalize to [-1,1] then scale and center
    targets = torch.tensor(np.stack(target_points), dtype=torch.float32, device=device) * config.target_radius
    targets += torch.tensor(config.arena_size, dtype=torch.float32, device=device) / 2

    model = GNNController(
        config.hidden_dim,
        config.message_passing_steps,
        target_count=len(target_points),
        max_speed=config.max_speed,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: List[float] = []

    # Per-target evaluation every 10% of iterations
    eval_interval = max(1, config.iterations // 10)
    per_target_losses: dict[int, List[float]] = {i: [] for i in range(len(target_points))}

    for iteration in range(config.iterations):
        # Sample batch
        target_ids = torch.randint(len(target_points), (config.batch_size,), generator=generator, device=device)
        starts = torch.rand((config.batch_size, config.robot_count, 2), generator=generator, device=device)
        starts *= torch.tensor(config.arena_size, device=device)
        selected_targets = targets[target_ids]

        # Rollout
        final_positions, commands = rollout(model, starts, target_ids, config, selected_targets)

        # Losses
        loss_shape = chamfer_loss(final_positions, selected_targets)
        loss_collision = collision_loss(final_positions, config.safety_distance)
        loss_motion = commands.square().mean()

        loss = (
            config.shape_loss_weight * loss_shape
            + config.collision_loss_weight * loss_collision
            + config.motion_loss_weight * loss_motion
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.detach()))

        # Periodic per-target eval
        if iteration % eval_interval == 0 or iteration == config.iterations - 1:
            with torch.no_grad():
                for tid in range(len(target_points)):
                    eval_starts = torch.rand((min(config.batch_size, 4), config.robot_count, 2), generator=generator, device=device)
                    eval_starts *= torch.tensor(config.arena_size, device=device)
                    selected = targets[tid].unsqueeze(0).expand(eval_starts.shape[0], -1, -1)
                    final_pos, _ = rollout(model, eval_starts, torch.full((eval_starts.shape[0],), tid, device=device), config, selected)
                    t_loss = chamfer_loss(final_pos, targets[tid].unsqueeze(0).expand(eval_starts.shape[0], -1, -1))
                    per_target_losses[tid].append(float(t_loss.detach()))

    # Final metrics
    metrics = {}
    with torch.no_grad():
        for tid in range(len(target_points)):
            starts = torch.rand((config.batch_size, config.robot_count, 2), generator=generator, device=device)
            starts *= torch.tensor(config.arena_size, device=device)
            selected = targets[tid].unsqueeze(0).expand(config.batch_size, -1, -1)
            final_pos, _ = rollout(model, starts, torch.full((config.batch_size,), tid, device=device), config, selected)
            metrics[tid] = float(chamfer_loss(final_pos, targets[tid].unsqueeze(0).expand(config.batch_size, -1, -1)).detach())

    # Save
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    torch.save({"model_state": model.state_dict(), "best_loss": min(losses)}, output / "best.pt")
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    np.savetxt(output / "loss.csv", np.array(losses), delimiter=",")

    for tid, history in per_target_losses.items():
        if history:
            np.savetxt(output / f"loss_target_{tid}.csv", np.array(history), delimiter=",")

    # Plots
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set(xlabel="iteration", ylabel="loss", title="Training loss")
    fig.savefig(output / "loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if any(len(v) > 0 for v in per_target_losses.values()):
        fig, ax = plt.subplots()
        for tid, history in per_target_losses.items():
            if history:
                iters = [i * eval_interval for i in range(len(history))]
                ax.plot(iters, history, label=f"Target {tid}")
        ax.set(xlabel="iteration", ylabel="per-target loss")
        ax.legend()
        fig.savefig(output / "loss_per_target.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return model, losses, metrics, per_target_losses