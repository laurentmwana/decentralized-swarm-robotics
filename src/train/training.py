"""Entraînement direct du GNN à travers une simulation différentiable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import Tensor

from src.models import GNNController


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration d'un apprentissage réduit, traçable et reproductible."""

    seed: int
    batch_size: int
    iterations: int
    episode_steps: int
    learning_rate: float
    robot_count: int
    arena_size: tuple[float, float]
    communication_radius: float
    max_speed: float
    time_step: float
    safety_distance: float
    target_radius: float
    shape_loss_weight: float
    collision_loss_weight: float
    motion_loss_weight: float
    hidden_dim: int = 32
    message_passing_steps: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Charge une configuration d'apprentissage."""
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        values["arena_size"] = tuple(values["arena_size"])
        return cls(**values)


def adjacency_from_positions(positions: Tensor, radius: float) -> Tensor:
    """Construit le graphe de communication dans un lot d'états."""
    distances = torch.cdist(positions, positions)
    adjacency = distances <= radius
    diagonal = torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
    return adjacency & ~diagonal


def chamfer_loss(positions: Tensor, target: Tensor) -> Tensor:
    """Distance de Chamfer symétrique entre robots et points de la forme."""
    distances = torch.cdist(positions, target).square()
    return distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()


def collision_loss(positions: Tensor, safety_distance: float) -> Tensor:
    """Pénalise les paires de robots trop proches."""
    distances = torch.cdist(positions, positions)
    mask = ~torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
    penalties = torch.relu(safety_distance - distances).square()
    return penalties.masked_select(mask.expand_as(penalties)).mean()


def rollout(model: GNNController, positions: Tensor, target_ids: Tensor, config: TrainingConfig) -> tuple[Tensor, Tensor]:
    """Déroule le simulateur différentiable et retourne positions et commandes."""
    velocities = torch.zeros_like(positions)
    arena = torch.tensor(config.arena_size, dtype=positions.dtype, device=positions.device)
    center = arena / 2
    all_commands: list[Tensor] = []
    for _ in range(config.episode_steps):
        adjacency = adjacency_from_positions(positions, config.communication_radius)
        commands = model(positions, velocities, adjacency, target_ids, positions - center)
        velocities = commands
        positions = torch.minimum(torch.maximum(positions + velocities * config.time_step, torch.zeros_like(arena)), arena)
        all_commands.append(commands)
    return positions, torch.stack(all_commands, dim=1)


def _compatible_model(config: TrainingConfig, target_count: int, initial_state: dict[str, Tensor] | None = None) -> GNNController:
    """Crée un modèle et recopie les paramètres compatibles d'un niveau précédent."""
    model = GNNController(config.hidden_dim, config.message_passing_steps, target_count=target_count, max_speed=config.max_speed)
    if initial_state is not None:
        current = model.state_dict()
        compatible = {name: value for name, value in initial_state.items() if name in current and current[name].shape == value.shape}
        model.load_state_dict(compatible, strict=False)
        if "target_embedding.weight" in initial_state:
            previous = initial_state["target_embedding.weight"]
            model.target_embedding.weight.data[: min(len(previous), target_count)] = previous[:target_count]
    return model


def train_targets(
    config: TrainingConfig,
    target_points: list[np.ndarray],
    output_directory: str | Path,
    initial_state: dict[str, Tensor] | None = None,
) -> tuple[GNNController, list[float], dict[int, float]]:
    """Entraîne un contrôleur conditionné sur plusieurs formes cibles."""
    if not target_points or any(points.shape != (config.robot_count, 2) for points in target_points):
        raise ValueError("Chaque forme cible doit contenir robot_count points 2D.")
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device("cpu")
    targets = torch.tensor(np.stack(target_points), dtype=torch.float32, device=device) * config.target_radius
    targets += torch.tensor(config.arena_size, dtype=torch.float32, device=device) / 2
    model = _compatible_model(config, len(target_points), initial_state).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for _ in range(config.iterations):
        target_ids = torch.randint(len(target_points), (config.batch_size,), generator=generator, device=device)
        starts = torch.rand((config.batch_size, config.robot_count, 2), generator=generator, device=device)
        starts *= torch.tensor(config.arena_size, device=device)
        final_positions, commands = rollout(model, starts, target_ids, config)
        selected_targets = targets[target_ids]
        loss_shape = chamfer_loss(final_positions, selected_targets)
        loss_collision = collision_loss(final_positions, config.safety_distance)
        loss_motion = commands.square().mean()
        loss = config.shape_loss_weight * loss_shape + config.collision_loss_weight * loss_collision + config.motion_loss_weight * loss_motion
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    metrics: dict[int, float] = {}
    with torch.no_grad():
        for target_id in range(len(target_points)):
            starts = torch.rand((config.batch_size, config.robot_count, 2), generator=generator, device=device)
            starts *= torch.tensor(config.arena_size, device=device)
            final_positions, _ = rollout(model, starts, torch.full((config.batch_size,), target_id, device=device), config)
            metrics[target_id] = float(chamfer_loss(final_positions, targets[target_id].unsqueeze(0).expand(config.batch_size, -1, -1)))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
    (output / "metrics.yaml").write_text(yaml.safe_dump({"loss_by_target": metrics}, sort_keys=False), encoding="utf-8")
    np.savetxt(output / "loss.csv", np.asarray(losses), delimiter=",")
    figure, axis = plt.subplots()
    axis.plot(losses)
    axis.set(xlabel="itération", ylabel="perte", title="Apprentissage progressif")
    figure.savefig(output / "loss.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return model, losses, metrics


def train_circle(config: TrainingConfig, target_points: np.ndarray, output_directory: str | Path) -> tuple[GNNController, list[float]]:
    """Entraîne le contrôleur sur le cercle et sauvegarde modèle, courbe et config."""
    model, losses, _ = train_targets(config, [target_points], output_directory)
    torch.save(model.state_dict(), Path(output_directory) / "circle-best.pt")
    return model, losses
