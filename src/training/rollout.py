"""Simulateur différentiable (phase 5) : rollout complet en tenseurs, sans bibliothèque GNN.

Mêmes règles que le simulateur numpy (docs/decisions/02) : commande = sortie du modèle
déjà bornée par le décodeur, intégration pos += v·dt, bornes d'arène par clamp, graphe
de communication recalculé à chaque pas **sans auto-boucle** (le masque booléen ne
reçoit pas de gradient ; les gradients circulent par les positions et les commandes).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.common.config import SimulatorConfig


@dataclass(frozen=True)
class Rollout:
    positions: torch.Tensor  # (T+1, N, 2)
    commands: torch.Tensor  # (T, N, 2)

    @property
    def step(self) -> int:
        return self.commands.shape[0]


class DifferentiableSimulator:
    def __init__(self, model, cfg: SimulatorConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.half_arena = cfg.arena_size / 2.0

    def adjacency(self, positions: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(positions, positions)
        return (d <= self.cfg.r_c) & (d > 0.0)  # d > 0 exclut la diagonale : pas d'auto-boucle

    def target_vector(self, positions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Unitaire positions -> point de cible le plus proche (local, translation-invariant, différentiable)."""
        delta = target[None, :, :] - positions[:, None, :]
        distances = torch.cdist(positions, target)
        nearest = torch.argmin(distances, dim=1)
        toward = delta[torch.arange(positions.shape[0]), nearest]
        norms = torch.linalg.norm(toward, dim=1, keepdim=True)
        return toward / norms.clamp_min(1e-9)

    def rollout(
        self,
        initial_positions: torch.Tensor,
        shape_id: int,
        target: torch.Tensor | None = None,
        communication: bool = True,
    ) -> Rollout:
        step_count = self.cfg.horizon
        positions = initial_positions
        velocity = torch.zeros_like(positions)
        position_history = [positions]
        command_history = []
        for _ in range(step_count):
            n_robots = positions.shape[0]
            if communication:
                adjacency = self.adjacency(positions)
            else:
                adjacency = torch.zeros(n_robots, n_robots, dtype=torch.bool)
            # La géométrie reste absente pour la variante fondée sur l'identifiant seul.
            geometry = None
            if self.model.cfg.use_target_geometry and target is not None:
                geometry = self.target_vector(positions, target)
            command = self.model(velocity, positions, adjacency, torch.tensor(shape_id), geometry)
            positions = torch.clamp(positions + command * self.cfg.dt, -self.half_arena, self.half_arena)
            velocity = command
            position_history.append(positions)
            command_history.append(command)
        return Rollout(positions=torch.stack(position_history), commands=torch.stack(command_history))

    @staticmethod
    def initial_positions(n_robots: int, radius: float, generator: torch.Generator) -> torch.Tensor:
        """Disque uniforme de rayon `rayon`, même loi que le simulateur numpy (générateur torch séparé)."""
        angles = torch.rand(n_robots, generator=generator) * 2.0 * torch.pi
        radii = radius * torch.sqrt(torch.rand(n_robots, generator=generator))
        return torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)
