"""Contrôleurs GNN invariants à l'ordre des robots."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MessagePassingLayer(nn.Module):
    """Un tour de messages avec agrégation moyenne des voisins."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, states: Tensor, positions: Tensor, adjacency: Tensor) -> Tensor:
        """Met à jour les états `[batch, robots, hidden]` à partir des voisins."""
        relative = positions[:, None, :, :] - positions[:, :, None, :]
        distances = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        edge_features = torch.cat((relative, distances), dim=-1)
        robot_count = states.shape[1]
        receiver = states[:, :, None, :].expand(-1, -1, robot_count, -1)
        sender = states[:, None, :, :].expand(-1, robot_count, -1, -1)
        messages = self.message(torch.cat((receiver, sender, edge_features), dim=-1))
        mask = adjacency.unsqueeze(-1).to(messages.dtype)
        total = (messages * mask).sum(dim=2)
        counts = mask.sum(dim=2).clamp_min(1.0)
        return self.update(torch.cat((states, total / counts), dim=-1))


class GNNController(nn.Module):
    """Politique partagée à messages locaux pour un nombre variable de robots."""

    def __init__(self, hidden_dim: int = 64, message_passing_steps: int = 2, target_count: int = 4, max_speed: float = 1.0) -> None:
        super().__init__()
        if message_passing_steps <= 0 or target_count <= 0 or max_speed <= 0:
            raise ValueError("Paramètres d'architecture invalides.")
        self.max_speed = max_speed
        self.target_embedding = nn.Embedding(target_count, hidden_dim)
        self.encoder = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.layers = nn.ModuleList(MessagePassingLayer(hidden_dim) for _ in range(message_passing_steps))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        adjacency: Tensor,
        target_ids: Tensor,
        goal_relative_positions: Tensor | None = None,
        target_geometry_features: Tensor | None = None,
    ) -> Tensor:
        """Retourne une commande bornée `[batch, robots, 2]`.

        La géométrie de la cible est une consigne partagée : chaque robot peut
        l'interroger à sa propre position, sans consulter des robots distants.
        """
        self._validate_inputs(positions, velocities, adjacency, target_ids)
        target = self.target_embedding(target_ids).unsqueeze(1).expand(-1, positions.shape[1], -1)
        relative_goal = torch.zeros_like(positions) if goal_relative_positions is None else goal_relative_positions
        if relative_goal.shape != positions.shape:
            raise ValueError("goal_relative_positions doit être [batch, robots, 2].")
        geometry = None if target_geometry_features is None else target_geometry_features
        if geometry is not None and geometry.shape != positions.shape[:2] + (3,):
            raise ValueError("target_geometry_features doit être [batch, robots, 3].")
        target_vector = relative_goal if geometry is None else geometry[..., :2]
        states = self.encoder(torch.cat((velocities, target_vector, target), dim=-1))
        for layer in self.layers:
            states = layer(states, positions, adjacency)
        return _limit_speed(self.decoder(states), self.max_speed)

    @staticmethod
    def _validate_inputs(positions: Tensor, velocities: Tensor, adjacency: Tensor, target_ids: Tensor) -> None:
        if positions.ndim != 3 or positions.shape[-1] != 2 or velocities.shape != positions.shape:
            raise ValueError("positions et velocities doivent être [batch, robots, 2].")
        if adjacency.shape != positions.shape[:2] + (positions.shape[1],):
            raise ValueError("adjacency doit être [batch, robots, robots].")
        if target_ids.shape != (positions.shape[0],):
            raise ValueError("target_ids doit être [batch].")


class NoCommunicationController(nn.Module):
    """Baseline apprise où chaque robot décide sans message de voisin."""

    def __init__(self, hidden_dim: int = 64, target_count: int = 4, max_speed: float = 1.0) -> None:
        super().__init__()
        self.max_speed = max_speed
        self.embedding = nn.Embedding(target_count, hidden_dim)
        self.network = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

    def forward(
        self,
        velocities: Tensor,
        target_ids: Tensor,
        goal_relative_positions: Tensor | None = None,
        target_geometry_features: Tensor | None = None,
    ) -> Tensor:
        """Retourne une commande bornée sans consultation du voisinage."""
        if velocities.ndim != 3 or velocities.shape[-1] != 2 or target_ids.shape != (velocities.shape[0],):
            raise ValueError("Entrées invalides pour la baseline sans communication.")
        target = self.embedding(target_ids).unsqueeze(1).expand(-1, velocities.shape[1], -1)
        relative_goal = torch.zeros_like(velocities) if goal_relative_positions is None else goal_relative_positions
        if relative_goal.shape != velocities.shape:
            raise ValueError("goal_relative_positions doit être [batch, robots, 2].")
        geometry = None if target_geometry_features is None else target_geometry_features
        if geometry is not None and geometry.shape != velocities.shape[:2] + (3,):
            raise ValueError("target_geometry_features doit être [batch, robots, 3].")
        target_vector = relative_goal if geometry is None else geometry[..., :2]
        return _limit_speed(self.network(torch.cat((velocities, target_vector, target), dim=-1)), self.max_speed)


def _limit_speed(commands: Tensor, max_speed: float) -> Tensor:
    """Borne la norme euclidienne de chaque commande, pas chaque composante."""
    bounded = torch.tanh(commands)
    norms = torch.linalg.vector_norm(bounded, dim=-1, keepdim=True).clamp_min(1e-12)
    return bounded * torch.clamp(max_speed / norms, max=1.0)
