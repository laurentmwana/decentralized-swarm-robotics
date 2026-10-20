"""GNN Controllers for swarm control.

Permutation-equivariant message passing controllers.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class MessagePassingLayer(nn.Module):
    """One round of message passing with mean neighbor aggregation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # Message: receiver_state + sender_state + edge_features (relative_pos + dist)
        # receiver: [hidden], sender: [hidden], edge: [3] (dx, dy, dist)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update: current_state + aggregated_messages
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, states: Tensor, positions: Tensor, adjacency: Tensor) -> Tensor:
        """
        Update states from neighbor messages.

        Args:
            states: [batch, robots, hidden]
            positions: [batch, robots, 2]
            adjacency: [batch, robots, robots] boolean

        Returns:
            Updated states [batch, robots, hidden]
        """
        batch, robot_count, _ = states.shape

        # Relative positions and distances
        relative = positions[:, None, :, :] - positions[:, :, None, :]  # [batch, robots, robots, 2]
        distances = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)  # [batch, robots, robots, 1]
        edge_features = torch.cat((relative, distances), dim=-1)  # [batch, robots, robots, 3]

        # Expand states for message computation
        # receiver: [batch, robots, 1, hidden] -> [batch, robots, robots, hidden]
        receiver = states[:, :, None, :].expand(-1, -1, robot_count, -1)
        # sender: [batch, 1, robots, hidden] -> [batch, robots, robots, hidden]
        sender = states[:, None, :, :].expand(-1, robot_count, -1, -1)

        # Message: MLP(receiver, sender, edge_features)
        messages = self.message(torch.cat((receiver, sender, edge_features), dim=-1))  # [batch, robots, robots, hidden]

        # Aggregate: mean over neighbors
        mask = adjacency.unsqueeze(-1).to(messages.dtype)  # [batch, robots, robots, 1]
        total = (messages * mask).sum(dim=2)  # [batch, robots, hidden]
        counts = mask.sum(dim=2).clamp_min(1.0)  # [batch, robots, 1]
        aggregated = total / counts  # [batch, robots, hidden]

        # Update: MLP(current_state, aggregated_messages)
        return self.update(torch.cat((states, aggregated), dim=-1))


class GNNController(nn.Module):
    """Permutation-equivariant GNN policy for variable robot counts."""

    def __init__(
        self,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
        target_count: int = 4,
        max_speed: float = 1.0,
    ) -> None:
        super().__init__()
        if message_passing_steps <= 0 or target_count <= 0 or max_speed <= 0:
            raise ValueError("Paramètres d'architecture invalides.")
        self.max_speed = max_speed

        # Target embedding: each target has a learned embedding
        self.target_embedding = nn.Embedding(target_count, hidden_dim)

        # Encoder: velocity(2) + goal_relative(2) + target_embedding(hidden) -> hidden
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim) for _ in range(message_passing_steps)
        )

        # Decoder: hidden -> velocity command (2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        positions: Tensor,
        velocities: Tensor,
        adjacency: Tensor,
        target_ids: Tensor,
        goal_relative_positions: Tensor | None = None,
        target_geometry_features: Tensor | None = None,
    ) -> Tensor:
        """
        Compute velocity commands for all robots.

        Args:
            positions: [batch, robots, 2]
            velocities: [batch, robots, 2]
            adjacency: [batch, robots, robots] boolean
            target_ids: [batch] integer indices
            goal_relative_positions: [batch, robots, 2] optional goal vectors
            target_geometry_features: [batch, robots, 3] optional geometry features

        Returns:
            Commands [batch, robots, 2] bounded by max_speed
        """
        self._validate_inputs(positions, velocities, adjacency, target_ids)

        batch, robot_count, _ = positions.shape

        # Target embedding shared across robots in batch
        target = self.target_embedding(target_ids).unsqueeze(1).expand(-1, robot_count, -1)

        # Goal relative positions (default: zero)
        relative_goal = torch.zeros_like(positions) if goal_relative_positions is None else goal_relative_positions
        if relative_goal.shape != positions.shape:
            raise ValueError("goal_relative_positions doit être [batch, robots, 2].")

        # Target geometry features (optional)
        geometry = None if target_geometry_features is None else target_geometry_features
        if geometry is not None and geometry.shape != positions.shape[:2] + (3,):
            raise ValueError("target_geometry_features doit être [batch, robots, 3].")

        # Goal vector: use geometry xy if available, else relative_goal
        target_vector = relative_goal if geometry is None else geometry[..., :2]

        # Encode initial state: velocity + goal_vector + target_embedding
        states = self.encoder(torch.cat((velocities, target_vector, target), dim=-1))

        # Message passing
        for layer in self.layers:
            states = layer(states, positions, adjacency)

        # Decode to velocity commands
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
    """Baseline: per-robot MLP without neighbor communication."""

    def __init__(
        self,
        hidden_dim: int = 64,
        target_count: int = 4,
        max_speed: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_speed = max_speed
        self.embedding = nn.Embedding(target_count, hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        velocities: Tensor,
        target_ids: Tensor,
        goal_relative_positions: Tensor | None = None,
        target_geometry_features: Tensor | None = None,
    ) -> Tensor:
        """Velocity commands without neighbor communication."""
        if velocities.ndim != 3 or velocities.shape[-1] != 2 or target_ids.shape != (velocities.shape[0],):
            raise ValueError("Entrées invalides pour la baseline sans communication.")

        batch, robot_count, _ = velocities.shape

        target = self.embedding(target_ids).unsqueeze(1).expand(-1, robot_count, -1)
        relative_goal = torch.zeros_like(velocities) if goal_relative_positions is None else goal_relative_positions
        if relative_goal.shape != velocities.shape:
            raise ValueError("goal_relative_positions doit être [batch, robots, 2].")

        geometry = None if target_geometry_features is None else target_geometry_features
        if geometry is not None and geometry.shape != velocities.shape[:2] + (3,):
            raise ValueError("target_geometry_features doit être [batch, robots, 3].")

        target_vector = relative_goal if geometry is None else geometry[..., :2]

        return _limit_speed(
            self.network(torch.cat((velocities, target_vector, target), dim=-1)),
            self.max_speed,
        )


def _limit_speed(commands: Tensor, max_speed: float) -> Tensor:
    """Bound Euclidean norm of each command, not per-component."""
    bounded = torch.tanh(commands)
    norms = torch.linalg.vector_norm(bounded, dim=-1, keepdim=True).clamp_min(1e-12)
    return bounded * torch.clamp(max_speed / norms, max=1.0)