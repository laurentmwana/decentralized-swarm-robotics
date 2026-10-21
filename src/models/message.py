"""Couche de message passing, écrite à la main (pas de bibliothèque de GNN).

Un tour = messages par arête à partir de (h_i, h_j, position relative), agrégation sur les
voisins, puis mise à jour du noeud avec connexion résiduelle. La position relative rend le
tour invariant par translation ; l'absence d'identifiant de robot le rend équivariant par
permutation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .aggregator import Aggregator


class MessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, aggregator: Aggregator, activation: str = "relu") -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.aggregator = aggregator

    def forward(self, h: torch.Tensor, positions: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """h (N,H), positions (N,2), adjacence (N,N) bool -> h' (N,H)."""
        n = h.shape[0]
        hi = h.unsqueeze(1).expand(n, n, h.shape[-1])
        hj = h.unsqueeze(0).expand(n, n, h.shape[-1])
        rel = (positions.unsqueeze(0) - positions.unsqueeze(1))  # rel[i,j] = pos_j - pos_i
        messages = self.edge_mlp(torch.cat([hi, hj, rel], dim=-1))
        aggregate = self.aggregator(messages, adjacency, h)
        return h + self.update_mlp(torch.cat([h, aggregate], dim=-1))


def _activation(name: str):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[name]
