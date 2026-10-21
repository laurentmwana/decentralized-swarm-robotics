"""Décodeur borné : la commande produite a toujours une norme <= v_max (conventions AGENT.md 3).

On borne dans le décodeur lui-même (et pas seulement dans le simulateur), pour que la
variante sans communication et le GNN partagent la même garantie de bornage.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, v_max: float, activation: str = "relu") -> None:
        super().__init__()
        self.v_max = float(v_max)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = torch.tanh(self.mlp(h))  # chaque composante dans (-1, 1)
        norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
        factor = torch.clamp(self.v_max / norms.clamp(min=1e-9), max=1.0)
        return raw * factor


def _activation(name: str):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[name]
