"""Décodeur borné : la commande produite a toujours une norme <= v_max (conventions AGENT.md 3).

On borne dans le décodeur lui-même (et pas seulement dans le simulateur), pour que la
variante sans communication et le GNN partagent la même garantie de bornage.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, v_max: float, activation: str = "relu", geometry_prior: float = 1.0) -> None:
        super().__init__()
        self.v_max = float(v_max)
        self.geometry_prior = float(geometry_prior)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h: torch.Tensor, geometry: torch.Tensor | None = None) -> torch.Tensor:
        """h (N,hidden), geometry (N,2) optionnel : residu qui oriente la commande vers le contour des l'initialisation."""
        raw = self.mlp(h)
        if geometry is not None and self.geometry_prior > 0:
            raw = raw + self.geometry_prior * geometry
        raw = torch.tanh(raw)  # chaque composante dans (-1, 1)
        norms = torch.linalg.norm(raw, dim=-1, keepdim=True)
        factor = torch.clamp(self.v_max / norms.clamp(min=1e-9), max=1.0)
        return raw * factor


def _activation(name: str):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[name]
