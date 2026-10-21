"""Encodeur d'observation (phase 4).

Ne voit que des grandeurs **invariantes par translation** : la vitesse du robot,
l'embedding de l'identifiant de forme (partagé), et éventuellement le vecteur vers le
point de contour le plus proche (relative, donc translation-invariante).
Aucune position absolue n'entre ici.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, n_shapes: int, use_target_geometry: bool, activation: str = "relu") -> None:
        super().__init__()
        self.use_target_geometry = use_target_geometry
        self.embedding = nn.Embedding(n_shapes, embedding_dim)
        obs_dim = 2 + (2 if use_target_geometry else 0)
        self.proj = nn.Linear(obs_dim + embedding_dim, hidden_dim)
        self.act = _activation(activation)

    def forward(self, velocity: torch.Tensor, shape_id: torch.Tensor, geometry: torch.Tensor | None) -> torch.Tensor:
        """velocity (N,2), shape_id scalaire (partagé), geometry (N,2) ou None -> h (N,hidden)."""
        n = velocity.shape[0]
        emb = self.embedding(shape_id).expand(n, -1)
        parts = [velocity]
        if self.use_target_geometry:
            if geometry is None:
                raise ValueError("use_target_geometry=true mais aucun vecteur de contour fourni")
            parts.append(geometry)
        x = torch.cat(parts + [emb], dim=1)
        return self.act(self.proj(x))


def _activation(name: str):
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU()}[name]
