"""Assemblage Encoder -> N tours de message -> Decoder borné (plan phase 4).

`SwarmGNN` est indépendant du nombre de robots : mêmes poids quel que soit N (20, 40, 100).
La variante sans communication (`NoCommGNNController`) réutilise ce modèle avec un graphe
vide : l'agrégat est nul à chaque tour, donc aucun message ne circule, mais la taille du
modèle reste comparable (exigence d'équité, plan phase 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.common.config import ModelConfig

from .aggregator import create_aggregator
from .decoder import Decoder
from .encoder import Encoder
from .message import MessageLayer


class SwarmGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, v_max: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.hidden_dim, cfg.embedding_dim, cfg.n_shapes, cfg.use_target_geometry, cfg.activation)
        self.layers = nn.ModuleList(
            MessageLayer(cfg.hidden_dim, create_aggregator(cfg.aggregator, cfg.hidden_dim), cfg.activation)
            for _ in range(cfg.message_rounds)
        )
        self.decoder = Decoder(cfg.hidden_dim, v_max, cfg.activation)

    def forward(
        self,
        velocity: torch.Tensor,
        positions: torch.Tensor,
        adjacency: torch.Tensor,
        shape_id: torch.Tensor,
        geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """velocity (N,2), positions (N,2), adjacence (N,N) bool, shape_id scalaire -> commande (N,2)."""
        h = self.encoder(velocity, shape_id, geometry)
        for layer in self.layers:
            h = layer(h, positions, adjacency)
        return self.decoder(h)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
