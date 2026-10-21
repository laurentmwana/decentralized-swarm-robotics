"""Agrégation des messages par arête.

Contrat (conventions AGENT.md 3) : voisinage vide -> agrégat nul, la commande doit rester
valide pour un robot isolé. Les agrégations bonus (somme normalisée, attention) sont des
sous-classes de `Aggregator`, sans toucher au reste (principe ouvert/fermé, plan phase 7).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Aggregator(nn.Module):
    def forward(self, messages: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """messages (N,N,H) m_ij, mask (N,N) bool (arêtes), query (N,H) h_i -> agrégat (N,H)."""
        raise NotImplementedError


def _zero_empty_aggregate(aggregate: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
    """Force l'agrégat à zéro pour les robots de degré nul (voisinage vide -> agrégat nul)."""
    return torch.where((degrees > 0).unsqueeze(-1), aggregate, torch.zeros_like(aggregate))


class MeanAggregator(Aggregator):
    def forward(self, messages, mask, query):
        degrees = mask.sum(dim=1, keepdim=True).to(messages.dtype)
        total = (messages * mask.unsqueeze(-1)).sum(dim=1)
        return _zero_empty_aggregate(total / degrees.clamp(min=1.0), degrees.squeeze(-1))


class NormalizedSumAggregator(Aggregator):
    """Somme des messages normalisée par le degré (bonus, plan phase 7)."""

    def forward(self, messages, mask, query):
        degrees = mask.sum(dim=1, keepdim=True).to(messages.dtype)
        total = (messages * mask.unsqueeze(-1)).sum(dim=1)
        norm = torch.linalg.norm(total, dim=-1, keepdim=True)
        factor = torch.clamp(norm, max=1.0) / norm.clamp(min=1e-9)
        return _zero_empty_aggregate(total * factor * degrees.clamp(min=1.0).log1p(), degrees.squeeze(-1))


class AttentionAggregator(Aggregator):
    """Attention par arête (bonus) : scores appris à partir de (h_i, m_ij)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(2 * hidden_dim, 1)

    def forward(self, messages, mask, query):
        n, k, h = messages.shape
        q = query.unsqueeze(1).expand(n, k, h)
        scores = self.score(torch.cat([q, messages], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        degrees = mask.sum(dim=1)
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        aggregate = (weights.unsqueeze(-1) * messages).sum(dim=1)
        return _zero_empty_aggregate(aggregate, degrees)


def create_aggregator(name: str, hidden_dim: int) -> Aggregator:
    if name == "mean":
        return MeanAggregator()
    if name == "normalized_sum":
        return NormalizedSumAggregator()
    if name == "attention":
        return AttentionAggregator(hidden_dim)
    raise ValueError(f"agregateur inconnu : {name!r}")
