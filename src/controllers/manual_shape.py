"""Règle manuelle avec consigne de forme (baseline réglée, phase 3).

Mécanisme du contrôleur trivial de la phase 1 — attraction vers le point de contour
le plus proche (calcul local, pas d'affectation globale) + répulsion entre voisins —
mais avec les gains **réglés sous budget** (config/baselines.yaml) pour servir de
plancher crédible face au GNN.
"""

from __future__ import annotations

from dataclasses import dataclass

from .trivial import TrivialController


@dataclass(frozen=True)
class ManualShapeController(TrivialController):
    pass
