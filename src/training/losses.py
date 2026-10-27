"""Perte à quatre termes (phase 5) : forme, couverture, collisions, mouvement.

Chaque terme est une sous-classe de `LossTerm` (principe ouvert/fermé) ; la `CompositeLoss`
mélange les valeurs pondérées définies dans config/training.yaml. Toutes les distances
sont normalisées par le rayon de la cible, toutes les vitesses par v_max : les ordres de
grandeur des quatre termes restent comparables et les poids ont un sens physique.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .rollout import Rollout


class LossTerm(ABC):
    name: str = "term"

    @abstractmethod
    def forward(self, rollout: Rollout, target: torch.Tensor) -> torch.Tensor:
        """Scalaire différentiable."""


def _nearest(d: torch.Tensor, axis: int) -> torch.Tensor:
    return d.min(dim=axis).values


class ShapeLoss(LossTerm):
    """Forme : distance moyenne robot -> point de cible le plus proche.

    Fenetre de pas configurable : moyennee sur les k derniers pas (signal dense)
    plutot que le seul pas final, qui ne donne presque aucun gradient au depart.
    """

    name = "shape"

    def __init__(self, target_radius: float, window: int = 1) -> None:
        self.target_radius = float(target_radius)
        self.window = max(1, int(window))

    def forward(self, rollout: Rollout, target: torch.Tensor) -> torch.Tensor:
        step = rollout.positions[-self.window :]
        distances = [_nearest(torch.cdist(p, target), 1).mean() for p in step]
        return torch.stack(distances).mean() / self.target_radius


class CoverageLoss(LossTerm):
    """Couverture : distance moyenne point de cible -> robot le plus proche (anti-trous)."""

    name = "coverage"

    def __init__(self, target_radius: float, window: int = 1) -> None:
        self.target_radius = float(target_radius)
        self.window = max(1, int(window))

    def forward(self, rollout: Rollout, target: torch.Tensor) -> torch.Tensor:
        step = rollout.positions[-self.window :]
        distances = [_nearest(torch.cdist(target, p), 1).mean() for p in step]
        return torch.stack(distances).mean() / self.target_radius


class CollisionLoss(LossTerm):
    """Collisions : pénalité quadratique douce ((r - d)/r)², moyennée sur les paires et les pas.

    Moyenne par paire (et non somme) : la valeur reste dans [0, 1] quel que soit N, donc
    comparable entre 20, 40 et 100 robots, et ne domine plus les termes de forme (une
    somme croît en N²). `skip_steps` exclut les premiers pas : le chevauchement issu du
    tirage initial n'est pas un choix de la politique et son gradient est du bruit.
    """

    name = "collisions"

    def __init__(self, collision_radius: float, skip_steps: int = 0) -> None:
        self.collision_radius = float(collision_radius)
        self.skip_steps = max(0, int(skip_steps))

    def forward(self, rollout: Rollout, target: torch.Tensor) -> torch.Tensor:
        r = self.collision_radius
        total = torch.zeros(())
        step_count = 0
        for t in range(self.skip_steps, rollout.commands.shape[0] + 1):
            p = rollout.positions[t]
            if p.shape[0] < 2:
                continue
            d = torch.cdist(p, p)
            mask = ~torch.eye(p.shape[0], dtype=torch.bool)
            deviation = torch.relu(r - d)[mask]
            # Moyenne sur les N(N-1) paires ordonnées = moyenne sur les paires non ordonnées.
            total = total + (deviation / r).pow(2).mean()
            step_count += 1
        return total / max(1, step_count)


class MotionLoss(LossTerm):
    """Mouvement : énergie cinétique moyenne ||v||² / v_max² sur tout l'épisode (trajectoires calmes)."""

    name = "motion"

    def __init__(self, v_max: float) -> None:
        self.v_max = float(v_max)

    def forward(self, rollout: Rollout, target: torch.Tensor) -> torch.Tensor:
        return rollout.commands.pow(2).sum(dim=-1).mean() / self.v_max**2


class CompositeLoss:
    def __init__(self, terms: list[LossTerm], weights: dict[str, float]) -> None:
        missing = [t.name for t in terms if t.name not in weights]
        if missing:
            raise ValueError(f"poids absents pour les termes : {missing}")
        self.terms = terms
        self.weights = weights

    def __call__(self, rollout: Rollout, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        detail = {term.name: term.forward(rollout, target) for term in self.terms}
        total = sum(self.weights[name] * value for name, value in detail.items())
        return total, {name: float(value.detach()) for name, value in detail.items()}

    @staticmethod
    def from_config(cfg_training: dict, cfg_sim, target_radius: float, collision_radius: float) -> "CompositeLoss":
        weights = cfg_training["losses"]["weights"]
        window = int(cfg_training["losses"].get("shape_window", 1))
        collision_skip = int(cfg_training["losses"].get("collision_skip_steps", 0))
        terms: list[LossTerm] = [
            ShapeLoss(target_radius, window),
            CoverageLoss(target_radius, window),
            CollisionLoss(collision_radius, collision_skip),
            MotionLoss(cfg_sim.v_max),
        ]
        return CompositeLoss(terms, weights)
