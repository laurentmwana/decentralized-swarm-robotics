"""Échantillonnage équilibré des cibles conservées par le curriculum."""

from __future__ import annotations

import torch


class CyclicTargetSampler:
    def __init__(self, targets: list[tuple[torch.Tensor, int]]) -> None:
        if not targets or len({shape_id for _, shape_id in targets}) != len(targets):
            raise ValueError("les cibles doivent être non vides et avoir des identifiants distincts")
        self.targets = tuple(targets)
        self.offset = 0

    def next(self) -> tuple[torch.Tensor, int]:
        target = self.targets[self.offset]
        self.offset = (self.offset + 1) % len(self.targets)
        return target
