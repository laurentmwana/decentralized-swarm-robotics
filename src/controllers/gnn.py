"""Contrôleurs GNN pour le simulateur (plan phase 4).

`GNNController` : décentralisé, partage un seul jeu de poids pour tous les robots,
n'utilise aucun identifiant de robot, que des positions relatives et la consigne partagée
(identifiant de forme, éventuellement + vecteur vers le contour). Sortie bornée à v_max.

`NoCommGNNController` : même modèle, graphe de communication vidé à chaque pas -> agrégat
nul -> aucun message ne circule. Taille de modèle comparable (équité de comparaison).
"""

from __future__ import annotations

import numpy as np
import torch

from src.common.config import ModelConfig
from src.common.mission import Mission
from src.controllers.base import Controller
from src.models.gnn import SwarmGNN


class GNNController(Controller):
    def __init__(self, cfg: ModelConfig, v_max: float, communication: bool = True, seed: int | None = None) -> None:
        self.cfg = cfg
        self.v_max = float(v_max)
        self.communication = communication
        torch.manual_seed(cfg.init_seed if seed is None else seed)
        self.model = SwarmGNN(cfg, self.v_max)
        self.model.eval()
        self._velocity = np.zeros((0, 2), dtype=np.float64)

    def reset(self) -> None:
        self._velocity = np.zeros((0, 2), dtype=np.float64)

    def target_vector(self, positions: np.ndarray, mission: Mission) -> np.ndarray:
        """Vecteur unitaire du robot vers son point de cible le plus proche (local, translation-invariant)."""
        delta = mission.target[None, :, :] - positions[:, None, :]
        distances = np.linalg.norm(delta, axis=2)
        nearest = np.argmin(distances, axis=1)
        toward = delta[np.arange(len(positions)), nearest]
        norms = np.linalg.norm(toward, axis=1, keepdims=True)
        return toward / np.maximum(norms, 1e-9)

    def command(self, state, adjacency: np.ndarray, mission: Mission) -> np.ndarray:
        positions = np.asarray(state.positions, dtype=np.float64)
        n = len(positions)
        if self._velocity.shape[0] != n:
            self._velocity = np.zeros((n, 2), dtype=np.float64)

        geometry = self.target_vector(positions, mission) if self.cfg.use_target_geometry else None
        graph = adjacency if self.communication else np.zeros_like(adjacency)

        with torch.no_grad():
            t_pos = torch.tensor(positions, dtype=torch.float32)
            t_vel = torch.tensor(self._velocity, dtype=torch.float32)
            t_adj = torch.tensor(np.asarray(graph, dtype=bool))
            t_shape = torch.tensor(int(mission.shape_id) % self.cfg.n_shapes, dtype=torch.long)
            t_geo = None if geometry is None else torch.tensor(geometry, dtype=torch.float32)
            output = self.model(t_vel, t_pos, t_adj, t_shape, t_geo).numpy().astype(np.float64)

        self._velocity = output
        return output

    def parameter_count(self) -> int:
        return self.model.parameter_count()


class NoCommGNNController(GNNController):
    def __init__(self, cfg: ModelConfig, v_max: float, seed: int | None = None) -> None:
        super().__init__(cfg, v_max, communication=False, seed=seed)
