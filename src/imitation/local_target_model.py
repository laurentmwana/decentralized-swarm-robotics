"""GAT à cible propre : un échange relatif, aucune cible privée voisine."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from .multiform_data import FEATURE_NAMES, MESSAGE_NAMES


@dataclass(frozen=True)
class LocalTargetConfig:
    hidden_dim: int = 64
    spatial_scale: float = 6.
    speed_scale: float = 1.
    count_scale: float = 12.
    init_seed: int = 42

    def __post_init__(self):
        if isinstance(self.hidden_dim, bool) or not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise ValueError('Dimension cachée entière positive requise')
        if isinstance(self.init_seed, bool) or not isinstance(self.init_seed, int) or self.init_seed < 0:
            raise ValueError('Graine entière positive ou nulle requise')
        if not all(math.isfinite(v) and v > 0 for v in (self.spatial_scale, self.speed_scale, self.count_scale)):
            raise ValueError('Échelles finies et positives requises')

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8')))


class LocalTargetGAT(nn.Module):
    def __init__(self, config, max_acceleration):
        super().__init__()
        if not math.isfinite(max_acceleration) or max_acceleration <= 0:
            raise ValueError('Limite d’accélération finie et positive requise')
        self.config, self.max_acceleration = config, float(max_acceleration)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.init_seed)
            h = config.hidden_dim
            self.encoder = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.message_encoder = nn.Sequential(nn.Linear(5, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.query = nn.Linear(h, h, bias=False)
            self.key = nn.Linear(h, h, bias=False)
            self.decoder = nn.Sequential(nn.Linear(2*h+5, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, own, neighbors, mask, *, return_attention=False):
        """own (...,8), voisins (...,K,4), masque (...,K) bool ; sortie (...,2)."""
        if own.ndim < 1 or own.shape[-1] != 8 or neighbors.ndim != own.ndim+1 or neighbors.shape[-1] != 4:
            raise ValueError('État propre (...,8) et voisins physiques (...,K,4) requis')
        if neighbors.shape[:-2] != own.shape[:-1] or mask.shape != neighbors.shape[:-1] or mask.dtype != torch.bool:
            raise ValueError('Lot de voisins et masque compatibles requis')
        if own.device != neighbors.device or own.device != mask.device or own.dtype != neighbors.dtype:
            raise ValueError('Types et appareils locaux compatibles requis')
        if not torch.is_floating_point(own) or not torch.isfinite(own).all() or not torch.isfinite(neighbors[mask]).all():
            raise ValueError('Observations locales flottantes finies requises')
        cfg = self.config
        goal = (own[..., 4:6]-own[..., :2])/cfg.spatial_scale
        velocity = own[..., 2:4]/cfg.speed_scale
        state = torch.cat((goal, velocity, own[..., 6:7]/3., own[..., 7:8]/cfg.count_scale), dim=-1)
        encoded = self.encoder(state)
        visible = torch.where(mask[..., None], neighbors, torch.zeros_like(neighbors))
        edges = torch.cat((visible[..., :2]/cfg.spatial_scale, visible[..., 2:]/cfg.speed_scale,
                           torch.linalg.vector_norm(visible[..., :2], dim=-1, keepdim=True)/cfg.spatial_scale), dim=-1)
        values = self.message_encoder(edges)
        scores = (self.query(encoded).unsqueeze(-2)*self.key(values)).sum(dim=-1)/math.sqrt(cfg.hidden_dim)
        scores = scores.masked_fill(~mask, float('-inf'))
        scores = torch.where(mask.any(dim=-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.)
        messages = (weights[..., None]*values).sum(dim=-2)
        degree = mask.sum(dim=-1, keepdim=True).to(own.dtype)/cfg.count_scale
        raw = self.decoder(torch.cat((encoded, messages, goal, velocity, degree), dim=-1))
        command = self.max_acceleration*raw/torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1.)
        return (command, weights) if return_attention else command

    def metadata(self):
        return dict(model_type='LocalTargetGAT', schema_version=1, dataset_schema_version=2,
                    config=asdict(self.config), max_acceleration=self.max_acceleration,
                    feature_names=list(FEATURE_NAMES), message_feature_names=list(MESSAGE_NAMES))


def save_model(path, model):
    with Path(path).open('xb') as stream:
        torch.save(dict(metadata=model.metadata(), state_dict=model.state_dict()), stream)


def load_model(path, *, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    meta = payload.get('metadata', {})
    if meta.get('model_type') != 'LocalTargetGAT' or meta.get('schema_version') != 1 or meta.get('dataset_schema_version') != 2:
        raise ValueError('Sauvegarde incompatible avec le GAT à cible locale')
    model = LocalTargetGAT(LocalTargetConfig(**meta['config']), meta['max_acceleration']).to(device)
    if model.metadata() != meta:
        raise ValueError('Contrat de caractéristiques du checkpoint incompatible')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model


class LocalTargetController:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, observation, target, phase, count):
        own = np.r_[observation.position, observation.velocity, target, phase, count]
        neighbors = np.concatenate((observation.relative_positions, observation.relative_velocities), axis=1)
        parameter = next(self.model.parameters())
        with torch.inference_mode():
            command = self.model(torch.as_tensor(own, dtype=parameter.dtype, device=parameter.device),
                torch.as_tensor(neighbors, dtype=parameter.dtype, device=parameter.device),
                torch.ones(len(neighbors), dtype=torch.bool, device=parameter.device))
        return command.cpu().numpy().astype(float, copy=True)
