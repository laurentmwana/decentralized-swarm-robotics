"""GAT partagé à un échange local, sortie d'accélération bornée."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
import yaml


@dataclass(frozen=True)
class GATConfig:
    hidden_dim: int
    spatial_scale: float
    speed_scale: float
    count_scale: float
    negative_slope: float
    init_seed: int

    def __post_init__(self):
        if isinstance(self.hidden_dim, bool) or not isinstance(self.hidden_dim, int) or self.hidden_dim < 1:
            raise ValueError('Dimension cachée entière positive requise')
        if not all(math.isfinite(v) and v > 0 for v in (self.spatial_scale, self.speed_scale, self.count_scale)):
            raise ValueError('Échelles finies et positives requises')
        if not math.isfinite(self.negative_slope) or not 0 < self.negative_slope < 1:
            raise ValueError('Pente LeakyReLU attendue dans ]0,1[')
        if isinstance(self.init_seed, bool) or not isinstance(self.init_seed, int) or self.init_seed < 0:
            raise ValueError('Graine entière positive ou nulle requise')

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            return cls(**yaml.safe_load(stream))


class OneHopAttention(nn.Module):
    """Un message de chaque voisin immédiat ; pas de propagation itérative."""
    def __init__(self, hidden_dim, negative_slope):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, 1, bias=False)
        self.key = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, encoded, adjacency):
        values = self.projection(encoded)
        scores = self.activation(self.query(values) + self.key(values).transpose(-1, -2))
        scores = scores.masked_fill(~adjacency, float('-inf'))
        has_neighbors = adjacency.any(dim=-1, keepdim=True)
        # Une ligne sans voisin ne passe jamais un softmax de -inf uniquement.
        safe_scores = torch.where(has_neighbors, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=-1).masked_fill(~adjacency, 0.)
        return weights @ values, weights


class AccelerationGAT(nn.Module):
    feature_count = 8

    def __init__(self, config: GATConfig, max_acceleration: float):
        super().__init__()
        if not math.isfinite(max_acceleration) or max_acceleration <= 0:
            raise ValueError('Limite d’accélération finie et positive requise')
        self.config = config
        self.max_acceleration = float(max_acceleration)
        # Reproductibilité sans modifier le générateur global du programme appelant.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.init_seed)
            h = config.hidden_dim
            self.encoder = nn.Sequential(nn.Linear(6, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
            self.attention = OneHopAttention(h, config.negative_slope)
            self.decoder = nn.Sequential(nn.Linear(2*h + 1, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, features, adjacency, *, return_attention=False):
        """features (...,N,8), adj (...,N,N) bool ; sortie (...,N,2), en m/s²."""
        if features.ndim < 2 or features.shape[-1] != self.feature_count or features.shape[-2] < 1:
            raise ValueError('Caractéristiques attendues (...,N,8), N>0')
        expected = features.shape[:-2] + (features.shape[-2], features.shape[-2])
        if adjacency.shape != expected or adjacency.dtype != torch.bool:
            raise ValueError('Graphe booléen compatible requis')
        if features.device != adjacency.device:
            raise ValueError('Graphe et caractéristiques doivent partager le même appareil')
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError('Caractéristiques flottantes finies requises')
        if adjacency.diagonal(dim1=-2, dim2=-1).any():
            raise ValueError('Les auto-boucles ne font pas partie du contrat')
        cfg = self.config
        own = torch.cat(((features[..., :2] - features[..., 4:6]) / cfg.spatial_scale,
                          features[..., 2:4] / cfg.speed_scale,
                          features[..., 6:7] / cfg.spatial_scale,
                          features[..., 7:8] / cfg.count_scale), dim=-1)
        encoded = self.encoder(own)
        messages, weights = self.attention(encoded, adjacency)
        degree = adjacency.sum(dim=-1, keepdim=True).to(features.dtype) / cfg.count_scale
        raw = self.decoder(torch.cat((encoded, messages, degree), dim=-1))
        # Norme euclidienne <= a_max, sans division par zéro ni écrêtage par composante.
        norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1.)
        output = self.max_acceleration * raw / norm
        return (output, weights) if return_attention else output

    def metadata(self):
        return dict(model_type='AccelerationGAT', schema_version=1, config=asdict(self.config),
                    max_acceleration=self.max_acceleration, feature_count=self.feature_count)
