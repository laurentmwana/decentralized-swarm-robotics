"""Sauvegardes GAT versionnées, distinctes des modèles historiques en vitesse."""
from pathlib import Path

import torch

from .model import AccelerationGAT, GATConfig


def save_model(path, model):
    """Sauvegarde les poids actuels sans affirmer qu'ils ont été entraînés."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    with path.open('xb') as stream:
        torch.save(dict(metadata=model.metadata(), state_dict=model.state_dict()), stream)


def load_model(path, *, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=True)
    metadata = payload.get('metadata', {})
    if metadata.get('model_type') != 'AccelerationGAT' or metadata.get('schema_version') != 1 or metadata.get('feature_count') != 8:
        raise ValueError('Sauvegarde incompatible avec le GAT à accélérations')
    model = AccelerationGAT(GATConfig(**metadata['config']), metadata['max_acceleration']).to(device)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model
