"""Objectif supervisé de commandes ; aucune optimisation n'est exécutée ici."""
import torch


def acceleration_mse(predictions, targets):
    """MSE moyenne sur les exemples et les deux axes, en m²/s⁴."""
    if predictions.shape != targets.shape or predictions.ndim < 2 or predictions.shape[-1] != 2 or predictions.numel() == 0:
        raise ValueError('Commandes compatibles non vides (...,2) requises')
    if not torch.is_floating_point(predictions) or not torch.is_floating_point(targets):
        raise ValueError('Commandes flottantes requises')
    if predictions.device != targets.device or predictions.dtype != targets.dtype:
        raise ValueError('Commandes de même type et sur le même appareil requises')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError('Commandes finies requises')
    return (predictions - targets).square().mean()
