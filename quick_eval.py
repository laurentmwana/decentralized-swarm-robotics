#!/usr/bin/env python3
"""Script d'évaluation rapide du modèle entraîné."""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Change to the project directory
os.chdir(Path(__file__).parent)

# Import the modules directly
import sys
sys.path.insert(0, 'src')

from train.training import TrainingConfig, chamfer_loss, _limit_speed, adjacency_from_positions, local_target_features
from src.models import GNNController
from sim import rules

def quick_evaluation():
    """Évalue rapidement le modèle entraîné."""
    print("=== Évaluation rapide du modèle entraîné ===")

    # Charger la configuration utilisée pour l'entraînement
    config = TrainingConfig(
        seed=41,
        batch_size=8,
        iterations=80,
        episode_steps=20,
        learning_rate=0.001,
        robot_count=20,
        arena_size=(10.0, 10.0),
        communication_radius=2.5,
        max_speed=1.0,
        time_step=0.3,
        safety_distance=0.3,
        target_radius=2.0,
        shape_loss_weight=1.0,
        collision_loss_weight=1.0,
        motion_loss_weight=0.05,
        coverage_loss_weight=0.20,
        hidden_dim=64,
        message_passing_steps=3
    )

    # Charger les points du cercle
    target_points = np.load("data/shapes/geometry/geometry_circle.npy")

    # Charger le modèle entraîné
    model_path = Path("results/debug_training/circle-final.pt")
    if not model_path.exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model = GNNController(
        config.hidden_dim,
        config.message_passing_steps,
        target_count=1,
        max_speed=config.max_speed
    )
    model.load_state_dict(state)
    model.eval()

    print(f"Modèle chargé depuis: {model_path}")
    print(f"Configuration: {config.robot_count} robots, {config.hidden_dim} dim caché, {config.message_passing_steps} étapes de message passing")

    # Faire un petit nombre d'épisodes d'évaluation
    num_eval_episodes = 5
    total_loss = 0.0
    losses = []

    with torch.no_grad():
        for i in range(num_eval_episodes):
            # Générer des positions initiales aléatoires
            starts = torch.rand((1, config.robot_count, 2))
            starts *= torch.tensor(config.arena_size)

            # Créer les cibles
            targets = torch.tensor(target_points, dtype=torch.float32) * config.target_radius
            targets += torch.tensor(config.arena_size, dtype=torch.float32) / 2
            targets = targets.unsqueeze(0)  # Ajouter la dimension batch

            target_ids = torch.zeros(1, dtype=torch.long)  # Toujours la première (et seule) cible

            # Exécuter un épisode
            final_positions, _ = rollout(model, starts, target_ids, config, targets)

            # Calculer la perte de forme
            loss_shape = chamfer_loss(final_positions, targets)
            loss_value = float(loss_shape.detach())
            losses.append(loss_value)
            total_loss += loss_value

            print(f"Épisode {i+1}: perte de forme = {loss_value:.6f}")

    avg_loss = total_loss / num_eval_episodes
    print(f"\n=== Résultats de l'évaluation ===")
    print(f"Perte moyenne sur {num_eval_episodes} épisodes: {avg_loss:.6f}")
    print(f"Perte minimale: {min(losses):.6f}")
    print(f"Perte maximale: {max(losses):.6f}")

    # Interprétation qualitative
    if avg_loss < 0.1:
        print("✅ Excellent résultat: perte très faible")
    elif avg_loss < 0.5:
        print("✅ Bon résultat: perte faible")
    elif avg_loss < 1.0:
        print("⚠️  Résultat moyen: perte modérée")
    else:
        print("❌ Résultat pauvre: perte élevée")

    return losses

def rollout(model: GNNController, positions: torch.Tensor, target_ids: torch.Tensor, config: TrainingConfig, target_points: torch.Tensor):
    """Exécute un épisode de simulation."""
    velocities = torch.zeros_like(positions)
    arena = torch.tensor(config.arena_size, dtype=positions.dtype, device=positions.device)
    center = arena / 2
    all_commands = []

    for _ in range(config.episode_steps):
        # Construire le graphe de communication
        adjacency = adjacency_from_positions(positions, config.communication_radius)

        # Caractéristiques de géométrie cible (si utilisées)
        geometry = local_target_features(positions, target_points, config.target_radius) if config.use_target_geometry and target_points is not None else None

        # Consigne (vitesse relative au centre pour simplifier)
        consigne = positions - center

        # Obtenir les commandes du modèle
        commands = model(positions, velocities, adjacency, target_ids, consigne, geometry)
        velocities = commands

        # Appliquer les contraintes de vitesse et mettre à jour les positions
        velocities = _limit_speed(velocities, config.max_speed)
        positions = torch.minimum(torch.maximum(positions + velocities * config.time_step, torch.zeros_like(arena)), arena)
        all_commands.append(commands)

    return positions, torch.stack(all_commands, dim=1)

if __name__ == "__main__":
    quick_evaluation()