#!/usr/bin/env python3
"""Script de débogage pour l'entraînement du modèle GNN."""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Change to the project directory and add src to path
os.chdir(Path(__file__).parent)
sys.path.insert(0, 'src')

from train.training import TrainingConfig, train_targets
from sim import rules

def debug_training():
    """Exécute un petit entraînement pour déboguer les performances."""
    print("=== Démarrage du débogage d'entraînement ===")

    # Configuration simple pour le débogage
    config = TrainingConfig(
        seed=42,
        batch_size=4,
        iterations=20,  # Réduit pour le débogage rapide
        episode_steps=10,
        learning_rate=0.001,
        robot_count=8,  # Réduit pour accélérer
        arena_size=(5.0, 5.0),
        communication_radius=1.5,
        max_speed=0.5,
        time_step=0.2,
        safety_distance=0.2,
        target_radius=1.0,
        shape_loss_weight=1.0,
        collision_loss_weight=1.0,
        motion_loss_weight=0.01,
        coverage_loss_weight=0.20,
        hidden_dim=16,  # Réduit pour le débogage
        message_passing_steps=2
    )

    # Créer une cible simple (cercle)
    target_points = rules.circle_points(config.robot_count)
    target_points = target_points * config.target_radius
    target_points += np.array(config.arena_size) / 2

    print(f"Configuration:")
    print(f"  - Robots: {config.robot_count}")
    print(f"  - Itérations: {config.iterations}")
    print(f"  - Taille de lot: {config.batch_size}")
    print(f"  - Dimension cachée: {config.hidden_dim}")
    print(f"  - Étapes de message passing: {config.message_passing_steps}")
    print(f"  - Taux d'apprentissage: {config.learning_rate}")

    # Entraîner sur une seule cible pour simplifier
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    try:
        model, losses, metrics, per_target_losses = train_targets(
            config,
            [target_points],  # Liste avec une seule cible
            output_dir
        )

        print(f"\n=== Résultats de l'entraînement ===")
        print(f"Loss initiale: {losses[0]:.6f}")
        print(f"Loss finale: {losses[-1]:.6f}")
        print(f"Amélioration totale: {losses[0] - losses[-1]:.6f}")
        print(f"Métriques: {metrics}")

        # Vérifier si l'entraînement a fonctionné
        if losses[-1] < losses[0] * 0.9:  # Amélioration d'au moins 10%
            print("✅ L'entraînement semble fonctionner (perte en diminution)")
        else:
            print("⚠️  L'entraînement ne semble pas améliorer suffisamment la perte")

        # Sauvegarder le modèle pour inspection
        torch.save(model.state_dict(), output_dir / "debug_model.pt")
        print(f"Modèle sauvegardé dans: {output_dir / 'debug_model.pt'}")

        return losses

    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_training()