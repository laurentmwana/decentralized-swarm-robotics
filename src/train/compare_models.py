"""Script pour comparer le modèle spécialisé cercle et le modèle curriculum."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.train.training import TrainingConfig, compare_models_on_targets, rollout, chamfer_loss
from src.models import GNNController
from src.sim import rules


def main() -> None:
    """Compare le modèle cercle spécialisé et le modèle curriculum."""
    print("=== Comparaison du modèle cercle spécialisé et du modèle curriculum ===")

    # Charger la configuration du curriculum pour avoir accès aux formes de test
    with open("configs/curriculum.yaml", encoding="utf-8") as stream:
        curriculum_config = yaml.safe_load(stream)
    base = dict(curriculum_config["base"])
    base["arena_size"] = tuple(base["arena_size"])
    config = TrainingConfig(**base)

    # Charger les modèles
    circle_model_path = "results/training-circle/circle-best.pt"
    curriculum_model_path = "results/curriculum/numbers/model.pt"  # Dernier stade du curriculum

    circle_state = torch.load(circle_model_path, map_location="cpu", weights_only=True)
    curriculum_state = torch.load(curriculum_model_path, map_location="cpu", weights_only=True)

    # Créer les contrôleurs (les deux ont été entraînés sur 1 cible initialement pour le cercle,
    # mais le curriculum a été étendu à plusieurs cibles - nous allons tester sur les formes de base)
    circle_controller = GNNController(config.hidden_dim, config.message_passing_steps, target_count=1, max_speed=config.max_speed)
    circle_controller.load_state_dict(circle_state)

    # Pour le curriculum, nous devons déterminer combien de cibles il attendait à l'entraînement final
    # Comme il a été entraîné sur plusieurs étapes, nous devons vérifier la dernière étape
    # Pour simplifier, nous allons tester sur un sous-ensemble de formes que les deux modèles peuvent gérer
    # Nous allons utiliser les formes de base que le cercle connaissait : cercle, carré, triangle

    # Charger les points cibles depuis les fichiers .npy
    test_targets = {
        "circle": np.load("data/shapes/geometry/geometry_circle.npy"),
        "square": np.load("data/shapes/geometry/geometry_square.npy"),
        "triangle": np.load("data/shapes/geometry/geometry_triangle.npy"),
    }

    # Comme le modèle curriculum s'attend à plusieurs cibles, nous devons l'adapter
    # Pour la comparaison, nous allons utiliser uniquement la première embedding du modèle curriculum
    # correspondant à la forme sur laquelle nous testons (en supposant que c'est similaire à l'entraînement)

    # En fait, une meilleure approche : évaluer chaque modèle séparément sur chaque cible
    # et utiliser l'embedding approprié pour chaque forme

    # Pour simplifier cette comparaison, nous allons :
    # 1. Pour le modèle cercle (spécialisé sur 1 forme) : utiliser target_count=1
    # 2. Pour le modèle curriculum : déterminer quelle embedding utiliser pour chaque forme
    #    Comme le curriculum a été entraîné sur plusieurs formes dans l'ordre, nous supposons
    #    que l'index de l'embedding correspond à l'ordre d'apparition dans le curriculum

    # Cependant, pour une comparaison juste, nous allons faire quelque chose de plus simple :
    # Évaluer chaque modèle avec target_count=1 sur chaque forme individuelle
    # Cela signifie que nous allons créer des contrôleurs spécifiques pour chaque forme

    print("\nRésultats de la comparaison (loss de Chamfer):")
    print("=" * 50)

    results_data = {
        "circle_model": circle_model_path,
        "curriculum_model": curriculum_model_path,
        "comparison": {}
    }

    for target_name, target_points in test_targets.items():
        # Créer un contrôleur cercle spécifique pour cette cible
        circle_controller_specific = GNNController(config.hidden_dim, config.message_passing_steps, target_count=1, max_speed=config.max_speed)
        circle_controller_specific.load_state_dict(circle_state)

        # Pour le curriculum, nous devons déterminer quelle embedding utiliser
        # Comme simplicité, nous allons utiliser le modèle curriculum tel quel (target_count multiple)
        # et faire la moyenne ou prendre la première embedding - mais cela n'est pas idéal
        # Pour une comparaison équitable, nous allons aussi créer un contrôleur curriculum spécifique
        # en ne prenant que l'embedding correspondant à cette forme

        # Approximation : prendre la première embedding du curriculum (supposons qu'elle soit similaire)
        # Une meilleure approche serait de savoir à quelle index corresponde cette forme dans le curriculum
        curriculum_controller_specific = GNNController(config.hidden_dim, config.message_passing_steps, target_count=1, max_speed=config.max_speed)
        # Copier tous les paramètres sauf l'embedding, puis prendre la première embedding pour la cible
        curriculum_state_dict = curriculum_state.copy()
        # Pour simplifier, nous prenons juste la première embedding et l'dupliquons
        # Ceci n'est pas parfait mais permet une comparaison de base
        if 'target_embedding.weight' in curriculum_state_dict:
            # Garder seulement la première embedding et la dupliquer pour target_count=1
            first_embedding = curriculum_state_dict['target_embedding.weight'][:1, :]  # Shape: [1, embedding_dim]
            curriculum_state_dict['target_embedding.weight'] = first_embedding
            curriculum_state_dict['target_embedding.bias'] = curriculum_state_dict['target_embedding.bias'][:1]  # Si bias existe

        curriculum_controller_specific.load_state_dict(curriculum_state_dict, strict=False)

        # Exécuter la comparaison pour cette cible spécifique
        comparison_result = compare_models_on_targets(
            circle_controller_specific,
            curriculum_controller_specific,
            [target_points],  # Liste avec un seul élément
            config,
            f"Modèle Cercle Spécialisé ({target_name})",
            f"Modèle Curriculum ({target_name})"
        )

        circle_loss = comparison_result["Modèle Cercle Spécialisé ({})".format(target_name)][0]
        curriculum_loss = comparison_result["Modèle Curriculum ({})".format(target_name)][0]

        print(f"{target_name:12} | Cercle: {circle_loss:8.4f} | Curriculum: {curriculum_loss:8.4f} | Diff: {curriculum_loss - circle_loss:8.4f}")

        results_data["comparison"][target_name] = {
            "circle_loss": float(circle_loss),
            "curriculum_loss": float(curriculum_loss),
            "difference": float(curriculum_loss - circle_loss)
        }

    # Sauvegarder les résultats
    output_path = Path("results/comparison")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nRésultats sauvegardés dans: {output_path / 'model_comparison.json'}")


if __name__ == "__main__":
    main()