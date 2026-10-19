"""Script principal d'entraînement pour la Phase 9 : Curriculum de formes."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from .training import (
    TrainingConfig,
    train_circle,
    train_targets,
    compare_models_on_targets,
)
from .curriculum import main as curriculum_main


def train_curriculum_model(config_path: str, initial_model: str, output_dir: str, stage: str | None) -> str:
    """Entraîne le modèle curriculum et retourne le chemin du modèle final."""
    print("=== Entraînement du modèle curriculum ===")
    curriculum_main_inner(config_path, initial_model, output_dir, stage)
    # Retourner le chemin du dernier modèle entraîné
    curriculum_output = Path(output_dir)
    if stage:
        final_model_path = curriculum_output / stage / "model.pt"
    else:
        # Trouver la dernière étape
        stages = ["geometry", "digits", "symbols", "numbers"]
        for stage_name in reversed(stages):
            model_path = curriculum_output / stage_name / "model.pt"
            if model_path.exists():
                final_model_path = model_path
                break
        else:
            final_model_path = curriculum_output / "geometry" / "model.pt"

    return str(final_model_path)


def curriculum_main_inner(config_path: str, initial_model: str, output_dir: str, stage: str | None) -> None:
    """Version interne de la fonction curriculum main."""
    with Path(config_path).open(encoding="utf-8") as stream:
        curriculum = yaml.safe_load(stream)
    base = dict(curriculum["base"])
    base["arena_size"] = tuple(base["arena_size"])
    # Créer une configuration de base avec toutes les valeurs requises
    base_config = TrainingConfig(
        seed=base["seed"],
        batch_size=base["batch_size"],
        iterations=base.get("iterations", 1),  # Valeur par défaut si pas spécifiée
        episode_steps=base["episode_steps"],
        learning_rate=base["learning_rate"],
        robot_count=base["robot_count"],
        arena_size=base["arena_size"],
        communication_radius=base["communication_radius"],
        max_speed=base["max_speed"],
        time_step=base["time_step"],
        safety_distance=base["safety_distance"],
        target_radius=base["target_radius"],
        shape_loss_weight=base["shape_loss_weight"],
        collision_loss_weight=base["collision_loss_weight"],
        motion_loss_weight=base["motion_loss_weight"],
        coverage_loss_weight=base.get("coverage_loss_weight", 0.0),
        hidden_dim=base.get("hidden_dim", 32),
        message_passing_steps=base.get("message_passing_steps", 2),
        use_target_geometry=base.get("use_target_geometry", False)
    )
    state = None if initial_model.lower() == "none" else torch.load(initial_model, map_location="cpu", weights_only=True)
    output = Path(output_dir)
    for stage_index, stage_config in enumerate(curriculum["stages"]):
        if stage is not None and stage_config["name"] != stage:
            continue
        targets = [np.load(path) for path in stage_config["targets"]]
        # Créer une nouvelle configuration au lieu de modifier celle gelée
        stage_config_obj = TrainingConfig(
            seed=base_config.seed + stage_index,
            batch_size=base_config.batch_size,
            iterations=stage_config["iterations"],
            episode_steps=base_config.episode_steps,
            learning_rate=base_config.learning_rate,
            robot_count=base_config.robot_count,
            arena_size=base_config.arena_size,
            communication_radius=base_config.communication_radius,
            max_speed=base_config.max_speed,
            time_step=base_config.time_step,
            safety_distance=base_config.safety_distance,
            target_radius=base_config.target_radius,
            shape_loss_weight=base_config.shape_loss_weight,
            collision_loss_weight=base_config.collision_loss_weight,
            motion_loss_weight=base_config.motion_loss_weight,
            coverage_loss_weight=base_config.coverage_loss_weight,
            hidden_dim=base_config.hidden_dim,
            message_passing_steps=base_config.message_passing_steps,
            use_target_geometry=base_config.use_target_geometry
        )
        stage_output = output / stage_config["name"]
        model, losses, metrics, per_target_losses = train_targets(stage_config_obj, targets, stage_output, state)
        (stage_output / "targets.yaml").write_text(yaml.safe_dump({"targets": stage_config["targets"]}, sort_keys=False), encoding="utf-8")

        # Sauvegarder l'historique des pertes par cible
        for target_id, loss_history in per_target_losses.items():
            if loss_history:
                np.savetxt(stage_output / f"loss_target_{target_id}.csv", np.asarray(loss_history), delimiter=",")

        state = model.state_dict()
        print(f"stage={stage_config['name']}; targets={len(targets)}; initial_loss={losses[0]:.4f}; final_loss={losses[-1]:.4f}; metrics={metrics}")


def evaluate_models_comparison(circle_model_path: str, curriculum_model_path: str, config_path: str, output_dir: str) -> None:
    """Compare le modèle spécialisé cercle et le modèle curriculum sur les mêmes épisodes."""
    print("=== Comparaison du modèle cercle spécialisé et du modèle curriculum ===")

    # Charger la configuration du curriculum pour avoir accès aux formes de test
    with open(config_path, encoding="utf-8") as stream:
        curriculum_config = yaml.safe_load(stream)
    base = dict(curriculum_config["base"])
    base["arena_size"] = tuple(base["arena_size"])
    config = TrainingConfig(**base)

    # Charger les modèles
    circle_model = torch.load(circle_model_path, map_location="cpu", weights_only=True)
    curriculum_model = torch.load(curriculum_model_path, map_location="cpu", weights_only=True)

    # Créer les contrôleurs (les deux ont été entraînés sur 1 cible initialement pour le cercle,
    # mais le curriculum a été étendu à plusieurs cibles - nous allons tester sur les formes de base)
    circle_controller = GNNController(config.hidden_dim, config.message_passing_steps, target_count=1, max_speed=config.max_speed)
    circle_controller.load_state_dict(circle_model)

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
        circle_controller_specific.load_state_dict(circle_model)

        # Pour le curriculum, créer un contrôleur spécifique avec une seule embedding
        curriculum_controller_specific = GNNController(config.hidden_dim, config.message_passing_steps, target_count=1, max_speed=config.max_speed)
        curriculum_state_dict = curriculum_model.copy()
        # Pour simplifier, nous prenons juste la première embedding et l'dupliquons pour target_count=1
        # Ceci n'est pas parfait mais permet une comparaison de base
        if 'target_embedding.weight' in curriculum_state_dict:
            # Garder seulement la première embedding et la dupliquer pour target_count=1
            first_embedding = curriculum_state_dict['target_embedding.weight'][:1, :]  # Shape: [1, embedding_dim]
            curriculum_state_dict['target_embedding.weight'] = first_embedding
            if 'target_embedding.bias' in curriculum_state_dict:
                curriculum_state_dict['target_embedding.bias'] = curriculum_state_dict['target_embedding.bias'][:1]

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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nRésultats sauvegardés dans: {output_path / 'model_comparison.json'}")


def replace(obj, **changes):
    """Helper function to replace dataclass fields."""
    for key, value in changes.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def main() -> None:
    """Point d'entrée principal pour l'entraînement Phase 9."""
    parser = argparse.ArgumentParser(description="Entraînement du swarm GNN avec curriculum de formes (Phase 9)")
    parser.add_argument("--circle-config", default="configs/training.yaml", help="Configuration pour l'entraînement cercle")
    parser.add_argument("--curriculum-config", default="configs/curriculum.yaml", help="Configuration pour le curriculum")
    parser.add_argument("--initial-model", default="results/training-circle/circle-best.pt", help="Modèle initial pour le curriculum")
    parser.add_argument("--circle-output", default="results/training-circle", help="Répertoire de sortie pour l'entraînement cercle")
    parser.add_argument("--curriculum-output", default="results/curriculum", help="Répertoire de sortie pour le curriculum")
    parser.add_argument("--comparison-output", default="results/comparison", help="Répertoire de sortie pour la comparaison")
    parser.add_argument("--stage", choices=("geometry", "digits", "symbols", "numbers"), help="Étapespecificque du curriculum à exécuter")
    parser.add_argument("--skip-circle", action="store_true", help="Ignorer l'entraînement du cercle spécialisé")
    parser.add_argument("--skip-curriculum", action="store_true", help="Ignorer l'entraînement du curriculum")
    parser.add_argument("--skip-comparison", action="store_true", help="Ignorer la comparaison des modèles")

    args = parser.parse_args()

    circle_model_path = None
    curriculum_model_path = None

    # Entraîner le modèle spécialisé cercle
    if not args.skip_circle:
        print("Phase 9: Entraînement du modèle spécialisé cercle")
        with Path(args.circle_config).open(encoding="utf-8") as stream:
            circle_config_dict = yaml.safe_load(stream)
        # Handle both config formats: with "base" key or direct parameters
        if "base" in circle_config_dict:
            base = dict(circle_config_dict["base"])
            base["arena_size"] = tuple(base["arena_size"])
            circle_config = TrainingConfig(**base)
        else:
            # Direct format (like training-circle.yaml)
            circle_config_dict["arena_size"] = tuple(circle_config_dict["arena_size"])
            circle_config = TrainingConfig(**circle_config_dict)

        # Charger les points du cercle depuis le fichier de forme
        circle_target_path = Path("data/shapes/geometry/geometry_circle.npy")
        if not circle_target_path.exists():
            raise FileNotFoundError(f"Le fichier de cible cercle n'existe pas: {circle_target_path}")
        circle_target = np.load(circle_target_path)

        circle_model, circle_losses = train_circle(circle_config, circle_target, args.circle_output)
        torch.save(circle_model.state_dict(), Path(args.circle_output) / "circle-final.pt")
        circle_model_path = str(Path(args.circle_output) / "circle-final.pt")
        print(f"Modèle cercle sauvegardé: {circle_model_path}")

    # Entraîner le modèle curriculum
    if not args.skip_curriculum:
        print("\nPhase 9: Entraînement du modèle curriculum")
        if args.skip_circle and not Path(args.initial_model).exists():
            print(f"ATTENTION: Le modèle initial {args.initial_model} n'existe pas. Utilisation d'un modèle aléatoire.")
        curriculum_model_path = train_curriculum_model(
            args.curriculum_config,
            args.initial_model if not args.skip_circle else "none",
            args.curriculum_output,
            args.stage
        )
        print(f"Modèle curriculum terminé. Dernier modèle: {curriculum_model_path}")

    # Comparer les modèles
    if not args.skip_comparison and circle_model_path and curriculum_model_path:
        print("\nPhase 9: Comparaison du modèle cercle spécialisé et du modèle curriculum")
        evaluate_models_comparison(
            circle_model_path,
            curriculum_model_path,
            args.curriculum_config,  # Utiliser la config du curriculum pour les formes de test
            args.comparison_output
        )
    elif args.skip_comparison:
        print("\nPhase 9: Comparaison ignorée selon les arguments")
    else:
        print("\nPhase 9: Comparaison ignorée - modèles manquants")


if __name__ == "__main__":
    main()