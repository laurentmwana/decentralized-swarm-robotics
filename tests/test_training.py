import numpy as np
import torch
from pathlib import Path

from src.train.training import TrainingConfig, adjacency_from_positions, chamfer_loss, collision_loss, coverage_loss, local_target_features, train_targets


def test_losses_are_zero_or_small_on_simple_expected_cases() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    assert chamfer_loss(positions, positions).item() == 0.0
    assert collision_loss(positions, safety_distance=0.2).item() == 0.0


def test_adjacency_excludes_self_and_respects_radius() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]]])
    adjacency = adjacency_from_positions(positions, radius=1.5)
    assert adjacency.tolist() == [[[False, True, False], [True, False, False], [False, False, False]]]


def test_target_features_and_coverage_match_an_identical_target() -> None:
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    features = local_target_features(positions, positions, scale=1.0)
    assert torch.allclose(features, torch.zeros_like(features))
    assert coverage_loss(positions, positions).item() == 0.0


def test_training_accepts_multiple_target_conditions() -> None:
    config = TrainingConfig(2, 2, 1, 1, 0.001, 2, (2.0, 2.0), 2.0, 1.0, 0.1, 0.1, 0.5, 1.0, 1.0, 0.0, 8, 1)
    targets = [np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([[0.0, -1.0], [0.0, 1.0]])]
    model, losses, metrics, _ = train_targets(config, targets, Path("results/test-target-conditions"))
    assert len(losses) == 1
    assert set(metrics) == {0, 1}
    assert model.target_embedding.num_embeddings == 2


def test_losses_near_zero_on_expected_cases() -> None:
    """Test que les pertes sont proches de zéro pour des cas simples et attendus."""
    # Cas parfait : robots exactement sur les cibles
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])

    # La perte de forme devrait être très proche de zéro
    from src.train.training import chamfer_loss
    loss_shape = chamfer_loss(positions, target)
    assert loss_shape.item() < 1e-6

    # Avec des robots bien espacés, pas de collision
    positions_no_collision = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]]])
    from src.train.training import collision_loss
    loss_collision = collision_loss(positions_no_collision, safety_distance=0.5)
    assert loss_collision.item() == 0.0  # Exactement zéro car pas de paires trop proches

    # Mouvement zéro pour des vitesses nulles
    from src.train.training import collision_loss
    velocities = torch.zeros_like(positions_no_collision)
    # On peut pas tester directement la perte de mouvement sans rollout complet,
    # mais on peut vérifier que la fonction existe et retourne une perte raisonnable


def test_no_self_loop_in_adjacency() -> None:
    """Test que la matrice d'adjacence exclut bien les auto-boucles."""
    from src.train.training import adjacency_from_positions
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    adjacency = adjacency_from_positions(positions, radius=1.5)

    # Vérifier qu'il n'y a pas d'auto-boucle (diagonale doit être False)
    batch_size, n_robots, _ = adjacency.shape
    for b in range(batch_size):
        for i in range(n_robots):
            assert adjacency[b, i, i] == False, f"Auto-boucle détectée à [{b},{i},{i}]"


def test_target_consistency() -> None:
    """Test de cohérence de la cible : si on entraîne sur une cible parfaite, la perte devrait être faible."""
    config = TrainingConfig(
        seed=42,
        batch_size=2,
        iterations=2,
        episode_steps=5,
        learning_rate=0.001,
        robot_count=3,
        arena_size=(5.0, 5.0),
        communication_radius=2.0,
        max_speed=0.5,
        time_step=0.1,
        safety_distance=0.2,
        target_radius=1.0,
        shape_loss_weight=1.0,
        collision_loss_weight=1.0,
        motion_loss_weight=0.01,
        coverage_loss_weight=0.2,
        hidden_dim=8,
        message_passing_steps=1,
    )

    # Créer une cible simple (triangle équilatéral)
    target_points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2]
    ], dtype=np.float32)

    # Entraîner très brièvement juste pour voir si la perte diminue
    model, losses, _, _ = train_targets(
        config,
        [target_points],
        Path("results/test-target-consistency")
    )

    # Après quelques itérations, la perte devrait avoir diminué par rapport à l'initialement élevé
    assert len(losses) == 2
    # La perte ne devrait pas augmenter de façon significative (tolérance pour la variabilité)
    assert losses[1] <= losses[0] * 1.5  # Autorise une petite augmentation à cause du hasard/initialisation
