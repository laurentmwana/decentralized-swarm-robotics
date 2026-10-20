import numpy as np

from src.eval.metrics import collision_count, nearest_neighbor_std, symmetric_chamfer_distance, graph_connectivity


def test_metrics_match_perfect_and_simple_cases() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert symmetric_chamfer_distance(positions, positions) == 0.0
    assert collision_count(positions, 0.5) == 0
    assert nearest_neighbor_std(positions) == 0.0


def test_collision_count_detects_one_pair() -> None:
    positions = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0]])
    assert collision_count(positions, 0.3) == 1


def test_graph_connectivity() -> None:
    """Test du calcul de la connectivité du graphe."""
    # Test avec des robots bien connectés (tous dans la portée)
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # Ligne de 3 robots
    assert graph_connectivity(positions, communication_radius=1.5) == 1  # Tous connectés

    # Test avec des robots séparés (pas de connexions)
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])  # Bien séparés
    assert graph_connectivity(positions, communication_radius=1.5) == 3  # Trois composantes

    # Test avec une connexion partielle
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])  # Premiers deux connectés, troisième isolé
    assert graph_connectivity(positions, communication_radius=1.5) == 2  # Deux composantes

    # Test avec zéro robot
    assert graph_connectivity(np.empty((0, 2)), communication_radius=1.5) == 0
