import numpy as np

from src.eval.metrics import collision_count, nearest_neighbor_std, symmetric_chamfer_distance


def test_metrics_match_perfect_and_simple_cases() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert symmetric_chamfer_distance(positions, positions) == 0.0
    assert collision_count(positions, 0.5) == 0
    assert nearest_neighbor_std(positions) == 0.0


def test_collision_count_detects_one_pair() -> None:
    positions = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0]])
    assert collision_count(positions, 0.3) == 1
