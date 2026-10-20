"""L'API de démonstration doit reproduire exactement le protocole du mémoire.

Chaque combinaison testée est comparée à ``src.eval.run.simulate_episode``,
le code qui a produit les tableaux et les figures du chapitre 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.run import simulate_episode
from src.web import server

FLOAT_KEYS = ("shape_distance", "regularity_std", "collision_rate")
INT_KEYS = ("collisions", "convergence_step", "success")


def _reference(target_points: list, method: str, failure: float, robots: int, seed: int, target_id: int) -> dict:
    """Exécute la référence du protocole avec la même cible déjà mise à l'échelle."""
    _, evaluation, _, _, model = server._load_experiment(
        server.DEFAULT_CONFIG, server.DEFAULT_MODEL, server.DEFAULT_STAGE
    )
    target = np.asarray(target_points)
    return simulate_episode(model, method, target, robots, failure, seed, evaluation, target_id=target_id)


@pytest.mark.parametrize(
    ("target", "method", "failure", "robots"),
    [
        ("geometry_circle", "gnn", 0.0, 20),
        ("geometry_circle", "gnn", 0.2, 20),
        ("geometry_circle", "gathering", 0.1, 20),
        ("geometry_circle", "gathering", 0.0, 40),
        ("number_123", "gnn", 0.0, 100),
    ],
)
def test_mesures_egales_au_protocole_pilote(target: str, method: str, failure: float, robots: int) -> None:
    seed = 142
    web = server.simulate_target(target, seed=seed, method=method, failure=failure, robots=robots)
    _, _, paths, _, _ = server._load_experiment(server.DEFAULT_CONFIG, server.DEFAULT_MODEL, server.DEFAULT_STAGE)
    target_id = [Path(path).stem for path in paths].index(target)
    reference = _reference(web["target"]["points"], method, failure, robots, seed, target_id)
    for key in FLOAT_KEYS:
        assert web["metrics"][key] == pytest.approx(reference[key], abs=1e-12), key
    for key in INT_KEYS:
        assert web["metrics"][key] == reference[key], key


def test_frames_couvrent_l_episode_et_les_robots() -> None:
    web = server.simulate_target("geometry_circle", seed=142, robots=20)
    steps = web["simulation"]["episode_steps"]
    assert len(web["frames"]) == steps + 1
    assert all(len(frame) == 20 for frame in web["frames"])
    assert web["simulation"]["retired"] == []
    assert web["simulation"]["failure_step"] is None


def test_robot_retire_figé_apres_le_retrait() -> None:
    web = server.simulate_target("geometry_circle", seed=142, failure=0.2)
    retired = web["simulation"]["retired"]
    assert len(retired) == 4  # 20 % de 20 robots
    after_step = web["simulation"]["failure_step"] + 1
    for robot in retired:
        frozen = web["frames"][after_step][robot]
        assert web["frames"][-1][robot] == frozen


def test_echelle_duplique_la_cible_comme_le_protocole() -> None:
    base = server.simulate_target("geometry_circle", seed=7, robots=20)
    scaled = server.simulate_target("geometry_circle", seed=7, robots=100)
    assert len(scaled["target"]["points"]) == 100
    expected = np.resize(np.asarray(base["target"]["points"]), (100, 2))
    assert np.allclose(np.asarray(scaled["target"]["points"]), expected)
    assert all(len(frame) == 100 for frame in scaled["frames"])


def test_parametres_invalides_rejetes() -> None:
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        server.simulate_target("geometry_circle", method="oracle")
    assert exc.value.status_code == 400
    assert "Méthode inconnue" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        server.simulate_target("geometry_circle", failure=0.15)
    assert exc.value.status_code == 400
    assert "Fraction de panne invalide" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        server.simulate_target("geometry_circle", robots=30)
    assert exc.value.status_code == 400
    assert "Nombre de robots invalide" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        server.simulate_target("alphabet_Z")
    assert exc.value.status_code == 400
    assert "Forme inconnue" in str(exc.value.detail)
