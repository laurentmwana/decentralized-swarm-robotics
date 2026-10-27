"""Tests unitaires de la démo : petits épisodes synthétiques, jamais d'entraînement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from demo.server import ApiServer, MAX_SEED, PROJECT_ROOT, SimulationService, WebDemoSettings, finite_json
from src.common.config import ConfigLoader
from src.controllers.base import Controller
from src.shapes.index import ShapeIndex


@pytest.fixture
def loader():
    return ConfigLoader(PROJECT_ROOT / "config")


@pytest.fixture
def settings(loader, tmp_path):
    raw = loader.load_raw("web-demo")
    raw.update(horizon=4, default_target="geometries_circle", default_swarm_size=20,
               training_root=str(tmp_path / "missing_training"), shapes_root=str(tmp_path / "missing_shapes"))
    return raw


@pytest.fixture
def shape_index():
    return ShapeIndex(
        rows=[],
        point_clouds={
            "geometries_circle": np.array([[1., 0.], [-.5, .8], [-.5, -.8]]),
            "geometries_star": np.array([[0., 1.], [-.7, -.7], [.7, -.7]]),
        },
        identifiers=["geometries_circle", "geometries_star"],
    )


@pytest.fixture
def service(loader, settings, shape_index):
    return SimulationService(loader=loader, index=shape_index, settings=settings)


@pytest.fixture
def client(service):
    with TestClient(ApiServer(service).app) as connection:
        yield connection


def test_health_and_root_are_usable_without_data(loader, settings, monkeypatch):
    calls = []

    def missing_index(directory):
        calls.append(directory)
        raise FileNotFoundError("détail interne à ne pas exposer")

    monkeypatch.setattr(ShapeIndex, "load", missing_index)
    service = SimulationService(loader=loader, settings=settings)
    with TestClient(ApiServer(service).app) as client:
        assert client.get("/api/health").json() == {"status": "ok", "mode": "read_only"}
        page = client.get("/")
        assert page.status_code == 200
        assert "text/html" in page.headers["content-type"]
        assert '<html lang="fr">' in page.text
        assert "innerHTML" not in page.text
        assert "https://" not in page.text
        assert "requestAnimationFrame" in page.text
        assert calls == []
        assert service._factory is None
        for endpoint in ("/api/targets", "/api/simulate"):
            response = client.get(endpoint)
            assert response.status_code == 503
            assert "Index" in response.json()["detail"]
            assert "détail interne" not in response.text
            assert "Traceback" not in response.text
    assert len(calls) == 2


def test_missing_index_can_be_loaded_after_start(loader, settings, shape_index, monkeypatch):
    def missing_index(directory):
        raise FileNotFoundError

    monkeypatch.setattr(ShapeIndex, "load", missing_index)
    service = SimulationService(loader=loader, settings=settings)
    with TestClient(ApiServer(service).app) as client:
        assert client.get("/api/targets").status_code == 503
        monkeypatch.setattr(ShapeIndex, "load", lambda directory: shape_index)
        assert client.get("/api/simulate").status_code == 200


def test_catalog_reports_exact_targets_and_controller_availability(client):
    response = client.get("/api/targets")
    assert response.status_code == 200
    data = response.json()
    assert [(item["key"], item["shape_id"]) for item in data["targets"]] == [
        ("geometries_circle", 0), ("geometries_star", 1),
    ]
    assert data["limits"]["swarm_sizes"] == [20, 40, 100]
    assert data["defaults"]["controller"] == "manual"
    controllers = {item["name"]: item for item in data["controllers"]}
    assert controllers["manual"]["available"]
    assert controllers["flocking"]["available"]
    assert not controllers["gnn_2_rounds"]["available"]
    assert "Checkpoint" in controllers["gnn_2_rounds"]["reason"]


@pytest.mark.parametrize("controller", ["manual", "flocking"])
@pytest.mark.parametrize("size", [20, 40, 100])
def test_baselines_are_deterministic_complete_and_finite(client, controller, size):
    params = {"target": "geometries_star", "controller": controller, "n_robots": size, "seed": 0}
    first = client.get("/api/simulate", params=params)
    second = client.get("/api/simulate", params=params)
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    data = first.json()
    assert data["shape_id"] == 1
    assert data["seed"] == 0
    assert data["training_seed"] is None
    assert np.asarray(data["target"]).shape == (3, 2)
    assert np.asarray(data["positions"]).shape == (5, size, 2)
    assert np.asarray(data["active"]).shape == (5, size)
    assert all(isinstance(flag, bool) for frame in data["active"] for flag in frame)
    assert np.isfinite(data["positions"]).all()
    assert np.isfinite(data["metrics"]["distance_curve"]).all()
    assert len(data["metrics"]["distance_curve"]) == 5
    assert isinstance(data["metrics"]["success"], bool)
    assert data["metrics"]["success_threshold"] > 0
    assert "NaN" not in first.text and "Infinity" not in first.text


def test_default_manual_episode_needs_no_checkpoint(client):
    response = client.get("/api/simulate")
    assert response.status_code == 200
    assert response.json()["controller"] == "manual"


def test_seed_changes_initial_positions(client):
    first = client.get("/api/simulate", params={"seed": 0}).json()
    second = client.get("/api/simulate", params={"seed": 1}).json()
    assert first["positions"][0] != second["positions"][0]
    assert client.get("/api/simulate", params={"seed": MAX_SEED}).status_code == 200


@pytest.mark.parametrize("params", [
    {"n_robots": 4}, {"n_robots": 21}, {"n_robots": 101}, {"n_robots": "20.0"},
    {"seed": -1}, {"seed": MAX_SEED + 1}, {"seed": "abc"},
    {"target": "../../secret"}, {"target": ""},
    {"horizon": 121}, {"checkpoint": "uploaded.pt"}, {"training_root": "elsewhere"},
    {"shape_id": 65}, {"unexpected": "value"},
])
def test_schema_rejects_invalid_or_extra_fields(client, params):
    response = client.get("/api/simulate", params=params)
    assert response.status_code == 422
    assert "Paramètres" in response.json()["detail"]


@pytest.mark.parametrize("params", [
    {"controller": "unknown"}, {"controller": "gnn_1_round"},
    {"target": "geometries_unknown"}, {"target": "geometries_Circle"},
    {"target": "letters_circle"},
])
def test_unknown_controller_or_exact_target_is_not_aliased(client, params):
    assert client.get("/api/simulate", params=params).status_code == 404


def test_missing_checkpoints_return_clear_503_without_creating_run(client, settings):
    root = Path(settings["training_root"])
    assert not root.exists()
    response = client.get("/api/simulate", params={"controller": "gnn_2_rounds"})
    assert response.status_code == 503
    assert "Checkpoint" in response.json()["detail"]
    assert str(root) not in response.text and "Traceback" not in response.text
    assert not root.exists()


def test_partial_checkpoint_is_not_built(loader, settings, shape_index, monkeypatch):
    from src.evaluation.factory import ControllerFactory

    factory = ControllerFactory(loader, settings, Path(settings["training_root"]))
    monkeypatch.setattr(Path, "is_file", lambda path: path.name == "best.pt")
    monkeypatch.setattr(factory, "build", lambda *args: pytest.fail("un run partiel ne doit pas être chargé"))
    service = SimulationService(loader=loader, index=shape_index, factory=factory, settings=settings)
    with TestClient(ApiServer(service).app) as client:
        assert client.get("/api/simulate", params={"controller": "gnn_2_rounds"}).status_code == 503


class StatefulController(Controller):
    """Contrôleur partagé volontairement, pour vérifier reset et sérialisation."""

    def __init__(self):
        self.steps = 0
        self.reset_count = 0

    def reset(self):
        self.steps = 0
        self.reset_count += 1

    def command(self, state, adjacency, mission):
        assert self.steps == state.step
        self.steps += 1
        return np.full_like(state.positions, .01 * self.steps)


class FakeFactory:
    def __init__(self, loader):
        self.controller = StatefulController()
        self.build_calls = []
        shapes = replace(loader.load_shapes(), geometries=("circle", "star"), digits=(), letters=(), numbers=())
        self.saved = {"model": {"n_shapes": 2, "message_rounds": 2}, "shapes": asdict(shapes)}

    def versions(self, name):
        return [11]

    def snapshot(self, name, training_seed):
        return self.saved

    def build(self, name, training_seed):
        self.build_calls.append((name, training_seed))
        return self.controller


def test_shared_controller_is_reset_and_parallel_requests_are_isolated(loader, settings, shape_index):
    factory = FakeFactory(loader)
    service = SimulationService(loader=loader, index=shape_index, factory=factory, settings=settings)
    with TestClient(ApiServer(service).app) as client:
        def call(_):
            response = client.get("/api/simulate", params={"controller": "gnn_2_rounds", "seed": 7})
            assert response.status_code == 200
            return response.content

        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(call, range(6)))
    assert all(response == responses[0] for response in responses)
    assert factory.controller.reset_count == 6
    assert factory.build_calls == [("gnn_2_rounds", 11)] * 6


@pytest.mark.parametrize("mismatch", ["vocabulary", "mapping", "rounds"])
def test_trained_shape_identity_and_model_contract_prevent_modulo_alias(loader, settings, shape_index, mismatch):
    factory = FakeFactory(loader)
    if mismatch == "vocabulary":
        factory.saved["model"]["n_shapes"] = 1
    elif mismatch == "mapping":
        factory.saved["shapes"]["geometries"] = ["star", "circle"]
    else:
        factory.saved["model"]["message_rounds"] = 1
    service = SimulationService(loader=loader, index=shape_index, factory=factory, settings=settings)
    with TestClient(ApiServer(service).app) as client:
        response = client.get("/api/simulate", params={"controller": "gnn_2_rounds", "target": "geometries_star"})
        assert response.status_code == 503
    assert factory.build_calls == []


def test_corrupt_target_is_a_clear_503(service, shape_index):
    shape_index.point_clouds["geometries_circle"][0, 0] = np.nan
    with TestClient(ApiServer(service).app) as client:
        response = client.get("/api/simulate")
        assert response.status_code == 503
        assert "Traceback" not in response.text


def test_finite_serialization_preserves_null_booleans_and_numpy_scalars():
    result = finite_json({"curve": np.array([1., np.nan, np.inf, -np.inf]), "step": None,
                          "success": np.bool_(True), "count": np.int64(3), "value": np.float32(2.)})
    assert result == {"curve": [1., None, None, None], "step": None, "success": True, "count": 3, "value": 2.}


def test_simulation_is_read_only_and_does_not_mutate_index(client, shape_index, monkeypatch):
    original = {key: value.copy() for key, value in shape_index.point_clouds.items()}

    def reject_write(*args, **kwargs):
        pytest.fail("la démo ne doit rien écrire")

    for name in ("mkdir", "write_text", "write_bytes", "touch"):
        monkeypatch.setattr(Path, name, reject_write)
    monkeypatch.setattr(np, "save", reject_write)
    assert client.get("/api/simulate").status_code == 200
    assert client.post("/api/simulate").status_code == 405
    for key, value in original.items():
        assert np.array_equal(shape_index.point_clouds[key], value)


@pytest.mark.parametrize("overrides", [
    {"horizon": 121}, {"horizon": 0}, {"swarm_sizes": [4]},
    {"host": "0.0.0.0"}, {"seed_max": MAX_SEED + 1}, {"default_seed": -1},
    {"controllers": {"manual": {"type": "baseline"}, "arbitrary": {"type": "baseline"}}},
    {"controllers": {"manual": {"type": "trained", "variant": "gnn_2_rounds"}}},
])
def test_configuration_cannot_expand_safety_limits(settings, overrides):
    with pytest.raises(ValidationError):
        WebDemoSettings.model_validate(dict(settings, **overrides))


def test_optional_training_root_is_backward_compatible(settings):
    settings.pop("training_root")
    assert WebDemoSettings.model_validate(settings).training_root == "outputs/training"


def test_catalog_explains_missing_checkpoint_without_local_paths(client, settings):
    response = client.get("/api/targets")
    gnn = next(item for item in response.json()["controllers"] if item["name"] == "gnn_2_rounds")
    assert not gnn["available"]
    assert gnn["remedy"] and all(step["text"] for step in gnn["remedy"])
    root = Path(settings["training_root"])
    for variant in (str(root), root.as_posix(), str(PROJECT_ROOT), PROJECT_ROOT.as_posix()):
        assert variant not in response.json()["controllers"].__repr__()


class IncompatibleFactory(FakeFactory):
    """Checkpoint complet mais entraîné avec une autre config des formes."""

    def build(self, name, training_seed):
        self.build_calls.append((name, training_seed))
        raise ValueError("index des formes différent de celui du checkpoint")


def test_incompatible_checkpoint_is_diagnosed_in_catalog_and_simulation(loader, settings, shape_index):
    factory = IncompatibleFactory(loader)
    service = SimulationService(loader=loader, index=shape_index, factory=factory, settings=settings)
    with TestClient(ApiServer(service).app) as client:
        gnn = next(item for item in client.get("/api/targets").json()["controllers"] if item["name"] == "gnn_2_rounds")
        assert not gnn["available"]
        assert "catalogue de formes" in gnn["reason"]
        assert gnn["remedy"][0] == {"text": "Régénérer le catalogue de formes", "command": "make shapes"}
        response = client.get("/api/simulate", params={"controller": "gnn_2_rounds"})
        assert response.status_code == 503
        assert "catalogue de formes" in response.json()["detail"]
        assert response.json()["remedy"][0]["command"] == "make shapes"
        assert "index des formes différent" not in response.text


def test_catalog_offers_figures_not_resolutions(client):
    # Une figure = un nom ; aucun nombre de points n'est exposé ni accepté.
    targets = client.get("/api/targets").json()["targets"]
    assert targets == [
        {"key": "geometries_circle", "shape_id": 0, "type": "geometries", "figure": "circle"},
        {"key": "geometries_star", "shape_id": 1, "type": "geometries", "figure": "star"},
    ]
    assert client.get("/api/simulate", params={"target": "geometries_circle@50"}).status_code == 422
