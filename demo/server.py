"""API locale illustrative : un épisode à la demande, sans entraînement ni sauvegarde."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.common.config import ConfigLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_SEED = 2**31 - 1
ControllerName = Literal["manual", "flocking", "gnn_2_rounds"]
SwarmSize = Literal[20, 40, 100]
TargetKey = Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9_-]+@[0-9]+$")]


class ControllerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    type: Literal["baseline", "trained"]
    variant: Literal["gnn_2_rounds"] | None = None


class WebDemoSettings(BaseModel):
    """Les limites de sécurité ne peuvent pas être élargies par le YAML."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    host: Literal["127.0.0.1"]
    port: int = Field(ge=1, le=65535)
    shapes_root: str
    training_root: str = "outputs/training"
    horizon: int = Field(ge=1, le=120)
    swarm_sizes: list[SwarmSize] = Field(min_length=1)
    seed_max: int = Field(ge=0, le=MAX_SEED)
    default_seed: int = Field(ge=0, le=MAX_SEED)
    default_target: TargetKey
    default_swarm_size: SwarmSize
    default_controller: Literal["manual", "flocking"]
    training_seeds: list[Annotated[int, Field(ge=0, le=MAX_SEED)]] = Field(min_length=1)
    controllers: dict[ControllerName, ControllerSettings]

    @model_validator(mode="after")
    def validate_choices(self) -> "WebDemoSettings":
        if len(set(self.swarm_sizes)) != len(self.swarm_sizes):
            raise ValueError("tailles d'essaim dupliquées")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("graines d'entraînement dupliquées")
        if self.default_seed > self.seed_max or self.default_swarm_size not in self.swarm_sizes:
            raise ValueError("valeurs par défaut hors des limites")
        if "manual" not in self.controllers or self.default_controller not in self.controllers:
            raise ValueError("la baseline manuelle et le contrôleur par défaut sont requis")
        for name, settings in self.controllers.items():
            expected = "trained" if name == "gnn_2_rounds" else "baseline"
            variant = "gnn_2_rounds" if expected == "trained" else None
            if settings.type != expected or settings.variant != variant:
                raise ValueError("type ou variante de contrôleur non autorisé")
        return self


class SimulationRequest(BaseModel):
    """Seuls ces paramètres sont publics ; aucun chemin, poids ou horizon libre."""

    model_config = ConfigDict(extra="forbid")
    target: TargetKey | None = None
    controller: Annotated[str, Field(min_length=1, max_length=32, pattern=r"^[a-z][a-z0-9_]*$")] | None = None
    n_robots: SwarmSize | None = None
    seed: Annotated[int, Field(ge=0, le=MAX_SEED)] | None = None

    @field_validator("n_robots", mode="before")
    @classmethod
    def parse_swarm_size(cls, value):
        # Une chaîne de requête doit devenir un entier avant la vérification Literal.
        return int(value) if isinstance(value, str) and value.isascii() and value.isdecimal() else value


class DemoUnavailable(Exception):
    """Erreur publique sans chemin local ni détail technique sensible."""


def finite_json(value: Any) -> Any:
    """Convertit récursivement NumPy et remplace les nombres non finis par null."""
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, dict):
        return {str(key): finite_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class SimulationService:
    """Dépendances injectables ; le verrou protège aussi une fabrique partageant un contrôleur."""

    def __init__(self, loader=None, index=None, factory=None, settings=None) -> None:
        self.loader = loader if loader is not None else ConfigLoader(PROJECT_ROOT / "config")
        raw = settings if settings is not None else self.loader.load_raw("web-demo")
        self.settings = raw if isinstance(raw, WebDemoSettings) else WebDemoSettings.model_validate(raw)
        self._index = index
        self._factory = factory
        self._lock = RLock()

    @staticmethod
    def _path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _get_index(self):
        if self._index is None:
            try:
                from src.shapes.index import ShapeIndex

                self._index = ShapeIndex.load(self._path(self.settings.shapes_root))
            except Exception as exc:
                raise DemoUnavailable("Index des formes indisponible ou incomplet. Générez les formes avant de simuler.") from exc
        if not self._index.point_clouds or not self._index.identifiers:
            raise DemoUnavailable("Index des formes vide ou incomplet.")
        return self._index

    def _get_factory(self):
        if self._factory is None:
            # L'import et les éventuels poids restent hors du démarrage du serveur.
            from src.evaluation.factory import ControllerFactory

            evaluation_config = {
                "controllers": {name: item.model_dump(exclude_none=True) for name, item in self.settings.controllers.items()},
                "training_seeds": list(self.settings.training_seeds),
            }
            self._factory = ControllerFactory(
                self.loader, evaluation_config, self._path(self.settings.training_root)
            )
        return self._factory

    def _trained_seed(self, name: str) -> int:
        try:
            versions = self._get_factory().versions(name)
            if versions != self.settings.training_seeds:
                raise ValueError("aucune graine d'entraînement commune avec le catalogue")
            return versions[0]
        except Exception as exc:
            # La cause réelle vient du factory : artefact manquant ou entraînement non terminé.
            raise DemoUnavailable(f"Checkpoint {name} indisponible : {str(exc).strip() or 'entraînement non terminé'}") from exc

    def targets(self) -> dict:
        with self._lock:
            index = self._get_index()
            entries = []
            for key in sorted(index.point_clouds):
                identifier, resolution = key.rsplit("@", 1)
                shape_id = index.shape_id(key)
                entries.append({
                    "key": key, "shape_id": shape_id, "name": identifier,
                    "resolution": int(resolution), "n_points": len(index.point_cloud_by_key(key)),
                })
            controllers = []
            for name, settings in self.settings.controllers.items():
                available, reason = True, None
                if settings.type == "trained":
                    try:
                        self._trained_seed(name)
                    except DemoUnavailable as exc:
                        available, reason = False, str(exc)
                controllers.append({"name": name, "type": settings.type, "available": available, "reason": reason})
            return {
                "targets": entries, "controllers": controllers,
                "defaults": {
                    "target": self.settings.default_target, "controller": self.settings.default_controller,
                    "n_robots": self.settings.default_swarm_size, "seed": self.settings.default_seed,
                },
                "limits": {"swarm_sizes": self.settings.swarm_sizes, "horizon": self.settings.horizon, "seed_max": self.settings.seed_max},
            }

    def simulate(self, request: SimulationRequest) -> dict:
        # Les routes synchrones tournent dans le pool FastAPI ; tout état reste isolé.
        with self._lock:
            name = request.controller if request.controller is not None else self.settings.default_controller
            key = request.target if request.target is not None else self.settings.default_target
            size = request.n_robots if request.n_robots is not None else self.settings.default_swarm_size
            seed = request.seed if request.seed is not None else self.settings.default_seed
            if name not in self.settings.controllers:
                raise HTTPException(404, "Contrôleur inconnu ou non autorisé.")
            if size not in self.settings.swarm_sizes or seed > self.settings.seed_max:
                raise HTTPException(422, "Taille d'essaim ou graine hors des limites configurées.")
            index = self._get_index()
            if key not in index.point_clouds:
                raise HTTPException(404, "Cible inconnue : sélectionnez une clé exacte de l'index.")
            training_seed = None
            try:
                from src.common.mission import Mission
                from src.metrics.suite import MetricSuite
                from src.simulator.episode import Episode

                shape_id = index.shape_id(key)
                if not 0 <= shape_id < len(index.identifiers) or index.identifiers[shape_id] != key.rsplit("@", 1)[0]:
                    raise DemoUnavailable("Identifiant de forme incohérent dans l'index.")
                target = np.array(index.point_cloud_by_key(key), dtype=np.float64, copy=True)
                if not np.isfinite(target).all():
                    raise DemoUnavailable("Le nuage cible contient des coordonnées non finies.")
                if self.settings.controllers[name].type == "trained":
                    training_seed = self._trained_seed(name)
                    snapshot = self._get_factory().snapshot(name, training_seed)
                    from src.common.config import ShapesConfig
                    from src.shapes.index import build_sources

                    identifiers = [source.identifier for source in build_sources(ShapesConfig(**snapshot["shapes"]))]
                    if identifiers != index.identifiers or not 0 <= shape_id < snapshot["model"]["n_shapes"]:
                        raise DemoUnavailable("Cible incompatible avec le vocabulaire du checkpoint ; aucun alias modulo n'est permis.")
                    if snapshot["model"]["message_rounds"] != 2:
                        raise DemoUnavailable("Le checkpoint ne correspond pas au GNN à deux tours.")
                controller = self._get_factory().build(name, training_seed)
                controller.reset()
                config = replace(self.loader.load_simulator(), horizon=self.settings.horizon, record_positions=True)
                mission = Mission(shape_id=shape_id, target=target)
                result = Episode(config).execute(mission, controller, size, seed)
                suite = MetricSuite(self.loader.load_metrics(), config, self.loader.load_shapes().target_radius)
                metrics = suite.compute(result.positions, result.active, target).values
                return finite_json({
                    "target_key": key, "shape_id": shape_id, "controller": name,
                    "training_seed": training_seed, "seed": seed, "n_robots": size,
                    "horizon": result.step, "dt": config.dt, "arena_size": config.arena_size,
                    "target": target, "positions": result.positions, "active": result.active,
                    "metrics": dict(metrics, success_threshold=suite.success_threshold),
                })
            except (HTTPException, DemoUnavailable):
                raise
            except Exception as exc:
                message = "Checkpoint GNN incompatible ou illisible." if training_seed is not None else "Simulation indisponible : vérifiez les données et la configuration locale."
                raise DemoUnavailable(message) from exc


class ApiServer:
    """Façade FastAPI sans opérations d'écriture ni exposition de dossiers locaux."""

    def __init__(self, service: SimulationService | None = None) -> None:
        self.service = service if service is not None else SimulationService()
        self.app = FastAPI(title="Démonstration locale d'essaim", docs_url=None, redoc_url=None)

        @self.app.exception_handler(DemoUnavailable)
        async def unavailable_handler(request, exc):
            return JSONResponse(status_code=503, content={"detail": str(exc)})

        @self.app.exception_handler(RequestValidationError)
        async def validation_handler(request, exc):
            return JSONResponse(status_code=422, content={
                "detail": "Paramètres invalides ou non autorisés.",
                "errors": [{"field": ".".join(str(part) for part in error["loc"]), "type": error["type"]} for error in exc.errors()],
            })

        @self.app.get("/", include_in_schema=False)
        def index_page():
            return FileResponse(PROJECT_ROOT / "demo" / "index.html", media_type="text/html; charset=utf-8")

        @self.app.get("/api/health")
        def health():
            return {"status": "ok", "mode": "read_only"}

        @self.app.get("/api/targets")
        def targets():
            return self.service.targets()

        @self.app.get("/api/simulate")
        def simulate(params: Annotated[SimulationRequest, Query()]):
            return self.service.simulate(params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Démonstration locale en lecture seule (aucun entraînement).")
    parser.add_argument("--config", default="web-demo", help="Nom de la configuration dans config/, sans extension.")
    args = parser.parse_args()
    if not args.config.replace("-", "").replace("_", "").isalnum():
        parser.error("le nom de configuration ne doit pas contenir de chemin")
    loader = ConfigLoader(PROJECT_ROOT / "config")
    service = SimulationService(loader=loader, settings=loader.load_raw(args.config))
    server = ApiServer(service)
    import uvicorn

    uvicorn.run(server.app, host=service.settings.host, port=service.settings.port)


if __name__ == "__main__":
    main()
