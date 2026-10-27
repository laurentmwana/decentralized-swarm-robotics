"""Chargement de contrôleurs gelés depuis la configuration sauvegardée de chaque run."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from src.common.config import ConfigLoader, ModelConfig, ShapesConfig, SimulatorConfig
from src.controllers.base import Controller
from src.controllers.flocking import FlockingController
from src.controllers.gnn import GNNController
from src.controllers.manual_shape import ManualShapeController
from src.training.checkpoint import CheckpointManager

TRAINING_DIR = Path("outputs/training")
REQUIRED_ARTIFACTS = ("best.pt", "effective_config.yaml", "summary.json", "status.json", "targets.json")

RUN_INCOMPLETE = "run incomplet"
TRAINING_UNFINISHED = "entraînement non terminé"
_BLOCKER_ERRORS = {RUN_INCOMPLETE: FileNotFoundError, TRAINING_UNFINISHED: ValueError}


class ControllerFactory:
    def __init__(self, loader: ConfigLoader | None = None, evaluation_config: dict | None = None, training_dir: Path = TRAINING_DIR) -> None:
        self.loader = loader or ConfigLoader()
        self.config = evaluation_config if evaluation_config is not None else self.loader.load_raw("evaluation")
        self.training_dir = Path(training_dir)

    def versions(self, name: str) -> list[int | None]:
        usable, blocked = self.inspect(name)
        if blocked:
            kind, message = blocked[0]
            raise _BLOCKER_ERRORS[kind](message)
        return usable

    def inspect(self, name: str) -> tuple[list[int | None], list[tuple[str, str]]]:
        """Graines réellement exploitables, puis les motifs de rejet des autres."""
        settings = self.config["controllers"][name]
        if settings["type"] == "baseline":
            return [None], []
        if settings["type"] != "trained":
            raise ValueError(f"type de contrôleur inconnu : {settings['type']}")
        seeds = self.config["training_seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("graines d'entraînement vides ou dupliquées")
        usable, blocked = [], []
        for seed in seeds:
            blocker = self._blocker(self.run_directory(name, seed))
            if blocker is None:
                usable.append(seed)
            else:
                blocked.append((blocker[0], f"variante {name} graine {seed}: {blocker[1]}"))
        return usable, blocked

    @staticmethod
    def _blocker(directory: Path) -> tuple[str, str] | None:
        missing = next((artifact for artifact in REQUIRED_ARTIFACTS if not (directory / artifact).is_file()), None)
        if missing is not None:
            return RUN_INCOMPLETE, f"run incomplet : {directory / missing}"
        status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if status["status"] != "completed":
            return TRAINING_UNFINISHED, f"entraînement non terminé : {directory}"
        return None

    def run_directory(self, name: str, training_seed: int) -> Path:
        settings = self.config["controllers"][name]
        root = Path(settings.get("training_root", self.config.get("training_root", self.training_dir)))
        return root / settings["variant"] / f"seed_{training_seed}"

    def snapshot(self, name: str, training_seed: int) -> dict:
        with (self.run_directory(name, training_seed) / "effective_config.yaml").open(encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def build(self, name: str, training_seed: int | None) -> Controller:
        settings = self.config["controllers"][name]
        if settings["type"] == "baseline":
            rules = self.loader.load_raw("baselines")["rules"]
            if name == "flocking":
                return FlockingController(**rules[name])
            if name == "manual":
                return ManualShapeController(**rules[name])
            raise ValueError(f"baseline inconnue : {name}")
        if training_seed is None:
            raise ValueError(f"{name} exige une graine d'entraînement")
        directory = self.run_directory(name, training_seed)
        snapshot = self.snapshot(name, training_seed)
        cfg_model = ModelConfig(**snapshot["model"])
        cfg_sim = SimulatorConfig(**snapshot["simulator"])
        current_sim = self.loader.load_simulator()
        for field in ("dt", "v_max", "r_c", "arena_size", "spawn_radius"):
            if getattr(cfg_sim, field) != getattr(current_sim, field):
                raise ValueError(f"configuration simulateur différente du checkpoint : {field}")
        if ShapesConfig(**snapshot["shapes"]) != self.loader.load_shapes():
            raise ValueError("index des formes différent de celui du checkpoint")
        communication = bool(snapshot["communication"])
        controller = GNNController(cfg_model, cfg_sim.v_max, communication=communication, seed=training_seed)
        metadata = CheckpointManager(directory).load(controller.model, "best")
        if metadata["model_config"] != asdict(cfg_model):
            raise ValueError("configuration du checkpoint différente de sa provenance")
        if metadata["communication"] != communication or metadata["seed"] != training_seed:
            raise ValueError("identité du checkpoint différente de sa provenance")
        controller.model.eval()
        controller.model.requires_grad_(False)
        return controller
