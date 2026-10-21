"""Boucle d'optimisation (Adam) sur rollout différentiable, choix du meilleur état sur validation.

Le jeu de validation ne vient jamais du split test (règle du découpage figé) : graines
de validation dédiées sur la cible d'entraînement + une forme du split val. Le meilleur
checkpoint est sélectionné par distance moyenne de Chamfer sur ces épisodes.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from src.common.config import MetricsConfig, SimulatorConfig
from src.metrics import MetricSuite

from .checkpoint import CheckpointManager
from .logger import TrainingLogger
from .losses import CompositeLoss
from .rollout import DifferentiableSimulator
from .sampling import CyclicTargetSampler


def _version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def validation_seeds(profile: dict) -> list[int]:
    """Exige les graines explicites du profil, sans dépendre de la graine d'entraînement."""
    seeds = [int(seed) for seed in profile["validation_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("les graines de validation doivent être non vides et distinctes")
    return seeds


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        simulator: DifferentiableSimulator,
        loss: CompositeLoss,
        cfg_sim: SimulatorConfig,
        cfg_metrics: MetricsConfig,
        profile: dict,
        optimizer: dict,
        target_radius: float,
        output_dir: Path,
        seed: int,
        communication: bool = True,
    ) -> None:
        self.model = model
        self.simulator = simulator
        self.loss = loss
        self.cfg_sim = cfg_sim
        self.profile = profile
        validation_seeds(profile)
        self.seed = int(seed)
        self.communication = communication
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = TrainingLogger(self.output_dir)
        self.checkpoints = CheckpointManager(self.output_dir)
        self.suite = MetricSuite(cfg_metrics, cfg_sim, target_radius)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.batch = int(optimizer["batch_episodes"])
        self.grad_clip = float(optimizer.get("grad_clip", 0.0))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(optimizer["lr"]))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=float(optimizer.get("lr_gamma", 1.0))
        )

    def train(self, training_target: torch.Tensor, shape_id: int, validations: list[tuple[torch.Tensor, int]]) -> dict:
        return self.train_targets([(training_target, shape_id)], validations)

    def train_targets(self, targets: list[tuple[torch.Tensor, int]], validations: list[tuple[torch.Tensor, int]]) -> dict:
        sampler = CyclicTargetSampler(targets)
        if not validations:
            raise ValueError("au moins une cible de validation est requise")
        torch.manual_seed(self.seed)
        iterations = int(self.profile["iterations"])
        interval = int(self.profile["validation_interval"])
        n_robots = int(self.profile["n_robots"])
        best = (float("inf"), -1)
        start = time.time()

        for it in range(1, iterations + 1):
            totals, details = [], []
            for _ in range(self.batch):
                training_target, shape_id = sampler.next()
                positions = DifferentiableSimulator.initial_positions(
                    n_robots, self.cfg_sim.spawn_radius, self.generator
                )
                rollout = self.simulator.rollout(positions, shape_id, training_target, self.communication)
                total, detail = self.loss(rollout, training_target)
                self.logger.target_loss(it, shape_id, float(total.detach()), detail)
                totals.append(total)
                details.append(detail)
            total = torch.stack(totals).mean()
            self.optimizer.zero_grad()
            total.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.logger.loss(it, float(total.detach()), {key: float(np.mean([d[key] for d in details])) for key in details[0]})

            if it % interval == 0 or it == iterations:
                score, values = self._validate(validations, n_robots)
                self.logger.validation(it, {"selection_score": score, **values})
                if score < best[0]:
                    best = (score, it)
                    self.checkpoints.save(self.model, "best", {
                        "iteration": it, "score": score, "seed": self.seed, "communication": self.communication,
                    })
                print(f"[train] it {it}/{iterations} perte={float(total.detach()):.4f} val={score:.4f}")

        self.checkpoints.save(self.model, "last", {
            "iteration": iterations, "seed": self.seed, "communication": self.communication,
        })
        self.logger.write()
        summary = {
            "seed": self.seed,
            "iterations": iterations,
            "best_iteration": best[1],
            "best_validation_score": best[0],
            "final_loss": self.logger.last_loss,
            "duration_seconds": round(time.time() - start, 1),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _validate(self, validations: list[tuple[torch.Tensor, int]], n_robots: int) -> tuple[float, dict[str, object]]:
        """Moyenne sur toutes les graines pour chaque cible ; score partagé entre les cibles."""
        if not validations:
            raise ValueError("au moins une cible de validation est requise")
        seeds = validation_seeds(self.profile)
        was_training = self.model.training
        self.model.eval()
        targets, episodes = [], []
        try:
            with torch.no_grad():
                for i, (target, shape_id) in enumerate(validations):
                    rows = []
                    for seed in seeds:
                        # Même disque initial pour chaque cible, variante et graine d'entraînement.
                        generator = torch.Generator().manual_seed(seed)
                        positions = DifferentiableSimulator.initial_positions(n_robots, self.cfg_sim.spawn_radius, generator)
                        rollout = self.simulator.rollout(positions, shape_id, target, self.communication)
                        np_pos = rollout.positions.detach().cpu().numpy()
                        np_active = np.ones(np_pos.shape[:2], dtype=bool)
                        report = self.suite.compute(np_pos, np_active, target.detach().cpu().numpy()).values
                        row = {
                            "target_index": i,
                            "shape_id": shape_id,
                            "validation_seed": seed,
                            "final_distance": float(report["final_distance"]),
                            "success": float(bool(report["success"])),
                            "collision_rate": float(report["collision_rate"]),
                        }
                        rows.append(row)
                        episodes.append(row)
                    targets.append({
                        "target_index": i,
                        "shape_id": shape_id,
                        "seed_count": len(seeds),
                        **{key: float(np.mean([row[key] for row in rows]))
                           for key in ("final_distance", "success", "collision_rate")},
                    })
        finally:
            self.model.train(was_training)
        # Les colonnes historiques décrivent la première cible ; la seconde reste séparée.
        values = {
            "validation_distance": targets[0]["final_distance"],
            "validation_success": targets[0]["success"],
            "validation_collision_rate": targets[0]["collision_rate"],
            "secondary_validation_distance": targets[1]["final_distance"] if len(targets) > 1 else float("nan"),
            "selection_success_rate": float(np.mean([row["success"] for row in targets])),
            "selection_collision_rate": float(np.mean([row["collision_rate"] for row in targets])),
            "validation_targets": targets,
            "validation_episodes": episodes,
        }
        return float(np.mean([row["final_distance"] for row in targets])), values
