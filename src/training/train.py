"""Point d'entrée d'entraînement : python -m src.training.train --variant gnn_2_rounds --seed 11 [--profile standard|smoke]

Tous les paramètres viennent de config/ (model.yaml + training.yaml + simulator.yaml +
metrics.yaml + shapes.yaml) ; les variantes diffèrent uniquement par les surcharges
`variants:` de training.yaml, avec un budget de réglage identique par construction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.config import ConfigLoader, ModelConfig
from src.models.gnn import SwarmGNN
from src.shapes.index import ShapeSplit, generate_index
from src.training.losses import CompositeLoss
from src.training.rollout import DifferentiableSimulator
from src.training.trainer import Trainer, _version, validation_seeds

OUTPUT = Path("outputs/training")


def build_model(cfg_model: ModelConfig, override: dict, v_max: float) -> SwarmGNN:
    fields = {**cfg_model.__dict__, **override}
    fields.pop("communication", None)  # surcharge de variante, pas du modèle
    return SwarmGNN(ModelConfig(**fields), v_max=v_max)


def validate_splits(split: ShapeSplit, training_key: str, validation_key: str) -> None:
    """Contrôle les identifiants sans résolution dans le découpage figé."""
    for key, expected in ((training_key, "train"), (validation_key, "val")):
        actual = split.mapping.get(key.split("@")[0])
        if actual != expected:
            raise ValueError(f"cible {key!r} du split {actual!r}, attendu {expected!r} : accès refusé")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(
    variant: str, seed: int, profile: str, output: Path = OUTPUT,
    loader: ConfigLoader | None = None, training_config: dict | None = None,
    training_keys: list[str] | None = None, validation_keys: list[str] | None = None,
    initial_checkpoint: Path | None = None, curriculum_context: dict | None = None,
) -> dict:
    output_dir = Path(output) / variant / f"seed_{seed}"
    if output_dir.exists():
        raise FileExistsError(f"dossier d'entraînement déjà occupé : {output_dir}")
    loader = loader or ConfigLoader()
    cfg_training = training_config if training_config is not None else loader.load_raw("training")
    cfg_sim = loader.load_simulator()
    cfg_metrics = loader.load_metrics()
    cfg_shapes = loader.load_shapes()
    cfg_model = loader.load_model()

    if variant not in cfg_training["variants"]:
        raise ValueError(f"variante inconnue : {variant} (connues : {sorted(cfg_training['variants'])})")
    override = cfg_training["variants"][variant]
    profile_config = cfg_training["profiles"][profile]
    seeds = validation_seeds(profile_config)

    cfg_rollout = replace(cfg_sim, horizon=int(profile_config["horizon"]))
    index, split = generate_index(cfg_shapes, Path("outputs/shapes"))
    training_keys = [cfg_training["training_target"]] if training_keys is None else training_keys
    validation_keys = [cfg_training["validation_target"]] if validation_keys is None else validation_keys
    if not training_keys or not validation_keys:
        raise ValueError("les cibles d'entraînement et de validation doivent être explicites et non vides")
    if len(set(training_keys)) != len(training_keys) or len(set(validation_keys)) != len(validation_keys):
        raise ValueError("cibles d'entraînement ou de validation dupliquées")
    for training_key in training_keys:
        for validation_key in validation_keys:
            validate_splits(split, training_key, validation_key)
    all_keys = training_keys + validation_keys
    for key in all_keys:
        validate_target(index.point_cloud_by_key(key))
        if not 0 <= index.shape_id(key) < cfg_model.n_shapes:
            raise ValueError(f"identifiant de forme hors vocabulaire : {key}")
    targets = [(torch.tensor(index.point_cloud_by_key(key), dtype=torch.float32), index.shape_id(key))
               for key in training_keys]
    validations = [(torch.tensor(index.point_cloud_by_key(key), dtype=torch.float32), index.shape_id(key))
                   for key in all_keys]

    torch.manual_seed(seed)
    model = build_model(replace(cfg_model, init_seed=int(seed)), override, cfg_sim.v_max)
    if initial_checkpoint is not None:
        payload = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        if payload["metadata"]["model_config"] != asdict(model.cfg):
            raise ValueError("architecture ou graine incompatibles avec le transfert de poids")
        if payload["metadata"]["seed"] != seed or payload["metadata"]["communication"] != bool(override.get("communication", True)):
            raise ValueError("identité du checkpoint incompatible avec le transfert de poids")
        model.load_state_dict(payload["state_dict"])
    simulator = DifferentiableSimulator(model, cfg_rollout)
    loss = CompositeLoss.from_config(
        cfg_training, cfg_rollout, cfg_shapes.target_radius, cfg_metrics.collision_radius
    )
    communication = bool(override.get("communication", True))
    effective_config = {
        "variant": variant,
        "override": override,
        "profile": profile,
        "communication": communication,
        "model": asdict(model.cfg),
        "training": cfg_training,
        "simulator": asdict(cfg_rollout),
        "metrics": asdict(cfg_metrics),
        "shapes": asdict(cfg_shapes),
    }
    if curriculum_context is not None:
        effective_config["curriculum"] = curriculum_context
    if initial_checkpoint is not None:
        import hashlib

        with initial_checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        effective_config["initial_checkpoint"] = {"path": str(initial_checkpoint), "sha256": digest}

    def target_record(key: str) -> dict:
        return {
            "key": key,
            "shape_id": index.shape_id(key),
            "split": split.mapping[key.split("@")[0]],
            "points": index.point_cloud_by_key(key).tolist(),
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "prepared", "variant": variant, "seed": int(seed), "profile": profile}
    _write_json(output_dir / "status.json", status)
    (output_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(effective_config, allow_unicode=True, sort_keys=False), encoding="utf-8",
    )
    _write_json(output_dir / "seeds.json", {
        "seed": int(seed),
        "training_seeds": sorted(set(int(value) for value in profile_config["seeds"]) | {int(seed)}),
        "validation_seeds": seeds,
        "model_init_seed": model.cfg.init_seed,
        "sampler_seed": cfg_shapes.sampler_seed,
    })
    (output_dir / "version.txt").write_text(_version() + "\n", encoding="utf-8")
    _write_json(output_dir / "targets.json", {
        "training_target": target_record(training_keys[0]),
        "training_targets": [target_record(key) for key in training_keys],
        "validation_targets": [target_record(key) for key in all_keys],
    })
    try:
        trainer = Trainer(
            model=model,
            simulator=simulator,
            loss=loss,
            cfg_sim=cfg_rollout,
            cfg_metrics=cfg_metrics,
            profile=profile_config,
            optimizer=cfg_training["optimizer"],
            target_radius=cfg_shapes.target_radius,
            output_dir=output_dir,
            seed=seed,
            communication=communication,
        )
        _write_json(output_dir / "status.json", {**status, "status": "running"})
        if len(targets) == 1:
            summary = trainer.train(targets[0][0], targets[0][1], validations)
        else:
            summary = trainer.train_targets(targets, validations)
        summary["parameter_count"] = model.parameter_count()
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "status.json", {**status, "status": "completed"})
    except BaseException as exc:
        # L'échec conserve la provenance et les éventuels checkpoints pour diagnostic.
        _write_json(output_dir / "status.json", {**status, "status": "failed", "error": str(exc)})
        raise
    return summary


def validate_target(target: np.ndarray) -> None:
    """Cohérence de la cible d'entraînement : points distincts, dimension 2 (leçon §0.6)."""
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError(f"cible invalide : shape {target.shape}")
    if len(np.unique(np.round(target, 9), axis=0)) != len(target):
        raise ValueError("la cible contient des doublons : échantillonnage à corriger")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    variants = parser.add_mutually_exclusive_group(required=True)
    variants.add_argument("--variant")
    variants.add_argument("--all-variants", action="store_true")
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = ConfigLoader().load_raw("training")
    selected_variants = list(cfg["variants"]) if args.all_variants else [args.variant]
    selected_seeds = cfg["profiles"][args.profile]["seeds"] if args.all_seeds else [args.seed]
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        parser.error("les graines doivent être non vides et distinctes")
    if set(selected_variants) - cfg["variants"].keys():
        parser.error("variante inconnue")
    jobs = [{"variant": variant, "seed": seed, "profile": args.profile,
             "output": str(args.output / variant / f"seed_{seed}")}
            for variant in selected_variants for seed in selected_seeds]
    if args.dry_run:
        print(json.dumps({"jobs": jobs, "profile": cfg["profiles"][args.profile]}, indent=2))
        return
    for job in jobs:
        if Path(job["output"]).exists():
            raise FileExistsError(f"dossier d'entraînement déjà occupé : {job['output']}")
    for job in jobs:
        run(job["variant"], job["seed"], args.profile, args.output)


if __name__ == "__main__":
    main()
