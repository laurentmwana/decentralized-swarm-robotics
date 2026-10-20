"""Programme d'apprentissage progressif par catégories de formes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml

from .training import TrainingConfig, train_targets


def main() -> None:
    """Enchaîne les niveaux et sauvegarde un modèle à chaque étape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/curriculum.yaml")
    parser.add_argument("--initial-model", default="results/training-circle/circle-best.pt")
    parser.add_argument("--output", default="results/curriculum")
    parser.add_argument("--stage", choices=("geometry", "digits", "symbols", "numbers"))
    arguments = parser.parse_args()
    with Path(arguments.config).open(encoding="utf-8") as stream:
        curriculum = yaml.safe_load(stream)
    base = dict(curriculum["base"])
    base["arena_size"] = tuple(base["arena_size"])
    config = TrainingConfig(iterations=1, **base)
    state = None if arguments.initial_model.lower() == "none" else torch.load(arguments.initial_model, map_location="cpu", weights_only=True)
    output = Path(arguments.output)
    for stage_index, stage in enumerate(curriculum["stages"]):
        if arguments.stage is not None and stage["name"] != arguments.stage:
            continue
        targets = [np.load(path) for path in stage["targets"]]
        stage_config = replace(config, iterations=stage["iterations"], seed=config.seed + stage_index)
        stage_output = output / stage["name"]
        model, losses, metrics, per_target_losses = train_targets(stage_config, targets, stage_output, state)
        (stage_output / "targets.yaml").write_text(yaml.safe_dump({"targets": stage["targets"]}, sort_keys=False), encoding="utf-8")

        # Save per-target loss history
        for target_id, loss_history in per_target_losses.items():
            if loss_history:  # Only save if we have data
                np.savetxt(stage_output / f"loss_target_{target_id}.csv", np.asarray(loss_history), delimiter=",")

        state = model.state_dict()
        print(f"stage={stage['name']}; targets={len(targets)}; initial_loss={losses[0]:.4f}; final_loss={losses[-1]:.4f}; metrics={metrics}")


if __name__ == "__main__":
    main()
