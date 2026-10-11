"""Commande d'entraînement réduite et reproductible sur un cercle."""

import argparse
from pathlib import Path

import numpy as np

from .training import TrainingConfig, train_circle


def main() -> None:
    """Exécute et archive le premier entraînement sur le cercle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training-circle.yaml")
    parser.add_argument("--target", default="data/shapes/geometry/geometry_circle.npy")
    parser.add_argument("--output", default="results/training-circle")
    arguments = parser.parse_args()
    model, losses = train_circle(TrainingConfig.from_yaml(arguments.config), np.load(Path(arguments.target)), arguments.output)
    print(f"iterations={len(losses)}; initial_loss={losses[0]:.6f}; final_loss={losses[-1]:.6f}; parameters={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
