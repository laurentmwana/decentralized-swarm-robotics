"""Démonstration reproductible du simulateur avec règle locale."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .core import SimulationConfig, SwarmSimulator
from .rules import local_gathering_and_repulsion
from .visualization import animate_history


def run_gathering_demo(config: SimulationConfig) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """Exécute un épisode de regroupement et retourne les états successifs."""
    generator = np.random.default_rng(config.seed)
    center = np.asarray(config.arena_size) / 2.0
    positions = generator.uniform(center - 2.0, center + 2.0, size=(config.robot_count, 2))
    simulator = SwarmSimulator(config, positions=positions)
    position_history, active_history = [simulator.positions.copy()], [simulator.active.copy()]
    collision_count = 0
    for _ in range(config.episode_steps):
        simulator.step(local_gathering_and_repulsion(simulator, repulsion_gain=10.0))
        collision_count += len(simulator.collision_pairs())
        position_history.append(simulator.positions.copy())
        active_history.append(simulator.active.copy())
    return position_history, active_history, collision_count


def main() -> None:
    """Exporte l'animation GIF de la règle de regroupement configurée."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulator.yaml")
    parser.add_argument("--output", default="results/simulator-gathering.gif")
    arguments = parser.parse_args()
    config = SimulationConfig.from_yaml(arguments.config)
    positions, active, collision_count = run_gathering_demo(config)
    animation = animate_history(positions, active, config.arena_size)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer="pillow", fps=12)
    print(f"animation={output}; frames={len(positions)}; collisions={collision_count}")


if __name__ == "__main__":
    main()
