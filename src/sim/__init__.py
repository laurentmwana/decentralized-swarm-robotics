"""Simulateur 2D d'essaim."""

from .core import SimulationConfig, SwarmSimulator
from .rules import local_gathering_and_repulsion, local_manual_with_shape_consensus

__all__ = ["SimulationConfig", "SwarmSimulator", "local_gathering_and_repulsion", "local_manual_with_shape_consensus"]
