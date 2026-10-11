"""Visualisation statique et animation de trajectoires d'essaim."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def plot_state(positions: np.ndarray, active: np.ndarray, arena_size: tuple[float, float]):
    """Affiche un état d'essaim et retourne la figure matplotlib."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.scatter(positions[active, 0], positions[active, 1], color="tab:blue", label="actif")
    axis.scatter(positions[~active, 0], positions[~active, 1], color="tab:red", marker="x", label="en panne")
    axis.set(xlim=(0, arena_size[0]), ylim=(0, arena_size[1]), aspect="equal", xlabel="x", ylabel="y")
    axis.legend()
    return figure


def animate_history(
    positions_history: Sequence[np.ndarray], active_history: Sequence[np.ndarray], arena_size: tuple[float, float]
):
    """Construit une animation matplotlib d'une séquence d'états."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    figure, axis = plt.subplots()
    scatter = axis.scatter([], [])
    axis.set(xlim=(0, arena_size[0]), ylim=(0, arena_size[1]), aspect="equal", xlabel="x", ylabel="y")

    def update(frame: int):
        positions, active = positions_history[frame], active_history[frame]
        scatter.set_offsets(positions)
        scatter.set_color(np.where(active, "tab:blue", "tab:red"))
        return (scatter,)

    return FuncAnimation(figure, update, frames=len(positions_history), interval=80, blit=True)
