"""Génère les schémas reproductibles utilisés dans le mémoire LaTeX."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUTPUT = Path("latex/figures")
BLUE = "#1f77b4"
ORANGE = "#e67e22"
DARK = "#243447"
LIGHT = "#eaf2f8"


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_robot(axis: plt.Axes, x: float, y: float, label: str, highlighted: bool = False) -> None:
    color = ORANGE if highlighted else BLUE
    axis.add_patch(Circle((x, y), 0.17, facecolor=color, edgecolor=DARK, linewidth=1.3, zorder=3))
    axis.add_patch(Circle((x - 0.055, y + 0.035), 0.018, color="white", zorder=4))
    axis.add_patch(Circle((x + 0.055, y + 0.035), 0.018, color="white", zorder=4))
    axis.text(x, y - 0.025, label, color="white", ha="center", va="center", fontsize=8, weight="bold", zorder=5)


def swarm_local_communication() -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    axis.set_xlim(0, 9)
    axis.set_ylim(0, 5.4)
    axis.set_aspect("equal")
    axis.axis("off")

    center = (4.4, 2.7)
    axis.add_patch(Circle(center, 1.85, facecolor=LIGHT, edgecolor=ORANGE, linestyle="--", linewidth=1.8, alpha=0.8))
    axis.text(4.4, 4.64, "portée de communication $r_c$", color=DARK, ha="center", fontsize=10)
    robots = [(4.4, 2.7, "$i$", True), (3.15, 3.55, "$j_1$", False), (5.65, 3.48, "$j_2$", False), (3.35, 1.45, "$j_3$", False), (5.62, 1.6, "$j_4$", False), (1.15, 3.2, "$k$", False), (7.75, 2.7, "$l$", False)]
    for x, y, label, highlighted in robots:
        if label.startswith("$j"):
            axis.plot([center[0], x], [center[1], y], color="#839192", linewidth=1.2, zorder=1)
        draw_robot(axis, x, y, label, highlighted)
    axis.text(4.4, 0.38, r"Le robot $i$ reçoit uniquement les messages de ses voisins $j\in\mathcal{N}(i)$.", ha="center", fontsize=10, color=DARK)
    save(figure, "swarm-local-communication.png")


def rounded_box(axis: plt.Axes, x: float, y: float, width: float, height: float, text: str, color: str) -> None:
    axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=color, edgecolor=DARK, linewidth=1.2))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9, color=DARK, wrap=True)


def message_passing() -> None:
    figure, axis = plt.subplots(figsize=(10, 4.2))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4.2)
    axis.axis("off")
    boxes = [
        (0.25, 1.35, 1.55, 1.1, "État local\n$v_i, p_i$, consigne", "#d6eaf8"),
        (2.45, 1.35, 1.55, 1.1, r"Messages" + "\n" + r"$\phi(h_i,h_j,p_j-p_i)$", "#fdebd0"),
        (4.65, 1.35, 1.55, 1.1, "Agrégation\npar moyenne", "#d5f5e3"),
        (6.85, 1.35, 1.55, 1.1, r"Mise à jour" + "\n" + r"$\psi(h_i,\bar m_i)$", "#e8daef"),
        (8.8, 1.35, 1.0, 1.1, "Commande\n$u_i$", "#f9e79f"),
    ]
    for x, y, width, height, text, color in boxes:
        rounded_box(axis, x, y, width, height, text, color)
    for start, end in [(1.8, 2.45), (4.0, 4.65), (6.2, 6.85), (8.4, 8.8)]:
        axis.add_patch(FancyArrowPatch((start, 1.9), (end, 1.9), arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DARK))
    axis.text(5, 3.35, "Un même bloc est exécuté par chaque robot et répété sur deux tours de messages.", ha="center", fontsize=11, color=DARK)
    axis.text(5, 0.55, "Les informations distantes ne sont accessibles qu'au travers de plusieurs sauts dans le graphe.", ha="center", fontsize=9.5, color=DARK)
    save(figure, "gnn-message-passing.png")


def experimental_pipeline() -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 4.5)
    axis.axis("off")
    boxes = [
        (0.25, 2.4, 1.55, 0.9, "Forme cible\nimage", "#d6eaf8"),
        (2.25, 2.4, 1.55, 0.9, "Nuage de\npoints", "#d5f5e3"),
        (4.25, 2.4, 1.55, 0.9, "GNN local\npartagé", "#fdebd0"),
        (6.25, 2.4, 1.55, 0.9, "Simulateur\n2D", "#e8daef"),
        (8.25, 2.4, 1.35, 0.9, "Métriques\net perte", "#f9e79f"),
    ]
    for x, y, width, height, text, color in boxes:
        rounded_box(axis, x, y, width, height, text, color)
    for start, end in [(1.8, 2.25), (3.8, 4.25), (5.8, 6.25), (7.8, 8.25)]:
        axis.add_patch(FancyArrowPatch((start, 2.85), (end, 2.85), arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DARK))
    axis.add_patch(FancyArrowPatch((8.93, 2.38), (5.02, 1.02), connectionstyle="arc3,rad=-0.3", arrowstyle="->", mutation_scale=14, linewidth=1.3, color=ORANGE))
    axis.text(6.9, 0.7, "rétropropagation pendant l'entraînement", color=ORANGE, ha="center", fontsize=9.5)
    axis.text(5, 3.85, "Chaîne expérimentale reproductible : données, contrôle local, simulation et évaluation.", ha="center", fontsize=11, color=DARK)
    save(figure, "experimental-pipeline.png")


if __name__ == "__main__":
    swarm_local_communication()
    message_passing()
    experimental_pipeline()
