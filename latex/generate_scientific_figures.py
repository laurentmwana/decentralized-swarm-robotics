"""Génère sept figures scientifiques déterministes, sans entraînement."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = ROOT / "images"
FIGURE_NAMES = (
    "neuron_and_message_passing",
    "activation_functions",
    "gnn_current_architecture",
    "shape_loss_geometry",
    "differentiable_training_flow",
    "success_criterion_comparison",
    "controller_information_matrix",
)
SOURCE_REFERENCES = {
    "neuron_and_message_passing": (
        "src/models/encoder.py", "src/models/message.py", "src/models/aggregator.py",
        "src/models/gnn.py",
        "Figure specification: conceptual schematic, no parameter values, no measured result",
    ),
    "activation_functions": ("src/models/encoder.py", "src/models/decoder.py"),
    "gnn_current_architecture": (
        "src/models/encoder.py", "src/models/message.py", "src/models/aggregator.py",
        "src/models/gnn.py", "src/models/decoder.py", "src/controllers/gnn.py",
        "config/model.yaml", "config/training.yaml", "config/simulator.yaml",
    ),
    "shape_loss_geometry": ("src/training/losses.py", "config/training.yaml"),
    "differentiable_training_flow": (
        "src/training/rollout.py", "src/training/trainer.py", "config/training.yaml",
    ),
    "success_criterion_comparison": (
        "src/metrics/success.py", "src/metrics/chamfer.py",
        "Figure specification: historical instant criterion, threshold 0.5",
    ),
    "controller_information_matrix": (
        "src/controllers/flocking.py", "src/controllers/trivial.py",
        "src/controllers/manual_shape.py", "src/controllers/gnn.py",
        "config/model.yaml", "config/training.yaml",
        "Figure specification: only historical GNN and flocking evaluated in pilot",
    ),
}

INK = "#243447"
BLUE = "#245fa0"
TEAL = "#137f82"
ORANGE = "#b65a21"
PURPLE = "#78539c"
GRAY = "#657386"
PALE_BLUE = "#edf3fa"
PALE_TEAL = "#ecf6f3"
PALE_ORANGE = "#fcf2e9"
GRID = "#dce2e9"
WIDTH = 10.5

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 17,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.edgecolor": GRAY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.8,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "pdf.use14corefonts": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, values)


def radial_projection(raw: np.ndarray, speed_limit: float) -> np.ndarray:
    if speed_limit <= 0:
        raise ValueError("speed_limit must be positive")
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    factor = np.minimum(1.0, speed_limit / np.maximum(norms, 1e-9))
    return raw * factor


def synthetic_success_curves() -> tuple[np.ndarray, ...]:
    time = np.arange(101)
    old_curve = np.interp(time, [0, 30, 55, 75, 100], [0.90, 0.50, 0.30, 0.55, 0.70])
    stable_curve = np.interp(time, [0, 35, 45, 55, 70, 100], [0.60, 0.15, 0.12, 0.24, 0.15, 0.10])
    final_only_curve = 0.15 + 0.30 * (1.0 - time / 100.0)
    return time, old_curve, stable_curve, final_only_curve


def stabilized_step(curve: np.ndarray, threshold: float) -> int | None:
    remaining_ok = np.logical_and.accumulate((curve <= threshold)[::-1])[::-1]
    candidates = np.flatnonzero(remaining_ok)
    return int(candidates[0]) if candidates.size else None


def verify_mathematics() -> None:
    inputs = np.array([-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0])
    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 3.0])
    np.testing.assert_array_equal(relu(inputs), expected)
    print("PASS: exact ReLU values for [-3, -1, -0.25, 0, 0.25, 1, 3].")
    grid = np.linspace(-10.0, 10.0, 301)
    horizontal, vertical = np.meshgrid(grid, grid)
    raw = np.tanh(np.column_stack((horizontal.ravel(), vertical.ravel())))
    assert np.linalg.norm(np.tanh([3.0, 3.0])) > 1.0
    for limit in (0.25, 1.0, 2.0):
        projected = radial_projection(raw, limit)
        norms = np.linalg.norm(projected, axis=1)
        assert np.all(norms <= limit + 1e-12)
        inside = np.linalg.norm(raw, axis=1) <= limit
        np.testing.assert_array_equal(projected[inside], raw[inside])
        np.testing.assert_array_equal(radial_projection(np.zeros((1, 2)), limit), np.zeros((1, 2)))
        print(f"PASS: radial norm bound, vmax={limit:g}, maximum={norms.max():.12g}; no model.")
    time, old, stable, final_only = synthetic_success_curves()
    assert int(time[np.flatnonzero(old[:-1] <= 0.5)[0]]) == 30
    assert old[-1] > 0.5
    assert stabilized_step(stable, 0.15) == 70
    assert stabilized_step(final_only, 0.15) == 100
    assert stable[35] <= 0.15 and stable[55] > 0.15
    print("PASS: illustrative events: old=30 (final above threshold), stabilized=70, final-only=100.")


def preflight_assets() -> None:
    if not ROOT.is_dir() or not OUTPUT_DIRECTORY.is_dir():
        raise FileNotFoundError("The latex and latex/images parents must already exist")
    if OUTPUT_DIRECTORY.is_symlink():
        raise ValueError("Refusing a symlinked output directory")
    for name in FIGURE_NAMES:
        for extension in ("pdf", "png"):
            path = OUTPUT_DIRECTORY / f"{name}.{extension}"
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"Refusing nonregular output: {path}")
            status = f"existing, {path.stat().st_size} bytes" if path.exists() else "absent"
            print(f"ASSET before write: {path} ({status})")


def header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=21, weight="bold")


def canvas(height: float) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def box(ax: plt.Axes, x: float, y: float, width: float, height: float,
        text: str, color: str = PALE_BLUE, fontsize: float = 16.5) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=color, edgecolor=GRID, linewidth=1.2,
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            fontsize=fontsize, linespacing=1.4)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
          color: str = GRAY, linewidth: float = 1.8, linestyle: str = "-",
          mutation_scale: float = 16, shrink: float = 0) -> None:
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        color=color, linewidth=linewidth, linestyle=linestyle,
        shrinkA=shrink, shrinkB=shrink,
    ))


def routed_arrow(ax: plt.Axes, points: list[tuple[float, float]],
                 color: str = GRAY, linestyle: str = "-") -> None:
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, linewidth=1.8, linestyle=linestyle)
    arrow(ax, points[-2], points[-1], color=color, linestyle=linestyle)


def style_plot(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, colors=GRAY)


def neuron_and_message_passing() -> plt.Figure:
    height = 5.6
    fig, ax = canvas(height)
    header(fig, "Du neurone formel au tour de passage de messages")
    ratio = WIDTH / height

    def node(x: float, y: float, radius: float, text: str, color: str = PALE_BLUE,
             edge: str = BLUE, fontsize: float = 15.5, weight: str = "normal") -> None:
        ax.add_patch(Ellipse((x, y), 2 * radius, 2 * radius * ratio,
                             facecolor=color, edgecolor=edge, linewidth=1.6, zorder=4))
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight, zorder=5)

    ax.text(0.255, 0.885, "(a) Un neurone formel", ha="center", va="top", weight="bold", fontsize=18)
    hub_a = (0.248, 0.560)
    for y, label, weight_label in ((0.700, "x₁", "w₁"), (0.560, "x₂", "w₂"), (0.420, "x₃", "w₃")):
        node(0.072, y, 0.025, label)
        arrow(ax, (0.098, y), (hub_a[0] - 0.031, hub_a[1]), color=GRAY, linewidth=1.5, mutation_scale=13)
        ax.text(0.163, (y + hub_a[1]) / 2 + 0.020, weight_label, ha="center", fontsize=14, color=GRAY)
    node(*hub_a, 0.029, "Σ", color="white", fontsize=19)
    box(ax, 0.312, 0.505, 0.122, 0.110, "Activation f", color=PALE_TEAL, fontsize=14)
    arrow(ax, (hub_a[0] + 0.030, hub_a[1]), (0.305, hub_a[1]), color=GRAY)
    node(0.478, hub_a[1], 0.025, "y", color=PALE_ORANGE, edge=ORANGE)
    arrow(ax, (0.437, hub_a[1]), (0.450, hub_a[1]), color=GRAY)
    ax.text(0.255, 0.330, "z = w₁x₁ + w₂x₂ + w₃x₃ + b,   y = f(z)",
            ha="center", va="top", fontsize=14.5)
    ax.text(0.255, 0.258, "Chaque entrée agit par son poids.\nLes poids s'ajustent par descente de gradient.",
            ha="center", va="top", fontsize=13.5, color=GRAY, linespacing=1.5)

    ax.text(0.755, 0.885, "(b) Un tour de passage de messages", ha="center", va="top",
            weight="bold", fontsize=18)
    hub = (0.598, 0.560)
    ax.add_patch(Ellipse(hub, 0.170, 0.170 * ratio, facecolor="white", edgecolor=GRAY,
                         linewidth=1.3, linestyle="--", zorder=1))
    ax.text(hub[0], 0.742, "portée de communication", ha="center", va="top", fontsize=13.5, color=GRAY)
    node(*hub, 0.027, "i", color=PALE_BLUE, weight="bold", fontsize=17)
    for angle, label in ((130, "j₁"), (200, "j₂"), (320, "j₃")):
        rad = np.deg2rad(angle)
        x = hub[0] + 0.062 * np.cos(rad)
        y = hub[1] + 0.105 * np.sin(rad)
        node(x, y, 0.021, label, color="white", edge=TEAL, fontsize=14)
        arrow(ax, (x, y), hub, color=TEAL, linewidth=1.5, mutation_scale=12, shrink=22)
    ax.text(0.505, 0.678, "m(i,j)", ha="center", fontsize=13, color=TEAL,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    rad = np.deg2rad(215)
    kx, ky = hub[0] + 0.105 * np.cos(rad), hub[1] + 0.175 * np.sin(rad)
    node(kx, ky, 0.021, "k", color="#f2f3f5", edge=GRAY, fontsize=14)
    ax.text(kx, ky - 0.058, "hors portée", ha="center", va="top", fontsize=13.5, color=GRAY)
    for index, (label, face) in enumerate((("Agrégation\nsymétrique", PALE_TEAL),
                                           ("Mise à jour de hᵢ", PALE_BLUE),
                                           ("Commande locale", PALE_ORANGE))):
        bottom = 0.655 - index * 0.168
        box(ax, 0.780, bottom, 0.190, 0.100, label, color=face, fontsize=14.5)
        if index:
            arrow(ax, (0.875, bottom + 0.132), (0.875, bottom + 0.106), color=GRAY, linewidth=1.6)
    arrow(ax, (hub[0] + 0.030, hub[1] - 0.015), (0.773, 0.660), color=TEAL)
    ax.text(0.755, 0.258, "Un message par arête du voisinage,\npuis une combinaison symétrique.",
            ha="center", va="top", fontsize=13.5, color=GRAY, linespacing=1.5)
    ax.text(0.50, 0.082, "Schéma conceptuel : le nombre de voisins change, les poids restent partagés entre les robots.",
            ha="center", va="top", fontsize=14.5)
    return fig


def activation_functions() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 5.6))
    header(fig, "Fonctions d'activation", "Courbes mathématiques exactes · rôles distincts dans le GNN")
    left = fig.add_axes([0.085, 0.27, 0.375, 0.49])
    right = fig.add_axes([0.595, 0.27, 0.375, 0.49])
    values = np.linspace(-3, 3, 1201)
    for ax in (left, right):
        style_plot(ax)
        ax.set_xlim(-3, 3)
        ax.set_xticks([-3, 0, 3])
        ax.set_xlabel("Préactivation z")
        ax.set_ylabel("Sortie")
        ax.axvline(0, color=GRAY, linewidth=1)
        ax.axhline(0, color=GRAY, linewidth=1)
    left.set_title("(a) ReLU · couches cachées", pad=18)
    left.axvspan(-3, 0, color=PALE_BLUE)
    left.plot(values, relu(values), color=BLUE)
    left.set_ylim(-0.25, 3.35)
    left.set_yticks([0, 1, 2, 3])
    left.text(-2.72, 0.83, "z < 0\ninactive", color=BLUE, fontsize=16)
    left.text(0.85, 0.35, "z > 0\nlinéaire", color=BLUE, fontsize=16)
    left.text(-2.72, 2.83, r"$f(z)=\max(0,z)$", fontsize=18)
    left.scatter([0], [0], color=BLUE, s=35, zorder=5)
    right.set_title("(b) tanh · sortie du décodeur", pad=18)
    right.plot(values, np.tanh(values), color=TEAL)
    right.set_ylim(-1.3, 1.3)
    right.set_yticks([-1, 0, 1])
    for bound in (-1, 1):
        right.axhline(bound, color=TEAL, linewidth=1.1, linestyle="--", alpha=0.6)
    right.text(-2.72, 0.42, r"$f(z)=\tanh(z)$", fontsize=18, color=TEAL)
    fig.text(0.275, 0.095, "ReLU n'est pas différentiable en 0.", ha="center", fontsize=16)
    fig.text(0.76, 0.095, "tanh borne les composantes, pas la norme.\nLa projection radiale impose la borne finale.",
             ha="center", va="center", fontsize=15.5, linespacing=1.4)
    return fig


def gnn_current_architecture() -> plt.Figure:
    fig, ax = canvas(9.3)
    header(fig, "Architecture actuelle du contrôleur GNN", "Dimensions par robot · agrégation moyenne · activations ReLU")
    box(ax, 0.04, 0.75, 0.555, 0.125,
        "Vitesse : 2 | géométrie optionnelle : 2\nDirection unitaire vers le point cible le plus proche\nID → plongement : 16 (64 identifiants de forme)\nConcaténation : 20 ; sans géométrie : 18", fontsize=15.3)
    box(ax, 0.675, 0.75, 0.28, 0.125,
        "Encodeur\n20 → 64 · ReLU\n18 → 64 sans géométrie", fontsize=16)
    arrow(ax, (0.605, 0.8125), (0.665, 0.8125))
    ax.add_patch(FancyBboxPatch((0.04, 0.32), 0.915, 0.37,
                              boxstyle="round,pad=0.006,rounding_size=0.012",
                              facecolor="white", edgecolor=BLUE, linewidth=1.5))
    ax.text(0.50, 0.666, "Un tour détaillé de communication", ha="center", weight="bold", fontsize=18)
    routed_arrow(ax, [(0.815, 0.741), (0.815, 0.708), (0.18, 0.708), (0.18, 0.618)], color=BLUE)
    ax.text(0.44, 0.713, "État latent hᵢ : 64", ha="center", va="bottom", fontsize=16, color=BLUE)
    box(ax, 0.062, 0.43, 0.305, 0.18,
        "Message par voisin\nhᵢ : 64  |  hⱼ : 64\nPosition relative : 2\n130 → 64 · ReLU → 64", fontsize=16)
    box(ax, 0.423, 0.452, 0.16, 0.136,
        "Moyenne\nmasquée\nAgrégat : 64", color=PALE_TEAL, fontsize=16)
    box(ax, 0.638, 0.43, 0.292, 0.18,
        "Mise à jour\nhᵢ : 64 | agrégat : 64\nConcaténation : 128\n128 → 64 · ReLU → 64", fontsize=16)
    arrow(ax, (0.378, 0.52), (0.413, 0.52))
    arrow(ax, (0.594, 0.52), (0.628, 0.52))
    ax.text(0.48, 0.413, "Masque d'adjacence ; isolé : agrégat nul", ha="center", va="top", fontsize=15, color=TEAL)
    routed_arrow(ax, [(0.18, 0.708), (0.022, 0.708), (0.022, 0.355), (0.758, 0.355)],
                 color=BLUE, linestyle="--")
    ax.text(0.365, 0.36, "Résidu : ajout de hᵢ", ha="center", va="bottom", color=BLUE, fontsize=16)
    ax.add_patch(Circle((0.785, 0.355), 0.021, facecolor=PALE_BLUE, edgecolor=BLUE, linewidth=1.5))
    ax.text(0.785, 0.355, "+", ha="center", va="center", fontsize=20, color=BLUE)
    arrow(ax, (0.785, 0.42), (0.785, 0.378), color=BLUE)
    arrow(ax, (0.808, 0.355), (0.927, 0.355), color=BLUE)
    ax.text(0.865, 0.374, "hᵢ suivant : 64", ha="center", fontsize=15, color=BLUE)
    ax.text(0.50, 0.288, "Répéter L = 1, 2 ou 3 tours · paramètres distincts entre tours", ha="center", fontsize=16)
    routed_arrow(ax, [(0.932, 0.355), (0.982, 0.355), (0.982, 0.256), (0.265, 0.256), (0.265, 0.238)], color=BLUE)
    box(ax, 0.04, 0.12, 0.455, 0.108,
        "Décodeur : 64 → 64 · ReLU → 2\nbrut = tanh(sortie)", fontsize=16.5)
    box(ax, 0.565, 0.105, 0.39, 0.123,
        "Projection radiale\nbrut × min(1, vₘₐₓ / ‖brut‖)\nSi brut = 0 : commande = 0\n‖commande‖ ≤ vₘₐₓ ; défaut : 1", color=PALE_TEAL, fontsize=14.5)
    arrow(ax, (0.506, 0.174), (0.553, 0.174), color=BLUE)
    ax.text(0.50, 0.074, "Poids partagés entre robots et pas de temps ; distincts entre tours.", ha="center", fontsize=16)
    ax.text(0.50, 0.034, "Sans communication : 2 tours et leurs paramètres conservés ; graphe et agrégats nuls.",
            ha="center", fontsize=15.5, color=TEAL)
    return fig


def shape_loss_geometry() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 8.5))
    header(fig, "Géométrie des pertes de forme et de couverture", "Exemple synthétique fixe · mêmes robots P et même cible éparse Q")
    angles = np.arange(12) * (2 * np.pi / 12)
    target = np.column_stack((np.cos(angles), np.sin(angles)))
    robots = np.array([[-0.55, 0.0], [-0.64, 0.28], [-0.45, 0.37],
                       [-0.31, 0.05], [-0.62, -0.20], [-0.40, -0.25]])
    distances = np.linalg.norm(robots[:, None, :] - target[None, :, :], axis=2)
    axes = [fig.add_axes([0.085, 0.475, 0.375, 0.30]), fig.add_axes([0.59, 0.475, 0.375, 0.30])]
    for index, ax in enumerate(axes):
        ax.set_title("(a) Forme : P → Q proche" if index == 0 else "(b) Couverture : Q → P proche", pad=14)
        ax.set(xlim=(-1.22, 1.22), ylim=(-1.22, 1.22), aspect="equal", xlabel="x", ylabel="y")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        style_plot(ax)
        if index == 0:
            for start, end in zip(robots, target[distances.argmin(axis=1)]):
                arrow(ax, tuple(start), tuple(end), color=ORANGE, linewidth=1.5, shrink=7, mutation_scale=13)
        else:
            for start, end in zip(target, robots[distances.argmin(axis=0)]):
                arrow(ax, tuple(start), tuple(end), color=TEAL, linewidth=1.5, shrink=7, mutation_scale=13)
        ax.scatter(target[:, 0], target[:, 1], s=67, facecolor="white", edgecolor=TEAL, linewidth=2, zorder=4)
        ax.scatter(robots[:, 0], robots[:, 1], s=57, color=ORANGE, edgecolor="white", linewidth=0.7, zorder=5)
    handles = [
        Line2D([], [], linestyle="none", marker="o", color=ORANGE, markersize=8, label="Robots P"),
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=TEAL,
               markeredgewidth=2, markersize=8, label="Points cibles Q"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.857), ncol=2, frameon=False)
    fig.text(0.5, 0.379, "Les flèches représentent des distances, pas des trajectoires ni une bijection.",
             ha="center", fontsize=15.5)
    ax = fig.add_axes([0, 0, 1, 1], zorder=-1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    box(ax, 0.04, 0.267, 0.92, 0.09,
        "Forme / couverture : distances non carrées, moyennées sur P / Q, divisées\npar le rayon cible ; puis moyenne sur les 10 derniers états.", fontsize=16)
    box(ax, 0.04, 0.148, 0.92, 0.09,
        "Collisions : somme des (1 − d/r)² pour les paires non ordonnées à d < r,\npuis moyenne sur tous les états. Pas de normalisation par le nombre de paires.",
        color=PALE_ORANGE, fontsize=15.7)
    box(ax, 0.04, 0.03, 0.92, 0.09,
        "Mouvement : moyenne de ‖commande‖² / vₘₐₓ² sur toutes les commandes.\nCe n'est pas une pénalité d'oscillation.", color=PALE_TEAL, fontsize=16)
    return fig


def differentiable_training_flow() -> plt.Figure:
    fig, ax = canvas(8.5)
    header(fig, "Apprentissage différentiable : déroulement du protocole", "Profil standard · schéma seulement, aucune optimisation exécutée")
    box(ax, 0.04, 0.720, 0.28, 0.115,
        "Épisodes d'entraînement\nPositions initiales + cible", fontsize=16)
    box(ax, 0.40, 0.690, 0.555, 0.18, "", color=PALE_BLUE)
    ax.text(0.677, 0.832, "Déroulement séquentiel sur T pas", ha="center", fontsize=17, weight="bold")
    ax.text(0.677, 0.790, "Même GNN partagé à chaque pas", ha="center", fontsize=16)
    ax.text(0.677, 0.733, "$p_0$ → GNN → $p_1$ → … → $p_T$", ha="center", fontsize=18, color=BLUE)
    arrow(ax, (0.332, 0.777), (0.388, 0.777), color=BLUE)
    box(ax, 0.40, 0.545, 0.555, 0.095,
        "Perte de chaque épisode\nForme + couverture + collisions + mouvement", fontsize=16)
    arrow(ax, (0.677, 0.679), (0.677, 0.652), color=BLUE)
    box(ax, 0.62, 0.40, 0.335, 0.09, "Moyenne du lot\n2 épisodes", fontsize=16.5)
    routed_arrow(ax, [(0.677, 0.534), (0.677, 0.517), (0.787, 0.517), (0.787, 0.50)], color=BLUE)
    box(ax, 0.315, 0.40, 0.25, 0.09,
        "Rétropropagation\nÉcrêtage : norme ≤ 1", fontsize=15.7)
    box(ax, 0.04, 0.40, 0.22, 0.09, "Adam\nMise à jour des poids", color=PALE_TEAL, fontsize=16)
    arrow(ax, (0.607, 0.445), (0.577, 0.445), color=BLUE)
    arrow(ax, (0.303, 0.445), (0.272, 0.445), color=BLUE)
    routed_arrow(ax, [(0.15, 0.50), (0.15, 0.663), (0.427, 0.663), (0.427, 0.679)], color=TEAL)
    ax.text(0.15, 0.60, "Itération\nsuivante", ha="center", fontsize=16, color=TEAL,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
    box(ax, 0.04, 0.212, 0.505, 0.12,
        "Validation réservée, hors gradient\nToutes les 50 itérations et à la fin\nCibles / graines de validation ; jamais le test", color=PALE_TEAL, fontsize=16)
    arrow(ax, (0.15, 0.388), (0.15, 0.343), color=TEAL)
    box(ax, 0.62, 0.212, 0.335, 0.12,
        "Meilleur état sélectionné\nChamfer moyen de validation\nDernier état aussi conservé", color=PALE_TEAL, fontsize=15.7)
    arrow(ax, (0.557, 0.272), (0.608, 0.272), color=TEAL)
    box(ax, 0.04, 0.035, 0.915, 0.125,
        "Adjacence booléenne et choix argmin : décisions discrètes non différentiées.\nGradients dans les positions et commandes : chemins localement par morceaux,\npas de différentiabilité globale des changements de voisinage.", color=PALE_ORANGE, fontsize=15.7)
    return fig


def success_criterion_comparison() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.5))
    header(fig, "Réussite instantanée ou franchissement stabilisé ?", "Courbes synthétiques pédagogiques, déterministes · aucune mesure expérimentale")
    left = fig.add_axes([0.085, 0.29, 0.38, 0.50])
    right = fig.add_axes([0.59, 0.29, 0.38, 0.50])
    time, old, stable, final_only = synthetic_success_curves()
    for ax in (left, right):
        style_plot(ax)
        ax.set(xlim=(0, 103), ylim=(0, 1.0), xlabel="Pas t")
        ax.set_xticks([0, 50, 100], ["0", "50", "T = 100"])
        ax.set_yticks([0, 0.5, 1], ["0", "0,5", "1"])
    left.set_ylabel("Distance historique")
    right.set_ylabel("Distance normalisée")
    left.set_title("(a) Ancien · seuil 0,50", pad=18)
    right.set_title("(b) Nouveau · seuil 0,15", pad=18)
    left.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.6)
    left.plot(time, old, color=ORANGE)
    left.scatter([30, 100], old[[30, 100]], color=ORANGE, s=55, zorder=5)
    left.annotate("Premier passage : t = 30\nSuccès avant T", xy=(30, old[30]), xytext=(5, 0.18),
                  fontsize=15, arrowprops={"arrowstyle": "->", "color": ORANGE, "connectionstyle": "arc3,rad=-0.15"})
    left.annotate("À T : au-dessus\ndu seuil", xy=(100, old[100]), xytext=(43, 0.87),
                  fontsize=15, arrowprops={"arrowstyle": "->", "color": ORANGE})
    left.text(2, 0.465, "Seuil 0,50", va="top", color=GRAY, fontsize=15)
    right.axvspan(70, 100, color=PALE_TEAL)
    right.axhline(0.15, color=GRAY, linestyle="--", linewidth=1.6)
    right.plot(time, stable, color=TEAL, label="Suffixe dès 70")
    right.plot(time, final_only, color=PURPLE, linestyle="--", label="Seulement à T")
    right.scatter([35, 70], stable[[35, 70]], color=TEAL, s=48, zorder=5)
    right.scatter([100], [final_only[-1]], color=PURPLE, s=55, zorder=6)
    right.annotate("Passage temporaire\nnon retenu", xy=(35, stable[35]), xytext=(1, 0.025),
                   fontsize=14.5, arrowprops={"arrowstyle": "->", "color": TEAL})
    right.annotate("Suffixe : t = 70\nJusqu'à T", xy=(70, stable[70]), xytext=(3, 0.74),
                   va="top", fontsize=14.5, arrowprops={"arrowstyle": "->", "color": TEAL})
    right.annotate("Suffixe réduit à T\n(autre courbe)", xy=(100, final_only[-1]), xytext=(52, 0.56),
                   va="top", fontsize=14.5, color=PURPLE, arrowprops={"arrowstyle": "->", "color": PURPLE})
    right.text(3, 0.182, "Seuil 0,15", color=GRAY, fontsize=14.5)
    right.legend(loc="upper right", frameon=False, fontsize=14.5)
    fig.text(0.5, 0.16, "Nouveau : premier t tel que d(s) ≤ 0,15 pour tout s de t à T (T inclus).", ha="center", fontsize=16)
    fig.text(0.5, 0.07, "Un passage au seul dernier pas est admis ; il ne prouve pas\nune convergence asymptotique au-delà de l'horizon observé.",
             ha="center", fontsize=16, linespacing=1.4)
    return fig


def controller_information_matrix() -> plt.Figure:
    fig, ax = canvas(6.7)
    header(fig, "Informations accessibles aux contrôleurs", "Les informations de cible ne sont pas équivalentes entre variantes")
    boundaries = np.array([0.03, 0.30, 0.42, 0.555, 0.87, 0.97])
    labels = ["Contrôleur", "Voisins", "ID de\nforme", "Géométrie de cible", "Poids\najustables"]
    header_top, header_bottom = 0.846, 0.746
    for index, label in enumerate(labels):
        start, end = boundaries[index:index + 2]
        ax.add_patch(Rectangle((start, header_bottom), end - start, header_top - header_bottom,
                               facecolor=INK, edgecolor="white", linewidth=1))
        ax.text((start + end) / 2, (header_top + header_bottom) / 2, label,
                ha="center", va="center", color="white", fontsize=12 if index == 4 else 16, weight="bold")
    rows = [
        ["Regroupement\n(flocking)", "Oui", "Non", "Aucune", "Non"],
        ["Règle manuelle", "Oui", "Non", "Déplacement au plus proche\n(direction + distance)", "Non"],
        ["GNN\nidentifiant seul", "Oui", "Oui", "Aucune", "Oui"],
        ["GNN\navec géométrie", "Oui", "Oui", "Direction unitaire\nvers le plus proche", "Oui"],
        ["GNN sans\ncommunication", "Non", "Oui", "Direction unitaire\nvers le plus proche", "Oui"],
    ]
    row_height = 0.102
    for row_index, row in enumerate(rows):
        top = header_bottom - row_index * row_height
        bottom = top - row_height
        for column, text in enumerate(row):
            start, end = boundaries[column:column + 2]
            background = "white" if row_index % 2 == 0 else PALE_BLUE
            if column in (1, 2, 4):
                background = PALE_TEAL if text == "Oui" else "#f2f3f5"
            ax.add_patch(Rectangle((start, bottom), end - start, row_height,
                                   facecolor=background, edgecolor=GRID, linewidth=0.9))
            ax.text((start + end) / 2, (top + bottom) / 2, text, ha="center", va="center",
                    fontsize=16 if column != 3 else 15.5,
                    color=TEAL if text == "Oui" else INK, linespacing=1.35)
    ax.text(0.50, 0.192, "Poids ajustables : paramètres à apprendre, sans entraînement actuel attesté.",
            ha="center", fontsize=15.5, color=GRAY)
    ax.text(0.05, 0.137, "Pilote évalué : ancien GNN et regroupement uniquement.", fontsize=16, weight="bold")
    ax.text(0.05, 0.098, "Autres variantes : nouvellement préparées ; aucun entraînement actuel.\nGNN sans communication : 2 tours conservés, mais graphe et agrégats nuls.",
            va="top", fontsize=15.5, linespacing=1.4)
    return fig


def save_figure(fig: plt.Figure, name: str) -> None:
    if name not in FIGURE_NAMES:
        raise ValueError(f"Unauthorized figure name: {name}")
    subject = "Original deterministic diagram; code provenance: " + "; ".join(SOURCE_REFERENCES[name])
    fig.savefig(OUTPUT_DIRECTORY / f"{name}.pdf", format="pdf", metadata={
        "Title": name,
        "Author": "Original scientific figures for swarm-gnn",
        "Subject": subject,
        "Creator": "generate_scientific_figures.py / Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    })
    fig.savefig(OUTPUT_DIRECTORY / f"{name}.png", format="png", dpi=300, metadata={
        "Title": name,
        "Description": subject,
        "Software": "generate_scientific_figures.py / Matplotlib",
    })
    plt.close(fig)
    for extension in ("pdf", "png"):
        path = OUTPUT_DIRECTORY / f"{name}.{extension}"
        assert path.is_file() and path.stat().st_size > 0
        print(f"SAVED: {path} ({path.stat().st_size} bytes)")


def main() -> None:
    preflight_assets()
    verify_mathematics()
    builders = (
        neuron_and_message_passing,
        activation_functions,
        gnn_current_architecture,
        shape_loss_geometry,
        differentiable_training_flow,
        success_criterion_comparison,
        controller_information_matrix,
    )
    for name, builder in zip(FIGURE_NAMES, builders, strict=True):
        save_figure(builder(), name)
    print("DONE: seven vector PDFs and seven 300-dpi PNGs; no training, optimizer, model or random draws.")


if __name__ == "__main__":
    main()
