"""Figures du mémoire construites uniquement à partir de données réelles.

Aucune figure n'est dessinée à la main : chaque panneau lit un fichier produit par le code
du projet (trajectoires enregistrées, journaux d'entraînement, épisodes bruts) ou exécute ce
code (rendu des figures cibles). Usage, depuis la racine du dépôt :

    .venv/Scripts/python.exe latex/generate_real_figures.py

Sources :
- outputs/evaluation_pilot/ : campagne pilote de l'implémentation actuelle (commit 2c86290,
  26 octobre 2026), 9 600 épisodes et 960 trajectoires enregistrées ;
- outputs/training/ : journaux d'entraînement (losses.csv, validation.csv) ;
- src/shapes/ : catalogue des figures, régénéré dans un dossier temporaire.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from src.common.config import ConfigLoader
from src.metrics import convergence_step
from src.shapes.index import build_sources, generate_index

PILOT = REPOSITORY / "outputs" / "evaluation_pilot"
TRAINING = REPOSITORY / "outputs" / "training"
OUTPUT = Path(__file__).resolve().parent / "images"

R_C = 1.5           # config/simulator.yaml
THRESHOLD = 0.15    # config/metrics.yaml
EPISODE_SEEDS = [555000, 555001, 555002, 555003, 555004]

INK = "#1f2933"
GRID = "#e3e7ec"
TARGET = "#2f8f5b"
INACTIVE = "#9aa5b1"
COLORS = {"flocking": "#8a94a0", "manual": "#c0622b", "no_comm": "#2a9d8f", "gnn_3_rounds": "#2c5ea8"}
LABELS = {"flocking": "Regroupement", "manual": "Règle manuelle",
          "no_comm": "GNN sans communication", "gnn_3_rounds": "GNN 3 tours"}
SHAPE_LABELS = {"circle": "Cercle (vu à l'entraînement)", "star": "Étoile (jamais vue)"}

# Largeur de la zone de texte du mémoire (\textwidth = 455 pt = 6,3 po). Les figures sont dessinées
# exactement à cette largeur : LaTeX les imprime à 100 %, sans réduction, et le texte des figures
# garde la taille choisie ici (9,5 à 11 pt), lisible sur papier.
PRINT_WIDTH = 6.3

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.edgecolor": "#6b7785", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "legend.frameon": False, "savefig.dpi": 300,
    "figure.constrained_layout.use": True, "lines.linewidth": 1.6,
})


def episode(controller: str, n: int, shape: str, seed: int, failure: int = 0, training_seed: int | None = 11):
    """Trajectoire enregistrée par l'évaluateur (positions, masque actif, cible)."""
    base = PILOT / controller
    if controller in ("gnn_3_rounds", "no_comm"):
        base = base / f"seed_{training_seed}"
    path = base / "positions" / f"n{n}_failure{failure:02d}_geometries_{shape}_at_50" / f"seed_{seed}.npz"
    data = np.load(path)
    return data["positions"], data["active"], data["target"]


def chamfer_terms(points: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(points[:, None] - target[None], axis=2)
    return float(d.min(axis=1).mean()), float(d.min(axis=0).mean())


def distance_curve(positions, active, target) -> np.ndarray:
    return np.array([sum(chamfer_terms(p[a], target)) for p, a in zip(positions, active)])


def pilot_rows() -> list[dict]:
    with (PILOT / "episodes.csv").open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def arena(axis, limit: float = 2.05) -> None:
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect("equal")
    axis.set_xticks([-2, -1, 0, 1, 2])
    axis.set_yticks([-2, -1, 0, 1, 2])


def draw_state(axis, points, active, target, robot_color: str) -> None:
    axis.scatter(target[:, 0], target[:, 1], s=30, facecolors="none", edgecolors=TARGET, linewidths=1.1, zorder=2)
    axis.scatter(points[~active, 0], points[~active, 1], s=34, facecolors="white", edgecolors=INACTIVE, linewidths=1.1, zorder=3)
    axis.scatter(points[active, 0], points[active, 1], s=34, color=robot_color, edgecolors="white", linewidths=0.6, zorder=4)


def corner(axis, text: str) -> None:
    axis.text(0.03, 0.97, text, transform=axis.transAxes, va="top", ha="left", fontsize=10.5,
              bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID})


def plain_log(axis) -> None:
    """Échelle log lisible : graduations en nombres décimaux (0,1 ; 0,2 ; 1 ; 2…), pas en puissances."""
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    axis.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name: str) -> None:
    # Pas de recadrage : l'image garde exactement PRINT_WIDTH pouces de large.
    fig.savefig(OUTPUT / f"{name}.png")
    plt.close(fig)
    print("écrit", OUTPUT / f"{name}.png")


def rolling(values: np.ndarray, window: int = 25) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def grouped(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        shape = row["shape"].split("_")[1].split("@")[0]
        groups[(row["controller"], int(row["n_robots"]), float(row["failure_fraction"]), shape)].append(float(row["final_distance"]))
    return groups


# --- Figures --------------------------------------------------------------------------------

def formation_sequence() -> None:
    """Un épisode réel du GNN à trois tours (40 robots, cercle, graine 555000), plus sa courbe de distance."""
    positions, active, target = episode("gnn_3_rounds", 40, "circle", 555000)
    curve = distance_curve(positions, active, target)
    fig, axes = plt.subplots(2, 2, figsize=(PRINT_WIDTH, 6.5))
    steps = (0, 10, len(positions) - 1)
    for axis, step, letter in zip(axes.flat, steps, "abc"):
        draw_state(axis, positions[step], active[step], target, COLORS["gnn_3_rounds"])
        arena(axis)
        corner(axis, f"({letter}) pas {step}")
    curve_axis = axes[1, 1]
    curve_axis.plot(curve, color=COLORS["gnn_3_rounds"])
    for step in steps:
        curve_axis.plot(step, curve[step], "o", color=COLORS["manual"], markersize=5)
    curve_axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    plain_log(curve_axis)
    curve_axis.set_xlabel("pas")
    curve_axis.set_ylabel("distance")
    corner(curve_axis, "(d) distance au fil des pas")
    save(fig, "real_formation_sequence")


def communication_graph() -> None:
    """Graphe de communication recalculé (portée 1,5) sur le même épisode réel."""
    positions, active, target = episode("gnn_3_rounds", 40, "circle", 555000)
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_WIDTH, 3.6), sharey=True)
    for axis, step, letter in zip(axes, (0, len(positions) - 1), "ab"):
        p = positions[step]
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        adjacency = (d <= R_C) & (d > 0)
        degree = adjacency.sum(axis=1)
        focus = int(np.argsort(degree)[len(degree) // 2])
        for i, j in zip(*np.nonzero(np.triu(adjacency, 1))):
            axis.plot(*p[[i, j]].T, color=GRID, linewidth=0.5, zorder=1)
        for j in np.flatnonzero(adjacency[focus]):
            axis.plot(*p[[focus, j]].T, color=COLORS["gnn_3_rounds"], linewidth=1.0, zorder=2)
        axis.add_patch(plt.Circle(p[focus], R_C, fill=False, linestyle="--", color=COLORS["gnn_3_rounds"], linewidth=1.0))
        axis.scatter(p[:, 0], p[:, 1], s=20, color="#4f6070", zorder=3)
        axis.scatter(*p[adjacency[focus]].T, s=26, color=COLORS["gnn_3_rounds"], zorder=4)
        axis.scatter(*p[focus], s=110, marker="*", color=COLORS["manual"], zorder=5)
        arena(axis, 2.6)
        corner(axis, f"({letter}) pas {step} : {degree[focus]} voisins")
    save(fig, "real_communication_graph")


def reference_points() -> None:
    """Rendu binaire et 50 points de référence, produits par le code du catalogue."""
    config = ConfigLoader(REPOSITORY / "config").load_shapes()
    names = ["geometries_circle", "geometries_star", "digits_8", "letters_A", "letters_K", "numbers_314"]
    labels = ["Cercle", "Étoile", "Chiffre 8", "Lettre A", "Lettre K", "Nombre 314"]
    sources = {source.identifier: source for source in build_sources(config)}
    with tempfile.TemporaryDirectory() as directory:
        index, _ = generate_index(config, Path(directory))
        # Quatre rangées de trois : rendu, points, rendu, points.
        fig, axes = plt.subplots(4, 3, figsize=(PRINT_WIDTH, 8.0))
        for k, (name, label) in enumerate(zip(names, labels)):
            row, column = 2 * (k // 3), k % 3
            source = sources[name]
            mask = source.binarize(source.render(config.render_size))
            axes[row, column].imshow(~mask, cmap="gray", interpolation="nearest")
            axes[row, column].set_title(label)
            points = index.point_cloud_by_key(name)
            axes[row + 1, column].scatter(points[:, 0], points[:, 1], s=14, color=TARGET)
            axes[row + 1, column].set_xlim(-1.15, 1.15)
            axes[row + 1, column].set_ylim(-1.15, 1.15)
            axes[row + 1, column].set_aspect("equal")
        for axis in axes.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
        for row in (0, 2):
            axes[row, 0].set_ylabel("rendu 128 × 128")
            axes[row + 1, 0].set_ylabel(f"{config.reference_points} points")
    save(fig, "real_reference_points")


def proximity_coverage() -> None:
    """Les deux termes de la distance sur un épisode réel de la règle manuelle (étoile, 40 robots)."""
    positions, active, target = episode("manual", 40, "star", 555000, training_seed=None)
    terms = np.array([chamfer_terms(p[a], target) for p, a in zip(positions, active)])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(PRINT_WIDTH, 8.0), gridspec_kw={"height_ratios": [1.25, 1]})
    final = positions[-1][active[-1]]
    d = np.linalg.norm(target[:, None] - final[None], axis=2)
    for k in np.argsort(d.min(axis=1))[-6:]:
        j = int(d[k].argmin())
        top.annotate("", xy=final[j], xytext=target[k], arrowprops={"arrowstyle": "->", "color": COLORS["manual"], "lw": 1.2})
    draw_state(top, positions[-1], active[-1], target, COLORS["manual"])
    arena(top, 1.25)
    top.set_xticks([-1, 0, 1])
    top.set_yticks([-1, 0, 1])
    corner(top, "(a) pas 120 : points les moins couverts")
    steps = np.arange(len(terms))
    bottom.plot(steps, terms[:, 0], color=COLORS["gnn_3_rounds"], label="proximité (robot → cible)")
    bottom.plot(steps, terms[:, 1], color=COLORS["manual"], label="couverture (cible → robot)")
    bottom.plot(steps, terms.sum(axis=1), color=INK, linewidth=2, label="distance = somme des deux")
    bottom.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    bottom.text(len(steps) - 1, THRESHOLD * 1.08, "seuil 0,15", ha="right", va="bottom")
    plain_log(bottom)
    bottom.set_xlabel("pas de simulation")
    bottom.set_ylabel("distance normalisée")
    bottom.legend(loc="center right")
    corner(bottom, "(b)")
    save(fig, "real_proximity_coverage")


def success_criterion() -> None:
    """Courbes de distance de cinq épisodes enregistrés et début du suffixe sous le seuil."""
    fig, axes = plt.subplots(2, 1, figsize=(PRINT_WIDTH, 6.8), sharex=True, sharey=True)
    for axis, shape, letter in zip(axes, ("circle", "star"), "ab"):
        for seed in EPISODE_SEEDS:
            positions, active, target = episode("gnn_3_rounds", 40, shape, seed)
            curve = distance_curve(positions, active, target)
            axis.plot(curve, linewidth=1.3, color=COLORS["gnn_3_rounds"], alpha=0.85)
            k = convergence_step(curve, THRESHOLD)
            if k is not None:
                axis.plot(k, curve[k], "o", color=COLORS["manual"], markersize=6, zorder=5)
        axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
        plain_log(axis)
        axis.set_ylabel("distance normalisée")
        corner(axis, f"({letter}) {SHAPE_LABELS[shape]}")
    axes[1].set_xlabel("pas de simulation")
    axes[0].plot([], [], "o", color=COLORS["manual"], label="début du maintien sous le seuil")
    axes[0].plot([], [], "--", color=INK, label="seuil 0,15")
    axes[0].legend(loc="upper right")
    save(fig, "real_success_criterion")


def training_curves() -> None:
    """Journaux réels : termes de perte (GNN 3 tours, graine 11) et distance de validation."""
    losses = read_csv(TRAINING / "gnn_3_rounds" / "seed_11" / "losses.csv")
    iterations = np.array([int(row["iteration"]) for row in losses])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(PRINT_WIDTH, 8.0), sharex=True)
    styles = {"shape": ("proximité", COLORS["gnn_3_rounds"]), "coverage": ("couverture", COLORS["manual"]),
              "collisions": ("collisions", COLORS["no_comm"]), "motion": ("mouvement", COLORS["flocking"])}
    for key, (label, color) in styles.items():
        values = np.array([float(row[key]) for row in losses])
        top.plot(iterations[24:], rolling(values), color=color, label=label)
    plain_log(top)
    top.set_ylabel("terme de perte (moyenne\nglissante sur 25 itérations)")
    top.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.03))
    corner(top, "(a) GNN 3 tours, graine 11")
    for seed in range(11, 16):
        rows = read_csv(TRAINING / "no_comm" / f"seed_{seed}" / "validation.csv")
        bottom.plot([int(r["iteration"]) for r in rows], [float(r["selection_score"]) for r in rows],
                    color=COLORS["no_comm"], linewidth=1.1, alpha=0.8, label="GNN sans communication (5 graines)" if seed == 11 else None)
    rows = read_csv(TRAINING / "gnn_3_rounds" / "seed_11" / "validation.csv")
    bottom.plot([int(r["iteration"]) for r in rows], [float(r["selection_score"]) for r in rows],
                color=COLORS["gnn_3_rounds"], linewidth=2.2, label="GNN 3 tours (graine 11)")
    bottom.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    plain_log(bottom)
    bottom.set_xlabel("itération")
    bottom.set_ylabel("distance de validation")
    bottom.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    corner(bottom, "(b) validation toutes les 50 itérations")
    save(fig, "real_training_curves")


def by_controller(groups: dict, x_values, key, xlabel: str, name: str, offsets=None, xticks=None) -> None:
    """Deux panneaux empilés (cercle, étoile) : distance finale moyenne ± écart-type par contrôleur."""
    fig, axes = plt.subplots(2, 1, figsize=(PRINT_WIDTH, 8.0), sharex=True, sharey=True)
    for axis, shape, letter in zip(axes, ("circle", "star"), "ab"):
        for index, controller in enumerate(COLORS):
            means = [np.mean(groups[key(controller, x, shape)]) for x in x_values]
            spreads = [np.std(groups[key(controller, x, shape)], ddof=1) for x in x_values]
            positions = np.arange(len(x_values)) + (offsets[index] if offsets else 0.0)
            axis.errorbar(positions, means, yerr=spreads, fmt="o" if offsets else "o-", color=COLORS[controller],
                          capsize=3.5, markersize=6, linewidth=1.6, label=LABELS[controller])
        axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
        plain_log(axis)
        axis.set_ylim(0.03, 0.8)
        axis.set_ylabel("distance finale")
        corner(axis, f"({letter}) {SHAPE_LABELS[shape]}")
    axes[1].set_xticks(range(len(x_values)), xticks)
    axes[1].set_xlabel(xlabel)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    save(fig, name)


def final_states(shape: str) -> None:
    """État final au pas 120 d'un même épisode (40 robots, graine 555000), quatre contrôleurs."""
    fig, axes = plt.subplots(2, 2, figsize=(PRINT_WIDTH, 6.9), sharex=True, sharey=True)
    for axis, controller in zip(axes.flat, ["flocking", "manual", "no_comm", "gnn_3_rounds"]):
        seed = None if controller in ("flocking", "manual") else 11
        positions, active, target = episode(controller, 40, shape, 555000, training_seed=seed)
        curve = distance_curve(positions, active, target)
        draw_state(axis, positions[-1], active[-1], target, COLORS[controller])
        arena(axis, 1.3)
        axis.set_xticks([-1, 0, 1])
        axis.set_yticks([-1, 0, 1])
        axis.set_title(LABELS[controller])
        corner(axis, f"distance {curve[-1]:.3f}".replace(".", ","))
    save(fig, f"real_final_states_{shape}")


def seed_dispersion(rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(PRINT_WIDTH, 5.2))
    sizes = [20, 40, 100]
    per_seed = defaultdict(list)
    for row in rows:
        if row["controller"] == "no_comm" and float(row["failure_fraction"]) == 0.0 and "circle" in row["shape"]:
            per_seed[(int(row["training_seed"]), int(row["n_robots"]))].append(float(row["final_distance"]))
    for seed, marker in zip(range(11, 16), "osD^v"):
        axis.plot(sizes, [np.mean(per_seed[(seed, n)]) for n in sizes], marker=marker, markersize=6, color=COLORS["no_comm"],
                  linewidth=1.0, alpha=0.85, label=f"sans communication, graine {seed}")
    reference = grouped(rows)
    for controller in ("gnn_3_rounds", "manual"):
        axis.plot(sizes, [np.mean(reference[(controller, n, 0.0, "circle")]) for n in sizes], marker="o", markersize=7,
                  color=COLORS[controller], linewidth=2.2, label=LABELS[controller] + (" (graine 11)" if controller == "gnn_3_rounds" else ""))
    axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_xticks(sizes, [str(n) for n in sizes])
    axis.minorticks_off()
    plain_log(axis)
    axis.set_xlabel("nombre de robots")
    axis.set_ylabel("distance finale moyenne, cercle")
    fig.legend(*axis.get_legend_handles_labels(), loc="outside lower center", ncol=2)
    save(fig, "real_seed_dispersion")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    rows = pilot_rows()
    groups = grouped(rows)
    formation_sequence()
    communication_graph()
    reference_points()
    proximity_coverage()
    success_criterion()
    training_curves()
    by_controller(groups, [20, 40, 100], lambda c, n, s: (c, n, 0.0, s), "nombre de robots", "real_pilot_by_size",
                  offsets=(-0.2, -0.07, 0.07, 0.2), xticks=["20", "40", "100"])
    by_controller(groups, [0.0, 0.05, 0.1, 0.2], lambda c, f, s: (c, 40, f, s), "robots retirés au pas 60 (40 robots)",
                  "real_pilot_failures", xticks=["0 %", "5 %", "10 %", "20 %"])
    final_states("circle")
    final_states("star")
    seed_dispersion(rows)


if __name__ == "__main__":
    main()
