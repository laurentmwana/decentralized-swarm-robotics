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

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
    "axes.edgecolor": "#6b7785", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.7, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.labelcolor": INK, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
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
    axis.tick_params(labelsize=9)


def draw_state(axis, points, active, target, robot_color: str) -> None:
    axis.scatter(target[:, 0], target[:, 1], s=22, facecolors="none", edgecolors=TARGET, linewidths=1.0, zorder=2)
    axis.scatter(points[~active, 0], points[~active, 1], s=26, facecolors="white", edgecolors=INACTIVE, linewidths=1.0, zorder=3)
    axis.scatter(points[active, 0], points[active, 1], s=26, color=robot_color, edgecolors="white", linewidths=0.6, zorder=4)


def corner(axis, text: str) -> None:
    axis.text(0.03, 0.97, text, transform=axis.transAxes, va="top", ha="left", fontsize=10,
              bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": GRID})


def plain_log(axis) -> None:
    """Échelle log lisible : graduations en nombres décimaux (0,1 ; 0,2 ; 1 ; 2…), pas en puissances."""
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    axis.yaxis.set_minor_formatter(NullFormatter())


def save(fig, name: str) -> None:
    fig.savefig(OUTPUT / f"{name}.png")
    plt.close(fig)
    print("écrit", OUTPUT / f"{name}.png")


# --- Figures --------------------------------------------------------------------------------

def formation_sequence() -> None:
    """Un épisode réel du GNN à trois tours : 40 robots, cercle, graine d'épisode 555000."""
    positions, active, target = episode("gnn_3_rounds", 40, "circle", 555000)
    curve = distance_curve(positions, active, target)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)
    for axis, step, letter in zip(axes, (0, 10, len(positions) - 1), "abc"):
        draw_state(axis, positions[step], active[step], target, COLORS["gnn_3_rounds"])
        arena(axis)
        corner(axis, f"({letter}) pas {step} · distance {curve[step]:.2f}".replace(".", ","))
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    save(fig, "real_formation_sequence")


def communication_graph() -> None:
    """Graphe de communication recalculé (portée 1,5) sur le même épisode réel."""
    positions, active, target = episode("gnn_3_rounds", 40, "circle", 555000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    for axis, step, letter in zip(axes, (0, len(positions) - 1), "ab"):
        p = positions[step]
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        adjacency = (d <= R_C) & (d > 0)
        degree = adjacency.sum(axis=1)
        focus = int(np.argsort(degree)[len(degree) // 2])
        for i, j in zip(*np.nonzero(np.triu(adjacency, 1))):
            axis.plot(*p[[i, j]].T, color=GRID, linewidth=0.6, zorder=1)
        for j in np.flatnonzero(adjacency[focus]):
            axis.plot(*p[[focus, j]].T, color=COLORS["gnn_3_rounds"], linewidth=1.1, zorder=2)
        axis.add_patch(plt.Circle(p[focus], R_C, fill=False, linestyle="--", color=COLORS["gnn_3_rounds"], linewidth=0.9))
        axis.scatter(p[:, 0], p[:, 1], s=22, color="#4f6070", zorder=3)
        axis.scatter(*p[adjacency[focus]].T, s=30, color=COLORS["gnn_3_rounds"], zorder=4)
        axis.scatter(*p[focus], s=80, marker="*", color=COLORS["manual"], zorder=5)
        arena(axis, 2.6)
        corner(axis, f"({letter}) pas {step} · robot marqué : {degree[focus]} voisins sur 39")
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    save(fig, "real_communication_graph")


def reference_points() -> None:
    """Rendu binaire et 50 points de référence, produits par le code du catalogue."""
    config = ConfigLoader(REPOSITORY / "config").load_shapes()
    names = ["geometries_circle", "geometries_star", "digits_8", "letters_A", "letters_K", "numbers_314"]
    labels = ["Cercle", "Étoile", "Chiffre 8", "Lettre A", "Lettre K", "Nombre 314"]
    sources = {source.identifier: source for source in build_sources(config)}
    with tempfile.TemporaryDirectory() as directory:
        index, _ = generate_index(config, Path(directory))
        fig, axes = plt.subplots(2, len(names), figsize=(12, 4.6))
        for column, (name, label) in enumerate(zip(names, labels)):
            source = sources[name]
            mask = source.binarize(source.render(config.render_size))
            axes[0, column].imshow(~mask, cmap="gray", interpolation="nearest")
            axes[0, column].set_title(label, fontsize=10)
            points = index.point_cloud_by_key(name)
            axes[1, column].scatter(points[:, 0], points[:, 1], s=9, color=TARGET)
            axes[1, column].set_xlim(-1.15, 1.15)
            axes[1, column].set_ylim(-1.15, 1.15)
            axes[1, column].set_aspect("equal")
            for axis in axes[:, column]:
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
        axes[0, 0].set_ylabel("rendu 128 × 128")
        axes[1, 0].set_ylabel(f"{config.reference_points} points")
    save(fig, "real_reference_points")


def proximity_coverage() -> None:
    """Les deux termes de la distance sur un épisode réel de la règle manuelle (étoile, 40 robots)."""
    positions, active, target = episode("manual", 40, "star", 555000, training_seed=None)
    terms = np.array([chamfer_terms(p[a], target) for p, a in zip(positions, active)])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})
    final = positions[-1][active[-1]]
    d = np.linalg.norm(target[:, None] - final[None], axis=2)
    far = np.argsort(d.min(axis=1))[-6:]
    for k in far:
        j = int(d[k].argmin())
        left.annotate("", xy=final[j], xytext=target[k], arrowprops={"arrowstyle": "->", "color": COLORS["manual"], "lw": 1.0})
    draw_state(left, positions[-1], active[-1], target, COLORS["manual"])
    arena(left, 1.4)
    corner(left, "(a) pas 120 : points les moins couverts")
    steps = np.arange(len(terms))
    right.plot(steps, terms[:, 0], color=COLORS["gnn_3_rounds"], label="proximité (robot → cible)")
    right.plot(steps, terms[:, 1], color=COLORS["manual"], label="couverture (cible → robot)")
    right.plot(steps, terms.sum(axis=1), color=INK, linewidth=1.6, label="distance = somme des deux")
    right.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    right.text(len(steps) - 1, THRESHOLD + 0.01, "seuil 0,15", ha="right", va="bottom", fontsize=9)
    plain_log(right)
    right.set_xlabel("pas de simulation")
    right.set_ylabel("distance normalisée (échelle log)")
    right.legend(fontsize=9, loc="upper right")
    corner(right, "(b)")
    save(fig, "real_proximity_coverage")


def success_criterion() -> None:
    """Courbes de distance de cinq épisodes enregistrés et début du suffixe sous le seuil."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    for axis, shape, letter in zip(axes, ("circle", "star"), "ab"):
        for seed in EPISODE_SEEDS:
            positions, active, target = episode("gnn_3_rounds", 40, shape, seed)
            curve = distance_curve(positions, active, target)
            line, = axis.plot(curve, linewidth=1.2, color=COLORS["gnn_3_rounds"], alpha=0.85)
            k = convergence_step(curve, THRESHOLD)
            if k is not None:
                axis.plot(k, curve[k], "o", color=COLORS["manual"], markersize=5, zorder=5)
        axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
        plain_log(axis)
        axis.set_xlabel("pas de simulation")
        corner(axis, f"({letter}) {SHAPE_LABELS[shape]}")
    axes[0].set_ylabel("distance normalisée (échelle log)")
    axes[1].plot([], [], "o", color=COLORS["manual"], label="début du maintien sous le seuil")
    axes[1].plot([], [], "--", color=INK, label="seuil 0,15")
    axes[1].legend(fontsize=9, loc="upper right")
    save(fig, "real_success_criterion")


def rolling(values: np.ndarray, window: int = 25) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def training_curves() -> None:
    """Journaux réels : termes de perte (GNN 3 tours, graine 11) et distance de validation."""
    losses = read_csv(TRAINING / "gnn_3_rounds" / "seed_11" / "losses.csv")
    iterations = np.array([int(row["iteration"]) for row in losses])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    styles = {"shape": ("proximité", COLORS["gnn_3_rounds"]), "coverage": ("couverture", COLORS["manual"]),
              "collisions": ("collisions", COLORS["no_comm"]), "motion": ("mouvement", COLORS["flocking"])}
    for key, (label, color) in styles.items():
        values = np.array([float(row[key]) for row in losses])
        left.plot(iterations[24:], rolling(values), color=color, label=label)
    plain_log(left)
    left.set_xlabel("itération")
    left.set_ylabel("terme de perte (moyenne glissante, 25 it.)")
    left.legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    corner(left, "(a) GNN 3 tours, graine 11")
    for seed in range(11, 16):
        rows = read_csv(TRAINING / "no_comm" / f"seed_{seed}" / "validation.csv")
        right.plot([int(r["iteration"]) for r in rows], [float(r["selection_score"]) for r in rows],
                   color=COLORS["no_comm"], linewidth=1.0, alpha=0.75, label="GNN sans communication (5 graines)" if seed == 11 else None)
    rows = read_csv(TRAINING / "gnn_3_rounds" / "seed_11" / "validation.csv")
    right.plot([int(r["iteration"]) for r in rows], [float(r["selection_score"]) for r in rows],
               color=COLORS["gnn_3_rounds"], linewidth=1.8, label="GNN 3 tours (graine 11)")
    right.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    plain_log(right)
    right.set_xlabel("itération")
    right.set_ylabel("distance de validation")
    right.legend(fontsize=9, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    corner(right, "(b) validation toutes les 50 itérations")
    save(fig, "real_training_curves")


def grouped(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        shape = row["shape"].split("_")[1].split("@")[0]
        groups[(row["controller"], int(row["n_robots"]), float(row["failure_fraction"]), shape)].append(float(row["final_distance"]))
    return groups


def pilot_by_size(groups: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    sizes = [20, 40, 100]
    for axis, shape, letter in zip(axes, ("circle", "star"), "ab"):
        for offset, controller in zip((-0.18, -0.06, 0.06, 0.18), COLORS):
            means = [np.mean(groups[(controller, n, 0.0, shape)]) for n in sizes]
            spreads = [np.std(groups[(controller, n, 0.0, shape)], ddof=1) for n in sizes]
            axis.errorbar(np.arange(3) + offset, means, yerr=spreads, fmt="o", color=COLORS[controller],
                          capsize=3, markersize=5, label=LABELS[controller])
        axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
        axis.set_xticks(range(3), [str(n) for n in sizes])
        axis.set_xlabel("nombre de robots")
        plain_log(axis)
        axis.set_ylim(0.03, 0.8)
        corner(axis, f"({letter}) {SHAPE_LABELS[shape]}")
    axes[0].set_ylabel("distance finale (moyenne ± écart-type)")
    axes[1].legend(fontsize=9, loc="lower left")
    save(fig, "real_pilot_by_size")


def pilot_failures(groups: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    fractions = [0.0, 0.05, 0.1, 0.2]
    for axis, shape, letter in zip(axes, ("circle", "star"), "ab"):
        for controller in COLORS:
            means = [np.mean(groups[(controller, 40, f, shape)]) for f in fractions]
            spreads = [np.std(groups[(controller, 40, f, shape)], ddof=1) for f in fractions]
            axis.errorbar([100 * f for f in fractions], means, yerr=spreads, marker="o", color=COLORS[controller],
                          capsize=3, markersize=4, label=LABELS[controller])
        axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
        axis.set_xticks([0, 5, 10, 20])
        axis.set_xlabel("robots retirés au pas 60 (%)")
        plain_log(axis)
        corner(axis, f"({letter}) {SHAPE_LABELS[shape]}, 40 robots")
    axes[0].set_ylabel("distance finale (moyenne ± écart-type)")
    axes[1].legend(fontsize=9, loc="lower right")
    save(fig, "real_pilot_failures")


def final_states() -> None:
    controllers = ["flocking", "manual", "no_comm", "gnn_3_rounds"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.3), sharex=True, sharey=True)
    for row, shape in enumerate(("circle", "star")):
        for column, controller in enumerate(controllers):
            seed = None if controller in ("flocking", "manual") else 11
            positions, active, target = episode(controller, 40, shape, 555000, training_seed=seed)
            curve = distance_curve(positions, active, target)
            draw_state(axes[row, column], positions[-1], active[-1], target, COLORS[controller])
            arena(axes[row, column])
            corner(axes[row, column], f"distance {curve[-1]:.3f}".replace(".", ","))
            if row == 0:
                axes[row, column].set_title(LABELS[controller], fontsize=10)
        axes[row, 0].set_ylabel(SHAPE_LABELS[shape].split(" (")[0])
    save(fig, "real_final_states")


def seed_dispersion(rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    sizes = [20, 40, 100]
    per_seed = defaultdict(list)
    for row in rows:
        if row["controller"] == "no_comm" and float(row["failure_fraction"]) == 0.0 and "circle" in row["shape"]:
            per_seed[(int(row["training_seed"]), int(row["n_robots"]))].append(float(row["final_distance"]))
    for seed, marker in zip(range(11, 16), "osD^v"):
        axis.plot(sizes, [np.mean(per_seed[(seed, n)]) for n in sizes], marker=marker, color=COLORS["no_comm"],
                  linewidth=0.9, alpha=0.85, label=f"sans communication, graine {seed}")
    reference = grouped(rows)
    for controller in ("gnn_3_rounds", "manual"):
        axis.plot(sizes, [np.mean(reference[(controller, n, 0.0, "circle")]) for n in sizes], marker="o",
                  color=COLORS[controller], linewidth=1.8, label=LABELS[controller] + (" (graine 11)" if controller == "gnn_3_rounds" else ""))
    axis.axhline(THRESHOLD, color=INK, linestyle="--", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_xticks(sizes, [str(n) for n in sizes])
    axis.minorticks_off()
    plain_log(axis)
    axis.set_xlabel("nombre de robots")
    axis.set_ylabel("distance finale moyenne, cercle")
    axis.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
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
    pilot_by_size(groups)
    pilot_failures(groups)
    final_states()
    seed_dispersion(rows)


if __name__ == "__main__":
    main()
