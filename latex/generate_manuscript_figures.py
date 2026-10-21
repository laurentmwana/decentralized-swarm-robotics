"""Génère seize figures professionnelles déterministes pour le mémoire, sans entraînement."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from generate_scientific_figures import (
    BLUE, GRAY, GRID, INK, ORANGE, PALE_BLUE, PALE_ORANGE, PALE_TEAL, PURPLE, TEAL, WIDTH,
    arrow, box, canvas, header, routed_arrow, style_plot,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = ROOT / "images"
OUTPUTS = ROOT.parent / "outputs"
REFERENCE_RUN = OUTPUTS / "references" / "flocking_geometries_cercle_at_50_20"
COMMUNICATION_RADIUS = 1.5
EPISODE_SEED = 142

FIGURE_NAMES = (
    "experimental_chain",
    "consensus_dynamics",
    "flocking_rules",
    "message_aggregation",
    "information_reach",
    "swarm_communication_graph",
    "target_preparation_pipeline",
    "software_modules",
    "formation_principle",
    "historical_pilot_summary",
    "historical_chamfer_vs_failures",
    "failure_timeline",
    "historical_scaling",
    "threshold_crossing_schema",
    "historical_seed_dispersion",
    "historical_curriculum_per_target",
)
SOURCE_REFERENCES = {
    "experimental_chain": ("src/training/trainer.py", "src/controllers/gnn.py", "config/"),
    "consensus_dynamics": ("Figure specification: exact averaged consensus iteration, path graph of five agents"),
    "flocking_rules": ("outputs/references/flocking_geometries_cercle_at_50_20", "src/controllers/flocking.py"),
    "message_aggregation": ("src/models/message.py", "src/models/aggregator.py"),
    "information_reach": ("src/models/gnn.py", "config/model.yaml"),
    "swarm_communication_graph": ("outputs/references/flocking_geometries_cercle_at_50_20", "src/simulator/graph.py"),
    "target_preparation_pipeline": ("outputs/shapes/images/letters_A.png", "outputs/shapes/point_clouds", "src/shapes/sampling.py"),
    "software_modules": ("src/", "config/", "outputs/"),
    "formation_principle": ("Figure specification: conceptual deterministic scenario, no controller run"),
    # Valeurs consignées dans latex/chapters/experiences.tex (prototype historique).
    "historical_pilot_summary": ("latex/chapters/experiences.tex: tab:pilote-echelle, tab:pannes-20"),
    "historical_chamfer_vs_failures": ("latex/chapters/experiences.tex: tab:pannes-20"),
    "failure_timeline": ("latex/chapters/experiences.tex: protocole de retrait à mi-épisode"),
    "historical_scaling": ("latex/chapters/experiences.tex: tab:pilote-echelle"),
    "threshold_crossing_schema": ("Figure specification: constructed curves for the historical threshold convention"),
    "historical_seed_dispersion": ("latex/chapters/experiences.tex: tab:dispersion"),
    "historical_curriculum_per_target": ("latex/chapters/experiences.tex: tab:curriculum-pertes"),
}

# Valeurs historiques recopiées des tableaux du chapitre 4, aucune nouvelle mesure.
SCALE_ROBOTS = (20, 40, 100)
SCALE_GNN = (0.592, 0.522, 0.428)
SCALE_FLOCKING = (3.956, 3.799, 3.507)
SCALE_SUCCESS = (0.10, 0.30, 1.00)
FAILURE_FRACTIONS = (0.0, 0.05, 0.10, 0.20)
FAILURE_GNN = (0.592, 0.612, 0.626, 0.638)
FAILURE_GNN_STD = (0.086, 0.085, 0.095, 0.079)
FAILURE_FLOCKING = (3.956, 3.966, 3.971, 4.162)
FAILURE_FLOCKING_STD = (1.027, 1.043, 1.014, 1.321)
SEED_LABELS = ("103", "101", "107", "109", "108", "110", "105", "106", "102", "104")
SEED_CHAMFER = (0.499, 0.503, 0.521, 0.528, 0.561, 0.592, 0.608, 0.643, 0.724, 0.735)
SEED_SUCCESS = (False, False, True, False, False, False, False, False, False, False)
CURRICULUM_STAGES = ("Géométrie", "Chiffres", "Symboles", "Nombres")
CURRICULUM_TARGETS = (3, 6, 9, 11)
CURRICULUM_OLD = (None, 0.333, 0.216, 0.163)
CURRICULUM_NEW = (0.316, 0.126, 0.141, 0.117)


def fr(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def archived_run() -> np.ndarray:
    return np.load(REFERENCE_RUN / "positions.npy")


def raster_mask() -> np.ndarray:
    image = plt.imread(OUTPUTS / "shapes" / "images" / "letters_A.png")
    luminance = image[..., :3].mean(axis=-1) if image.ndim == 3 else image
    return luminance < 0.5


def shape_cloud(name: str) -> np.ndarray:
    return np.load(OUTPUTS / "shapes" / "point_clouds" / name)


def adjacency(positions: np.ndarray, radius: float) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    return (distances > 0.0) & (distances <= radius)


def consensus_passes(values: np.ndarray, passes: int) -> np.ndarray:
    count = values.size
    history = [values]
    for _ in range(passes):
        current = history[-1]
        updated = np.empty_like(current)
        for index in range(count):
            neighborhood = [current[index]]
            if index > 0:
                neighborhood.append(current[index - 1])
            if index < count - 1:
                neighborhood.append(current[index + 1])
            updated[index] = np.mean(neighborhood)
        history.append(updated)
    return np.array(history)


def verify_checks() -> None:
    history = consensus_passes(np.array([0.0, 2.0, 4.0, 1.0, 3.0]), 400)
    spread = history.max(axis=1) - history.min(axis=1)
    assert np.all(np.diff(spread) <= 1e-12) and spread[-1] < 1e-6
    print("PASS: averaged consensus on a five-agent path graph; spread decreases monotonically to zero.")
    positions = archived_run()
    mask = adjacency(positions[0], COMMUNICATION_RADIUS)
    assert np.array_equal(mask, mask.T) and not mask.diagonal().any()
    print(f"PASS: archived initial positions give {int(mask.sum() // 2)} unordered edges at r_c = 1.5; symmetric, no self-loop.")
    for name, expected in (("letters_A@50.npy", 50), ("letters_A@20.npy", 20)):
        cloud = shape_cloud(name)
        assert cloud.shape == (expected, 2)
        assert len(np.unique(cloud, axis=0)) == expected
        assert np.linalg.norm(cloud, axis=1).max() <= 1.0 + 1e-9
    mask_pixels = raster_mask()
    assert 500 < int(mask_pixels.sum()) < 20000
    print("PASS: archived shape artifacts: distinct points, radius 1, nonempty 128x128 mask.")
    assert len(SEED_CHAMFER) == 10 and abs(np.mean(SEED_CHAMFER) - 0.592) < 0.001
    assert len(FAILURE_GNN) == len(FAILURE_GNN_STD) == len(FAILURE_FRACTIONS) == 4
    assert CURRICULUM_OLD[0] is None and len(CURRICULUM_NEW) == 4
    print("PASS: historical values copied from the chapter 4 tables are self-consistent.")


def preflight_assets() -> None:
    if not OUTPUT_DIRECTORY.is_dir() or OUTPUT_DIRECTORY.is_symlink():
        raise ValueError("latex/images must exist and be a regular directory")
    for name in FIGURE_NAMES:
        for extension in ("pdf", "png"):
            path = OUTPUT_DIRECTORY / f"{name}.{extension}"
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"Refusing nonregular output: {path}")
            status = f"existing, {path.stat().st_size} bytes" if path.exists() else "absent"
            print(f"ASSET before write: {path} ({status})")


def experimental_chain() -> plt.Figure:
    fig, ax = canvas(6.4)
    header(fig, "Chaîne expérimentale", "Apprentissage centralisé puis exécution décentralisée de la politique partagée")
    box(ax, 0.04, 0.76, 0.92, 0.10,
        "Configurations YAML\nsimulateur · formes · modèle · apprentissage · mesures", color=PALE_TEAL, fontsize=16)
    ax.add_patch(Rectangle((0.04, 0.30), 0.46, 0.42, facecolor="none", edgecolor=BLUE, linewidth=1.4, linestyle="--"))
    ax.text(0.27, 0.68, "Phase centralisée : apprentissage", ha="center", va="center", fontsize=16, weight="bold", color=BLUE)
    box(ax, 0.06, 0.52, 0.20, 0.12, "Simulateur\népisodes de T pas", fontsize=15.5)
    box(ax, 0.28, 0.52, 0.20, 0.12, "Formes cibles\nnuages de points", fontsize=15.5)
    box(ax, 0.06, 0.34, 0.42, 0.12,
        "Apprentissage centralisé\nperte · rétropropagation · Adam", fontsize=15.5)
    arrow(ax, (0.16, 0.51), (0.16, 0.472), color=BLUE)
    arrow(ax, (0.38, 0.51), (0.38, 0.472), color=BLUE)
    ax.add_patch(Rectangle((0.54, 0.30), 0.42, 0.42, facecolor="none", edgecolor=TEAL, linewidth=1.4, linestyle="--"))
    ax.text(0.75, 0.68, "Phase décentralisée : exécution", ha="center", va="center", fontsize=16, weight="bold", color=TEAL)
    box(ax, 0.56, 0.52, 0.38, 0.12, "Politique partagée\nmêmes poids pour chaque robot", color=PALE_TEAL, fontsize=15.5)
    box(ax, 0.56, 0.34, 0.38, 0.12,
        "Exécution décentralisée\nobservation + voisins + consigne", color=PALE_TEAL, fontsize=15.5)
    arrow(ax, (0.75, 0.51), (0.75, 0.472), color=TEAL)
    routed_arrow(ax, [(0.27, 0.33), (0.27, 0.24), (0.75, 0.24), (0.75, 0.33)], color=BLUE)
    ax.text(0.51, 0.255, "poids appris", ha="center", va="bottom", fontsize=15, color=BLUE)
    return fig


def consensus_dynamics() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.4))
    header(fig, "Consensus discret sur un graphe linéaire", "Mise à jour moyennée exacte · illustration théorique")
    left = fig.add_axes([0.085, 0.30, 0.375, 0.48])
    right = fig.add_axes([0.59, 0.30, 0.375, 0.48])
    history = consensus_passes(np.array([0.0, 2.0, 4.0, 1.0, 3.0]), 40)
    passes = np.arange(history.shape[0])
    colors = (BLUE, TEAL, ORANGE, PURPLE, GRAY)
    style_plot(left)
    left.set_title("(a) Valeurs par passe", pad=16)
    for index in range(history.shape[1]):
        left.plot(passes, history[:, index], color=colors[index], linewidth=2.2, label=f"agent {index + 1}")
    common = float(history[-1, 0])
    left.axhline(common, color=INK, linestyle=":", linewidth=1.4)
    left.annotate("valeur commune atteinte", xy=(20, common), xytext=(6, 3.45), fontsize=14.5, color=INK,
                  arrowprops={"arrowstyle": "->", "color": INK, "connectionstyle": "arc3,rad=-0.15"})
    left.set(xlim=(0, 40), ylim=(-0.2, 4.2), xlabel="Passe", ylabel="Valeur de l'agent")
    left.legend(loc="center right", frameon=False, fontsize=13.5)
    spread = history.max(axis=1) - history.min(axis=1)
    style_plot(right)
    right.set_title("(b) Écart entre valeurs extrêmes", pad=16)
    right.semilogy(passes[1:], np.maximum(spread[1:], 1e-12), color=BLUE, linewidth=2.4)
    right.set(xlim=(0, 40), xlabel="Passe", ylabel="Écart (échelle logarithmique)")
    return fig


def flocking_rules() -> plt.Figure:
    positions = archived_run()
    fig = plt.figure(figsize=(WIDTH, 7.8))
    header(fig, "Règle de regroupement : trajectoires et contributions",
           "Exécution archivée du simulateur actuel · décomposition exacte au pas choisi")
    left = fig.add_axes([0.06, 0.32, 0.42, 0.46])
    style_plot(left)
    left.set_title("(a) Trajectoires archivées · graine 142", pad=16)
    for robot in range(positions.shape[1]):
        left.plot(positions[:, robot, 0], positions[:, robot, 1], color=BLUE, linewidth=1.2, alpha=0.55)
    left.scatter(positions[0, :, 0], positions[0, :, 1], s=16, color=GRAY, label="départ", zorder=4)
    left.scatter(positions[-1, :, 0], positions[-1, :, 1], s=42, color=ORANGE, edgecolor="white",
                 linewidth=0.7, label="arrivée", zorder=5)
    left.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal", xlabel="x", ylabel="y")
    left.legend(loc="upper left", frameon=False, fontsize=13.5)
    step = 10
    current = positions[step]
    velocity = positions[step + 1] - positions[step]
    mask = adjacency(current, COMMUNICATION_RADIUS)
    degrees = mask.sum(axis=1)
    focal = int(np.argmax(degrees))
    neighbors = np.flatnonzero(mask[focal])
    cohesion = current[neighbors].mean(axis=0) - current[focal]
    nearest = neighbors[int(np.argmin(np.linalg.norm(current[neighbors] - current[focal], axis=1)))]
    separation = current[focal] - current[nearest]
    alignment = velocity[neighbors].mean(axis=0) - velocity[focal]
    vectors = {"cohésion": (cohesion, TEAL), "séparation": (separation, ORANGE), "alignement": (alignment, PURPLE)}
    resultant = cohesion + separation + alignment
    right = fig.add_axes([0.56, 0.32, 0.40, 0.46])
    style_plot(right)
    right.set_title(f"(b) Contributions au pas t = {step}", pad=16)
    scale = 0.85 / max(np.linalg.norm(resultant), *(np.linalg.norm(v) for v, _ in vectors.values()), 1e-9)
    for label, (vector, color) in vectors.items():
        arrow(right, tuple(current[focal]), tuple(current[focal] + scale * vector), color=color,
              linewidth=2.6, mutation_scale=18)
    arrow(right, tuple(current[focal]), tuple(current[focal] + scale * resultant), color=INK,
          linewidth=2.6, linestyle="--", mutation_scale=18)
    right.scatter(current[:, 0], current[:, 1], s=34, color=PALE_BLUE, edgecolor=GRAY, linewidth=1.0, zorder=3)
    right.scatter(current[neighbors, 0], current[neighbors, 1], s=44, facecolor="white", edgecolor=TEAL,
                  linewidth=1.8, zorder=4)
    right.scatter([current[focal, 0]], [current[focal, 1]], s=70, color=ORANGE, edgecolor="white",
                  linewidth=0.8, zorder=5)
    right.set(xlim=(current[focal, 0] - 1.3, current[focal, 0] + 1.3),
              ylim=(current[focal, 1] - 1.3, current[focal, 1] + 1.3), aspect="equal", xlabel="x", ylabel="y")
    handles = [
        Line2D([], [], color=TEAL, linewidth=2.6, label="cohésion : centre des voisins"),
        Line2D([], [], color=ORANGE, linewidth=2.6, label="séparation : loin du plus proche"),
        Line2D([], [], color=PURPLE, linewidth=2.6, label="alignement : vitesse moyenne"),
        Line2D([], [], color=INK, linewidth=2.6, linestyle="--", label="résultante"),
    ]
    right.legend(handles=handles, loc="upper left", frameon=False, fontsize=13)
    return fig


def message_aggregation() -> plt.Figure:
    fig, ax = canvas(6.2)
    header(fig, "Agrégation locale des messages", "Moyenne masquée sur le voisinage · mise à jour de l'état latent")
    for x, label in ((0.12, "voisins j ∈ N(i)"), (0.38, "messages"), (0.635, "agrégation"), (0.835, "état local")):
        ax.text(x, 0.845, label, ha="center", fontsize=16, color=GRAY)
    heights = (0.70, 0.50, 0.30)
    for index, y in enumerate(heights, start=1):
        ax.add_patch(Circle((0.12, y), 0.038, facecolor=BLUE, edgecolor="white", linewidth=1.2))
        ax.text(0.12, y, f"j{index}", ha="center", va="center", color="white", fontsize=15)
        box(ax, 0.30, y - 0.048, 0.16, 0.096, f"m$_{{ij{index}}}$", color=PALE_ORANGE, fontsize=16)
        arrow(ax, (0.162, y), (0.292, y))
        arrow(ax, (0.468, y), (0.545, 0.50 + (y - 0.50) * 0.35))
    box(ax, 0.55, 0.38, 0.17, 0.24,
        "moyenne\n$\\bar{m}_i=\\frac{1}{|\\mathcal{N}(i)|}\\sum_j m_{ij}$\nagrégat : 64", color=PALE_TEAL, fontsize=15)
    arrow(ax, (0.728, 0.50), (0.752, 0.50))
    box(ax, 0.76, 0.38, 0.15, 0.24, "mise à jour\nψ$_\\ell$\nh : 64 → 64", fontsize=15)
    ax.add_patch(Circle((0.835, 0.80), 0.034, facecolor=ORANGE, edgecolor="white", linewidth=1.2))
    ax.text(0.835, 0.80, "i", ha="center", va="center", color="white", fontsize=15)
    arrow(ax, (0.835, 0.762), (0.835, 0.632), color=ORANGE)
    ax.text(0.855, 0.70, "h$_i^\\ell$", fontsize=16, color=ORANGE)
    arrow(ax, (0.918, 0.50), (0.955, 0.50), color=INK)
    ax.text(0.958, 0.552, "h$_i^{\\ell+1}$", ha="center", fontsize=15)
    return fig


def information_reach() -> plt.Figure:
    fig, ax = canvas(6.6)
    header(fig, "Portée de l'information selon les tours", "Un tour : voisinage direct · deux tours : anneau suivant")
    center = (0.50, 0.50)
    ax.add_patch(Circle(center, 0.34, facecolor=PALE_TEAL, edgecolor="none", alpha=0.55, zorder=0))
    ax.add_patch(Circle(center, 0.19, facecolor=PALE_BLUE, edgecolor="none", alpha=0.85, zorder=1))
    ring1 = [(0.50 + 0.14 * np.cos(a), 0.50 + 0.14 * np.sin(a))
             for a in np.deg2rad([45, 135, 225, 315])]
    ring2 = [(0.50 + 0.27 * np.cos(a), 0.50 + 0.27 * np.sin(a))
             for a in np.deg2rad([15, 75, 150, 200, 260, 330])]
    beyond = [(0.50 + 0.40 * np.cos(a), 0.50 + 0.40 * np.sin(a))
              for a in np.deg2rad([25, 205, 335])]
    for node in ring1:
        arrow(ax, center, node, color=BLUE, linewidth=1.4, mutation_scale=12)
    links = ((0, 0), (0, 1), (1, 2), (2, 3), (2, 4), (3, 5))
    for first, second in links:
        ax.plot([ring1[first][0], ring2[second][0]], [ring1[first][1], ring2[second][1]],
                color=TEAL, linewidth=1.4)
    for first, third in ((1, 0), (4, 1), (5, 2)):
        ax.plot([ring2[first][0], beyond[third][0]], [ring2[first][1], beyond[third][1]],
                color=GRID, linewidth=1.4)
    ax.add_patch(Circle(center, 0.030, facecolor=ORANGE, edgecolor="white", linewidth=1.2, zorder=5))
    ax.text(center[0], center[1], "i", ha="center", va="center", color="white", fontsize=14, zorder=6)
    for node in ring1:
        ax.add_patch(Circle(node, 0.024, facecolor=BLUE, edgecolor="white", linewidth=1.0, zorder=5))
    for node in ring2:
        ax.add_patch(Circle(node, 0.024, facecolor=TEAL, edgecolor="white", linewidth=1.0, zorder=5))
    for node in beyond:
        ax.add_patch(Circle(node, 0.022, facecolor="#c8cfd8", edgecolor="white", linewidth=1.0, zorder=5))
    ax.text(0.50, 0.655, "1 tour", ha="center", fontsize=16, color=BLUE, weight="bold")
    ax.text(0.50, 0.825, "2 tours", ha="center", fontsize=16, color=TEAL, weight="bold")
    ax.text(0.863, 0.72, "au-delà :\nabsent de l'état", ha="center", fontsize=13.5, color=GRAY, linespacing=1.3)
    return fig


def swarm_communication_graph() -> plt.Figure:
    positions = archived_run()
    initial = positions[0]
    mask = adjacency(initial, COMMUNICATION_RADIUS)
    fig = plt.figure(figsize=(WIDTH, 7.2))
    header(fig, "Graphe de communication et masque d'adjacence",
           f"Positions initiales archivées · arêtes 0 < d ≤ r_c = {fr(COMMUNICATION_RADIUS, 1)}")
    left = fig.add_axes([0.06, 0.30, 0.50, 0.48])
    style_plot(left)
    left.set_title("(a) Voisinages au pas t = 0", pad=16)
    ys, xs = np.nonzero(np.triu(mask, 1))
    for first, second in zip(ys, xs):
        left.plot(initial[[first, second], 0], initial[[first, second], 1], color=GRAY, linewidth=0.9, zorder=1)
    left.scatter(initial[:, 0], initial[:, 1], s=46, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
    focal = int(np.argmax(mask.sum(axis=1)))
    left.add_patch(Circle(tuple(initial[focal]), COMMUNICATION_RADIUS, facecolor="none",
                          edgecolor=TEAL, linewidth=1.6, linestyle="--", zorder=3))
    left.annotate("portée r_c", xy=tuple(initial[focal] + np.array([0.0, COMMUNICATION_RADIUS])),
                  xytext=(initial[focal, 0] - 1.5, initial[focal, 1] + 1.75),
                  fontsize=14.5, color=TEAL, arrowprops={"arrowstyle": "->", "color": TEAL})
    left.set(xlim=(-2.2, 2.2), ylim=(-2.2, 2.2), aspect="equal", xlabel="x", ylabel="y")
    right = fig.add_axes([0.63, 0.30, 0.33, 0.48])
    right.set_title("(b) Masque d'adjacence A⁰", pad=16)
    right.imshow(mask, cmap="Blues", interpolation="nearest")
    right.set_xticks([])
    right.set_yticks([])
    right.set_xlabel("robot j")
    right.set_ylabel("robot i")
    return fig


def target_preparation_pipeline() -> plt.Figure:
    mask = raster_mask()
    cloud50 = shape_cloud("letters_A@50.npy")
    cloud20 = shape_cloud("letters_A@20.npy")
    fig = plt.figure(figsize=(WIDTH, 6.8))
    header(fig, "Préparation réelle d'un nuage cible", "Lettre A · grille 128 × 128 · échantillonnage sans doublon")
    axes = [fig.add_axes([0.055, 0.34, 0.27, 0.44]), fig.add_axes([0.375, 0.34, 0.27, 0.44]),
            fig.add_axes([0.695, 0.34, 0.27, 0.44])]
    rows, columns = np.nonzero(mask)
    style_plot(axes[0])
    axes[0].set_title("(a) Masque du trait", pad=14)
    axes[0].scatter(columns, 127 - rows, s=1.4, color=INK, marker="s")
    axes[0].set(xlim=(-4, 132), ylim=(-4, 132), aspect="equal", xlabel="pixel", ylabel="pixel")
    for ax, cloud, title, color, size in ((axes[1], cloud50, "(b) 50 points distincts", TEAL, 26),
                                          (axes[2], cloud20, "(c) 20 points distincts", ORANGE, 34)):
        style_plot(ax)
        ax.set_title(title, pad=14)
        ax.scatter(cloud[:, 0], cloud[:, 1], s=size, color=color, edgecolor="white", linewidth=0.6)
        ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), aspect="equal", xlabel="x", ylabel="y")
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
    return fig


def software_modules() -> plt.Figure:
    fig, ax = canvas(7.4)
    header(fig, "Responsabilités et flux de données", "Modules de l'implémentation actuelle · niveau scientifique")
    box(ax, 0.04, 0.79, 0.92, 0.08, "config/ · configurations YAML et configurations effectives",
        color=PALE_TEAL, fontsize=16)
    box(ax, 0.04, 0.62, 0.28, 0.12, "src/simulator\ndynamique · graphe\npannes", fontsize=13)
    box(ax, 0.04, 0.44, 0.28, 0.12, "src/shapes\nrendus · échantillonnage\nsplits", fontsize=13)
    box(ax, 0.36, 0.62, 0.28, 0.12, "src/models\nencodeur · messages\ndécodeur", fontsize=13)
    box(ax, 0.36, 0.44, 0.28, 0.12, "src/controllers\nGNN · regroupement\nmanuel", fontsize=13)
    box(ax, 0.68, 0.62, 0.28, 0.12, "src/training\nrollout · pertes\ncurriculum", color=PALE_TEAL, fontsize=13)
    box(ax, 0.68, 0.44, 0.28, 0.12, "src/metrics\nChamfer · succès\ncollisions", color=PALE_TEAL, fontsize=13)
    box(ax, 0.68, 0.26, 0.28, 0.12, "src/evaluation\ncampagnes · transferts\nrobustesse", color=PALE_TEAL, fontsize=13)
    box(ax, 0.36, 0.26, 0.28, 0.12, "src/reports\ntableaux · figures\nquestionnaire", fontsize=13)
    box(ax, 0.04, 0.08, 0.60, 0.10, "outputs/ · artefacts, graines,\nconfigurations effectives, rapports",
        color=PALE_ORANGE, fontsize=14)
    for x in (0.18, 0.50, 0.82):
        arrow(ax, (x, 0.785), (x, 0.745), color=TEAL)
    arrow(ax, (0.33, 0.68), (0.35, 0.68), color=BLUE)
    arrow(ax, (0.33, 0.50), (0.35, 0.50), color=BLUE)
    routed_arrow(ax, [(0.18, 0.61), (0.18, 0.575), (0.75, 0.575), (0.75, 0.745)], color=BLUE)
    routed_arrow(ax, [(0.18, 0.43), (0.18, 0.395), (0.70, 0.395), (0.70, 0.384)], color=BLUE)
    arrow(ax, (0.50, 0.61), (0.50, 0.572), color=BLUE)
    routed_arrow(ax, [(0.50, 0.43), (0.50, 0.395), (0.76, 0.395), (0.76, 0.384)], color=TEAL)
    routed_arrow(ax, [(0.82, 0.61), (0.82, 0.572)], color=TEAL)
    routed_arrow(ax, [(0.82, 0.43), (0.82, 0.384)], color=TEAL)
    routed_arrow(ax, [(0.96, 0.68), (0.985, 0.68), (0.985, 0.32), (0.96, 0.32)], color=TEAL)
    arrow(ax, (0.67, 0.32), (0.65, 0.32), color=TEAL)
    routed_arrow(ax, [(0.82, 0.25), (0.82, 0.13), (0.655, 0.13)], color=TEAL)
    arrow(ax, (0.50, 0.25), (0.50, 0.192), color=TEAL)
    return fig


def formation_principle() -> plt.Figure:
    target_angles = np.arange(24) * (2 * np.pi / 24)
    target = np.column_stack((np.cos(target_angles), np.sin(target_angles)))
    indices = np.arange(18)
    angles = indices * 2.399963
    radii = 1.85 - 0.055 * indices
    start = np.column_stack((radii * np.cos(angles) + 0.08 * np.sin(3 * indices),
                             radii * np.sin(angles) + 0.08 * np.cos(2 * indices)))
    final = target[(indices * 24) // 18] + 0.06 * np.column_stack((np.sin(5 * indices), np.cos(7 * indices)))
    middle = 0.45 * start + 0.55 * final + 0.10 * np.column_stack((np.sin(2 * indices + 1), np.cos(3 * indices)))
    fig = plt.figure(figsize=(WIDTH, 6.6))
    header(fig, "Du départ dispersé à la formation approchée", "Scénario conceptuel déterministe · aucune exécution du contrôleur")
    panels = ((start, "(a) Départs dispersés"), (middle, "(b) Étape intermédiaire"), (final, "(c) Formation approchée"))
    for index, (points, title) in enumerate(panels):
        ax = fig.add_axes([0.055 + 0.32 * index, 0.32, 0.27, 0.46])
        style_plot(ax)
        ax.set_title(title, pad=14)
        ax.scatter(target[:, 0], target[:, 1], s=40, marker="x", color=TEAL, linewidth=1.8, label="cible")
        ax.scatter(points[:, 0], points[:, 1], s=34, color=ORANGE, edgecolor="white", linewidth=0.6, label="robots")
        ax.set(xlim=(-2.1, 2.1), ylim=(-2.1, 2.1), aspect="equal", xlabel="x", ylabel="y")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
    return fig


def _bars(ax: plt.Axes, labels: tuple[str, ...], first: tuple[float, ...], second: tuple[float, ...],
          first_std: tuple[float, ...] | None = None, second_std: tuple[float, ...] | None = None) -> None:
    xs = np.arange(len(labels))
    width = 0.36
    first_offsets = first_std if first_std else (0.0,) * len(first)
    second_offsets = second_std if second_std else (0.0,) * len(second)
    ax.bar(xs - width / 2, first, width, yerr=first_std, color=BLUE, edgecolor="white",
           linewidth=0.8, capsize=4, label="GNN historique")
    ax.bar(xs + width / 2, second, width, yerr=second_std, color=ORANGE, edgecolor="white",
           linewidth=0.8, capsize=4, label="Regroupement")
    for position, value, offset in zip(xs - width / 2, first, first_offsets):
        ax.text(position, value + offset + 0.06, fr(value), ha="center", va="bottom",
                fontsize=12.5, color=BLUE)
    for position, value, offset in zip(xs + width / 2, second, second_offsets):
        ax.text(position, value + offset + 0.06, fr(value), ha="center", va="bottom",
                fontsize=12.5, color=ORANGE)
    ax.set_xticks(xs, labels)
    style_plot(ax)


def historical_pilot_summary() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.6))
    header(fig, "Synthèse du protocole pilote historique", "Valeurs consignées · dix graines par condition")
    left = fig.add_axes([0.085, 0.30, 0.375, 0.48])
    left.set_title("(a) Taille de l'essaim, sans panne", pad=16)
    _bars(left, tuple(str(n) for n in SCALE_ROBOTS), SCALE_GNN, SCALE_FLOCKING)
    left.set(xlim=(-0.6, 2.6), ylim=(0, 4.6), xlabel="Robots", ylabel="Distance de Chamfer moyenne")
    left.legend(loc="upper left", frameon=False, fontsize=13.5)
    right = fig.add_axes([0.59, 0.30, 0.375, 0.48])
    right.set_title("(b) Retraits, 20 robots", pad=16)
    _bars(right, tuple(f"{int(f * 100)} %" for f in FAILURE_FRACTIONS), FAILURE_GNN, FAILURE_FLOCKING,
          FAILURE_GNN_STD, FAILURE_FLOCKING_STD)
    right.set(xlim=(-0.6, 3.6), ylim=(0, 6.0), xlabel="Fraction retirée à mi-épisode",
              ylabel="Distance de Chamfer moyenne")
    return fig


def historical_chamfer_vs_failures() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.0))
    header(fig, "Distance de Chamfer selon la fraction de robots retirés", "Moyenne ± 1 écart-type sur dix épisodes · prototype historique")
    percents = [int(f * 100) for f in FAILURE_FRACTIONS]
    left = fig.add_axes([0.085, 0.30, 0.375, 0.48])
    style_plot(left)
    left.set_title("(a) GNN historique", pad=16)
    left.errorbar(percents, FAILURE_GNN, yerr=FAILURE_GNN_STD, color=BLUE, marker="o", markersize=7,
                  capsize=5, linewidth=2.2)
    left.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.5)
    left.text(0.6, 0.53, "seuil de succès τ = 0,5", fontsize=14, color=GRAY)
    left.set(xlim=(-1.5, 21.5), ylim=(0.30, 0.95), xlabel="Fraction retirée (%)",
             ylabel="Distance de Chamfer moyenne")
    left.set_xticks(percents)
    right = fig.add_axes([0.59, 0.30, 0.375, 0.48])
    style_plot(right)
    right.set_title("(b) Règle de regroupement", pad=16)
    right.errorbar(percents, FAILURE_FLOCKING, yerr=FAILURE_FLOCKING_STD, color=ORANGE, marker="s",
                   markersize=7, capsize=5, linewidth=2.2)
    right.set(xlim=(-1.5, 21.5), ylim=(2.4, 6.0), xlabel="Fraction retirée (%)",
              ylabel="Distance de Chamfer moyenne")
    right.set_xticks(percents)
    return fig


def failure_timeline() -> plt.Figure:
    fig, ax = canvas(5.4)
    header(fig, "Chronologie d'un épisode avec retrait à mi-parcours", "Déconnexion propre · mesure finale sur les robots actifs")
    ax.plot([0.08, 0.92], [0.66, 0.66], color=BLUE, linewidth=3.0)
    ax.plot([0.08, 0.50], [0.38, 0.38], color=ORANGE, linewidth=3.0)
    ax.plot([0.50, 0.92], [0.38, 0.38], color=GRAY, linewidth=2.2, linestyle="--")
    ax.plot([0.50, 0.50], [0.30, 0.74], color=INK, linewidth=1.4, linestyle=":")
    for x, label in ((0.08, "pas 0"), (0.50, "pas 10"), (0.92, "pas 20")):
        ax.plot([x, x], [0.24, 0.27], color=GRAY, linewidth=1.4)
        ax.text(x, 0.19, label, ha="center", fontsize=15, color=GRAY)
    ax.text(0.08, 0.705, "robots actifs", ha="left", va="bottom", fontsize=15.5, color=BLUE)
    ax.text(0.08, 0.425, "robots retirés", ha="left", va="bottom", fontsize=15.5, color=ORANGE)
    ax.text(0.50, 0.78, "retrait à mi-épisode (fraction f)", ha="center", fontsize=16, weight="bold")
    ax.text(0.71, 0.315, "figés · exclus du graphe, des collisions et de la mesure",
            ha="center", va="top", fontsize=14.5, color=GRAY)
    ax.add_patch(Circle((0.92, 0.66), 0.014, facecolor=TEAL, edgecolor="white", linewidth=1.0))
    ax.text(0.92, 0.72, "mesure finale sur les actifs", ha="right", fontsize=15, color=TEAL)
    return fig


def historical_scaling() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.2))
    header(fig, "Passage à l'échelle du pilote historique", "Même politique évaluée à 20, 40 et 100 robots")
    ax = fig.add_axes([0.085, 0.30, 0.83, 0.48])
    _bars(ax, tuple(str(n) for n in SCALE_ROBOTS), SCALE_GNN, SCALE_FLOCKING)
    ax.set(xlim=(-0.6, 2.6), ylim=(0, 4.6), xlabel="Robots", ylabel="Distance de Chamfer moyenne")
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    for index, success in enumerate(SCALE_SUCCESS):
        ax.text(index - 0.18, 0.22, f"succès {int(success * 100)} %", ha="center", fontsize=13,
                color="white", weight="bold", rotation=90)
    for index in range(3):
        ax.text(index + 0.18, 0.22, "succès 0 %", ha="center", fontsize=13, color="white", weight="bold", rotation=90)
    return fig


def threshold_crossing_schema() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.2))
    header(fig, "Premier franchissement du seuil historique", "Courbes construites pour la convention τ = 0,5")
    ax = fig.add_axes([0.085, 0.30, 0.83, 0.48])
    steps = np.arange(21)
    admitted = np.array([1.70, 1.50, 1.32, 1.15, 1.00, 0.88, 0.78, 0.70, 0.63, 0.57, 0.52, 0.48,
                         0.45, 0.43, 0.47, 0.52, 0.58, 0.62, 0.60, 0.57, 0.55])
    final_only = np.array([1.60, 1.48, 1.36, 1.24, 1.13, 1.03, 0.94, 0.86, 0.79, 0.73, 0.68, 0.64,
                           0.61, 0.58, 0.56, 0.54, 0.53, 0.52, 0.51, 0.505, 0.49])
    style_plot(ax)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.6)
    ax.text(0.4, 0.52, "seuil τ = 0,5", fontsize=14.5, color=GRAY)
    ax.plot(steps, admitted, color=BLUE, linewidth=2.6, label="franchissement avant le dernier pas")
    ax.plot(steps, final_only, color=PURPLE, linewidth=2.2, linestyle="--", label="franchissement au seul pas 20")
    ax.scatter([11], [admitted[11]], s=60, color=BLUE, edgecolor="white", linewidth=1.0, zorder=5)
    ax.annotate("premier franchissement : pas 11\ncomptabilisé comme succès", xy=(11, admitted[11]),
                xytext=(2.2, 0.20), fontsize=14.5, color=BLUE,
                arrowprops={"arrowstyle": "->", "color": BLUE, "connectionstyle": "arc3,rad=-0.2"})
    ax.scatter([20], [final_only[20]], s=60, color=PURPLE, edgecolor="white", linewidth=1.0, zorder=5)
    ax.annotate("au seul dernier pas : exclu", xy=(20, final_only[20]), xytext=(13.4, 0.30),
                fontsize=14.5, color=PURPLE, arrowprops={"arrowstyle": "->", "color": PURPLE})
    ax.set(xlim=(0, 20.6), ylim=(0.1, 1.8), xlabel="Pas t", ylabel="Distance de forme")
    ax.set_xticks([0, 10, 20])
    ax.legend(loc="upper right", frameon=False, fontsize=13.5)
    return fig


def historical_seed_dispersion() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.0))
    header(fig, "Dispersion des dix départs du pilote à 20 robots", "Valeurs consignées, triées par distance croissante")
    ax = fig.add_axes([0.085, 0.30, 0.83, 0.48])
    colors = [TEAL if success else BLUE for success in SEED_SUCCESS]
    xs = np.arange(len(SEED_CHAMFER))
    ax.bar(xs, SEED_CHAMFER, 0.62, color=colors, edgecolor="white", linewidth=0.8)
    for position, value in zip(xs, SEED_CHAMFER):
        ax.text(position, value + 0.012, fr(value), ha="center", va="bottom", fontsize=12.5, color=INK)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.5)
    ax.text(9.45, 0.508, "τ = 0,5", ha="right", fontsize=14, color=GRAY)
    ax.axhline(0.592, color=INK, linestyle=":", linewidth=1.5)
    ax.text(9.45, 0.600, "moyenne 0,592", ha="right", fontsize=14, color=INK)
    ax.set_xticks(xs, SEED_LABELS)
    ax.set(xlim=(-0.7, 9.7), ylim=(0.40, 0.82), xlabel="Graine de l'épisode", ylabel="Distance de Chamfer finale")
    ax.annotate("sous τ au dernier pas,\nfranchissement tardif exclu", xy=(0, 0.499), xytext=(0.4, 0.435),
                fontsize=13.5, color=GRAY, arrowprops={"arrowstyle": "->", "color": GRAY})
    ax.annotate("succès malgré 0,521", xy=(2, 0.521), xytext=(2.6, 0.70), fontsize=13.5, color=TEAL,
                arrowprops={"arrowstyle": "->", "color": TEAL})
    return fig


def historical_curriculum_per_target() -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH, 6.2))
    header(fig, "Pertes par cible à la fin de chaque étape du curriculum", "Cibles déjà vues et cibles introduites · prototype historique")
    ax = fig.add_axes([0.085, 0.30, 0.83, 0.48])
    xs = np.arange(len(CURRICULUM_STAGES))
    width = 0.36
    old_values = [0.0 if value is None else value for value in CURRICULUM_OLD]
    ax.bar(xs - width / 2, old_values, width, color=BLUE, edgecolor="white", linewidth=0.8, label="cibles déjà vues")
    ax.bar(xs + width / 2, CURRICULUM_NEW, width, color=TEAL, edgecolor="white", linewidth=0.8,
           label="cibles introduites")
    for position, value in zip(xs + width / 2, CURRICULUM_NEW):
        ax.text(position, value + 0.008, fr(value), ha="center", va="bottom", fontsize=12.5, color=TEAL)
    for position, value in zip(xs - width / 2, CURRICULUM_OLD):
        if value is None:
            ax.text(position, 0.012, "—", ha="center", va="bottom", fontsize=13, color=BLUE)
        else:
            ax.text(position, value + 0.008, fr(value), ha="center", va="bottom", fontsize=12.5, color=BLUE)
    for position, count in zip(xs, CURRICULUM_TARGETS):
        ax.text(position, 0.365, f"{count} cibles", ha="center", fontsize=13.5, color=GRAY)
    ax.set_xticks(xs, CURRICULUM_STAGES)
    ax.set(xlim=(-0.6, 3.6), ylim=(0, 0.40), xlabel="Étape du curriculum réduit", ylabel="Perte moyenne par cible")
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    style_plot(ax)
    return fig


BUILDERS = (
    experimental_chain,
    consensus_dynamics,
    flocking_rules,
    message_aggregation,
    information_reach,
    swarm_communication_graph,
    target_preparation_pipeline,
    software_modules,
    formation_principle,
    historical_pilot_summary,
    historical_chamfer_vs_failures,
    failure_timeline,
    historical_scaling,
    threshold_crossing_schema,
    historical_seed_dispersion,
    historical_curriculum_per_target,
)


def save_figure(fig: plt.Figure, name: str) -> None:
    if name not in FIGURE_NAMES:
        raise ValueError(f"Unauthorized figure name: {name}")
    subject = "Original deterministic diagram; provenance: " + "; ".join(SOURCE_REFERENCES[name])
    fig.savefig(OUTPUT_DIRECTORY / f"{name}.pdf", format="pdf", metadata={
        "Title": name,
        "Author": "Original scientific figures for swarm-gnn",
        "Subject": subject,
        "Creator": "generate_manuscript_figures.py / Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    })
    fig.savefig(OUTPUT_DIRECTORY / f"{name}.png", format="png", dpi=300, metadata={
        "Title": name,
        "Description": subject,
        "Software": "generate_manuscript_figures.py / Matplotlib",
    })
    plt.close(fig)
    for extension in ("pdf", "png"):
        path = OUTPUT_DIRECTORY / f"{name}.{extension}"
        assert path.is_file() and path.stat().st_size > 0
        print(f"SAVED: {path} ({path.stat().st_size} bytes)")


def main() -> None:
    preflight_assets()
    verify_checks()
    for name, builder in zip(FIGURE_NAMES, BUILDERS, strict=True):
        save_figure(builder(), name)
    print("DONE: sixteen vector PDFs and sixteen 300-dpi PNGs; no training, optimizer or model instantiation.")


if __name__ == "__main__":
    main()
