"""Génère les schémas reproductibles utilisés dans le mémoire LaTeX."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def mean_aggregation() -> None:
    """Illustre la moyenne des messages locaux employée par le GNN."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.set_xlim(0, 9)
    axis.set_ylim(0, 4.8)
    axis.axis("off")
    receiver = (4.45, 2.3)
    neighbours = [(1.45, 3.7, "$j_1$", "#d6eaf8"), (1.45, 2.3, "$j_2$", "#d5f5e3"), (1.45, 0.9, "$j_3$", "#fdebd0")]
    for x, y, label, color in neighbours:
        rounded_box(axis, x - 0.62, y - 0.32, 1.25, 0.64, f"message\n$m_{{i{label[1:-1]}}}$", color)
        axis.add_patch(FancyArrowPatch((x + 0.65, y), (3.55, 2.3), arrowstyle="->", mutation_scale=13, linewidth=1.3, color=DARK))
    rounded_box(axis, 3.55, 1.65, 1.8, 1.3, r"moyenne" + "\n" + r"$\bar m_i=\frac{1}{|\mathcal{N}(i)|}\sum_j m_{ij}$", "#e8daef")
    axis.add_patch(FancyArrowPatch((5.35, 2.3), (6.05, 2.3), arrowstyle="->", mutation_scale=13, linewidth=1.3, color=DARK))
    rounded_box(axis, 6.05, 1.65, 1.65, 1.3, r"mise à jour" + "\n" + r"$h_i^+=\psi(h_i,\bar m_i)$", "#f9e79f")
    axis.text(receiver[0], 4.3, "Les messages sont calculés séparément, puis résumés sans dépendre de leur ordre.", ha="center", fontsize=10.5, color=DARK)
    axis.text(receiver[0], 0.25, "La moyenne maintient une échelle comparable quand le nombre de voisins varie.", ha="center", fontsize=9.5, color=DARK)
    save(figure, "aggregation-mean.png")


def pilot_summary_grid() -> None:
    """Produit une figure de synthèse à partir des agrégats du protocole pilote."""
    summary = Path("results/evaluation-pilot/summary.csv")
    if not summary.exists():
        return
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    styles = {"gnn": (BLUE, "GNN"), "gathering": (ORANGE, "Regroupement")}
    for method, (color, label) in styles.items():
        subset = [row for row in rows if row["method"] == method and float(row["failure_fraction"]) == 0.0]
        subset.sort(key=lambda row: int(row["robots"]))
        axes[0].plot([int(row["robots"]) for row in subset], [float(row["shape_distance_mean"]) for row in subset], marker="o", color=color, label=label)
        subset = [row for row in rows if row["method"] == method and int(row["robots"]) == 20]
        subset.sort(key=lambda row: float(row["failure_fraction"]))
        axes[1].plot([100 * float(row["failure_fraction"]) for row in subset], [float(row["shape_distance_mean"]) for row in subset], marker="o", color=color, label=label)
    axes[0].set(title="Passage à l'échelle", xlabel="Nombre de robots", ylabel="Distance de Chamfer moyenne")
    axes[1].set(title="Pannes à 20 robots", xlabel="Robots retirés (%)", ylabel="Distance de Chamfer moyenne")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    save(figure, "pilot-summary-grid.png")


def curriculum_losses() -> None:
    """Superpose les courbes de perte des quatre étapes du curriculum réduit."""
    stages = [("geometry", "Géométrie"), ("digits", "Chiffres"), ("symbols", "Symboles"), ("numbers", "Nombres")]
    curves = []
    for directory, label in stages:
        path = Path(f"results/curriculum/{directory}/loss.csv")
        if not path.exists():
            return
        curves.append((label, [float(value) for value in path.read_text().split()]))
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharey=True)
    for axis, (label, values) in zip(axes.flat, curves):
        axis.plot(range(1, len(values) + 1), values, color=BLUE, linewidth=1.4)
        axis.set(title=label, xlabel="Itération")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("Perte")
    axes[1][0].set_ylabel("Perte")
    figure.suptitle("Pertes enregistrées à chaque étape du curriculum réduit", color=DARK)
    figure.tight_layout()
    save(figure, "curriculum-losses.png")


def shape_pipeline() -> None:
    """Illustre le passage d'une image à un nuage de points normalisé."""
    import numpy as np
    from PIL import Image

    image_path = Path("data/shapes/geometry/geometry_circle.png")
    points_path = Path("data/shapes/geometry/geometry_circle.npy")
    if not image_path.exists() or not points_path.exists():
        return
    image = np.asarray(Image.open(image_path))
    points = np.load(points_path)

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image rendue", fontsize=10)
    threshold = float(np.median(image))
    axes[1].imshow((image > threshold).astype(float), cmap="gray")
    axes[1].set_title("Masque binarisé", fontsize=10)
    axes[2].scatter(points[:, 0], -points[:, 1], s=26, color=BLUE, edgecolor=DARK, linewidth=0.7)
    axes[2].set_title("Nuage échantillonné et normalisé", fontsize=10)
    axes[2].set_aspect("equal")
    axes[2].set_xlim(-1.0, 1.0)
    axes[2].set_ylim(-1.0, 1.0)
    axes[2].grid(alpha=0.25)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.tight_layout()
    save(figure, "shape-pipeline.png")


def module_architecture() -> None:
    """Dessine les modules du prototype et les artefacts qui circulent entre eux."""
    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.set_xlim(0, 10.5)
    axis.set_ylim(0, 5.2)
    axis.axis("off")

    rounded_box(axis, 0.3, 3.5, 2.0, 1.1, "configs/\nYAML versionnés", "#f9e79f")
    rounded_box(axis, 3.0, 3.5, 2.0, 1.1, "src/shapes\nformes cibles", "#d5f5e3")
    rounded_box(axis, 5.7, 3.5, 2.0, 1.1, "src/models\ncontrôleurs", "#d6eaf8")
    rounded_box(axis, 8.2, 3.5, 2.0, 1.1, "src/eval\nmesures, CSV", "#e8daef")

    rounded_box(axis, 3.0, 1.3, 2.0, 1.1, "src/sim\nsimulation", "#d6eaf8")
    rounded_box(axis, 5.7, 1.3, 2.0, 1.1, "src/train\nentraînement", "#fdebd0")
    rounded_box(axis, 0.3, 1.3, 2.0, 1.1, "tests/\npropriétés", "#f2f3f4")
    rounded_box(axis, 8.2, 1.3, 2.0, 1.1, "src/web + scripts\ndémonstration, figures", "#fadbd8")

    arrows = [((2.35, 4.05), (2.95, 4.05)), ((5.05, 4.05), (5.65, 4.05)), ((7.75, 4.05), (8.15, 4.05)),
              ((4.0, 3.45), (4.0, 2.45)), ((6.7, 3.45), (6.7, 2.45)), ((5.05, 1.85), (5.65, 1.85)),
              ((7.75, 1.85), (8.15, 1.85))]
    for start, end in arrows:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DARK))
    axis.add_patch(FancyArrowPatch((4.0, 1.25), (2.45, 1.25), connectionstyle="arc3,rad=0.3", arrowstyle="->", mutation_scale=13, linewidth=1.3, color=ORANGE, linestyle="--"))

    axis.text(5.25, 4.95, "Les modules échangent des fichiers et des tableaux, jamais des objets en mémoire.", ha="center", fontsize=10.5, color=DARK)
    axis.text(4.6, 0.72, "La rétropropagation traverse src/sim et src/models ; src/eval reste lecture seule.", ha="center", fontsize=9.5, color=ORANGE)
    save(figure, "module-architecture.png")


def consensus_convergence() -> None:
    """Illustre la convergence d'un consensus sur un graphe linéaire de cinq agents."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    initial = [0.1, 0.55, 0.95, 0.35, 0.75]
    steps = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    history = [initial]
    for _ in steps[1:]:
        previous = history[-1]
        updated = [previous[0]]
        for node in range(1, len(previous) - 1):
            updated.append((previous[node - 1] + previous[node] + previous[node + 1]) / 3)
        updated.append(previous[-1])
        history.append(updated)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(steps)))
    for step, values, color in zip(steps, history, colors):
        axes[0].plot(range(len(values)), values, marker="o", color=color, label=f"passe {step:g}")
    axes[0].set(title="Valeurs des agents", xlabel="Identifiant de l'agent", ylabel="Opinion")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    spread = [max(values) - min(values) for values in history]
    axes[1].semilogy(steps, spread, marker="o", color=BLUE)
    axes[1].set(title="Réduction de l'écart", xlabel="Passe de voisinage", ylabel="max $-$ min")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    save(figure, "consensus-convergence.png")


def flocking_rules() -> None:
    """Dessine les trois composantes d'une règle de flocking sur un essaim miniature."""
    figure, axis = plt.subplots(figsize=(9.5, 4.6))
    axis.set_xlim(0, 9.5)
    axis.set_ylim(0, 4.6)
    axis.set_aspect("equal")
    axis.axis("off")

    agents = [(4.6, 2.3), (3.3, 3.0), (5.9, 2.9), (3.6, 1.3), (5.8, 1.5), (7.6, 3.4), (1.6, 2.2)]
    for index, (x, y) in enumerate(agents):
        draw_robot(axis, x, y, f"{index}", highlighted=index == 0)
    center = (np.mean([x for x, _ in agents]), np.mean([y for _, y in agents]))
    axis.add_patch(FancyArrowPatch((4.6, 2.3), center, arrowstyle="->", mutation_scale=15, linewidth=1.8, color="#27ae60", connectionstyle="arc3,rad=0.2"))
    axis.text(center[0] + 0.15, center[1] + 0.1, "cohésion", color="#27ae60", fontsize=10)
    axis.add_patch(FancyArrowPatch((3.3, 3.0), (2.9, 3.4), arrowstyle="->", mutation_scale=15, linewidth=1.8, color="#c0392b"))
    axis.text(2.2, 3.55, "séparation", color="#c0392b", fontsize=10)
    axis.plot([5.9, 6.9], [2.9, 2.55], color="#2980b9", linewidth=2.0)
    axis.add_patch(FancyArrowPatch((5.9, 2.9), (6.9, 2.55), arrowstyle="->", mutation_scale=15, linewidth=1.8, color="#2980b9"))
    axis.text(6.35, 3.15, "alignement", color="#2980b9", fontsize=10)
    axis.text(4.75, 0.28, "Chaque agent pondère cohésion, séparation et alignement à partir de son voisinage local.", ha="center", fontsize=10, color=DARK)
    save(figure, "flocking-rules.png")


def loss_terms() -> None:
    """Réunit les quatre termes de la fonction de perte avec leur rôle."""
    figure, axis = plt.subplots(figsize=(10.5, 4.4))
    axis.set_xlim(0, 10.5)
    axis.set_ylim(0, 4.4)
    axis.axis("off")
    terms = [
        (0.3, 2.6, "forme\n$\\lambda_f\\,d_C(P,Q)$", "#d6eaf8"),
        (2.85, 2.6, "collisions\n$\\lambda_c\\,\\mathcal{E}_{col}$", "#fdebd0"),
        (5.4, 2.6, "mouvement\n$\\lambda_u\\|u_i\\|^2$", "#d5f5e3"),
        (7.95, 2.6, "couverture\n$\\lambda_r\\,\\mathcal{E}_{cov}$", "#e8daef"),
    ]
    for x, y, text, color in terms:
        rounded_box(axis, x, y, 2.2, 1.15, text, color)
        axis.add_patch(FancyArrowPatch((x + 1.1, y), (5.25, 1.85), arrowstyle="->", mutation_scale=13, linewidth=1.3, color=DARK, connectionstyle="arc3,rad=0.08"))
    rounded_box(axis, 3.9, 0.95, 2.7, 0.85, "perte totale\n$\\mathcal{L}$", "#f9e79f")
    axis.text(5.25, 3.95, "Une seule perte par épisode est rétropropagée vers les paramètres partagés.", ha="center", fontsize=10.5, color=DARK)
    axis.text(5.25, 0.42, "Les poids $\\lambda$ sont versionnés dans les configurations.", ha="center", fontsize=9.5, color=DARK)
    save(figure, "loss-terms.png")


def chamfer_geometry() -> None:
    """Montre les distances retenues par la distance de Chamfer symétrique."""
    figure, axis = plt.subplots(figsize=(8.6, 4.2))
    axis.set_xlim(0, 8.6)
    axis.set_ylim(0, 4.2)
    axis.set_aspect("equal")
    axis.axis("off")
    target = np.array([[1.2 + i * 1.4, 3.1 + 0.35 * np.sin(i)] for i in range(5)])
    robots = np.array([[1.5 + i * 1.35, 1.15 + 0.3 * np.cos(i)] for i in range(5)])
    for x, y in target:
        axis.add_patch(Circle((x, y), 0.11, facecolor=BLUE, edgecolor=DARK, linewidth=1.0, zorder=3))
    for x, y in robots:
        draw_robot(axis, x, y, "", highlighted=True)
    for rx, ry in robots:
        index = int(np.argmin(np.hypot(*(target - np.array([rx, ry])).T)))
        axis.plot([rx, target[index, 0]], [ry, target[index, 1]], color="#839192", linestyle="--", linewidth=1.1)
    axis.text(4.3, 3.85, "Points cibles $Q$ (bleu) et robots $P$ (orange)", ha="center", fontsize=10.5, color=DARK)
    axis.text(4.3, 0.25, "Chaque terme relie un point au plus proche de l'autre ensemble, dans les deux sens.", ha="center", fontsize=9.5, color=DARK)
    save(figure, "chamfer-geometry.png")


def curriculum_stages() -> None:
    """Planche des quatre étapes du curriculum réduit avec leurs cibles."""
    from PIL import Image

    samples = [
        ("data/shapes/geometry/geometry_circle.png", "Géométrie", "3 cibles"),
        ("data/shapes/digits/digit_0.png", "Chiffres", "+3 cibles"),
        ("data/shapes/alphabet/alphabet_A.png", "Symboles", "+3 cibles"),
        ("data/shapes/numbers/number_010.png", "Nombres", "+2 cibles"),
    ]
    figure, axes = plt.subplots(1, 4, figsize=(9.6, 3.0))
    for axis, (path, label, count) in zip(axes, samples):
        if not Path(path).exists():
            plt.close(figure)
            return
        axis.imshow(Image.open(path), cmap="gray")
        axis.set_title(label, fontsize=10.5, color=DARK)
        axis.set_xlabel(count, fontsize=9, color=DARK)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Progression du curriculum réduit : les cibles ajoutées rejoignent celles déjà vues", color=DARK)
    figure.tight_layout()
    save(figure, "curriculum-stages.png")


def differentiable_rollout() -> None:
    """Schéma de la rétropropagation à travers les pas de simulation."""
    figure, axis = plt.subplots(figsize=(10.2, 4.0))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 4.0)
    axis.axis("off")
    for step in range(5):
        x = 0.5 + step * 1.95
        rounded_box(axis, x, 2.15, 1.6, 0.95, f"pas $t={step + 1}$\nGNN $\\rightarrow$ simu", "#d6eaf8")
        if step < 4:
            axis.add_patch(FancyArrowPatch((x + 1.6, 2.62), (x + 1.95, 2.62), arrowstyle="->", mutation_scale=13, linewidth=1.3, color=DARK))
    rounded_box(axis, 8.5, 0.55, 1.5, 0.8, "perte $\\mathcal{L}$", "#f9e79f")
    axis.add_patch(FancyArrowPatch((9.25, 2.1), (9.25, 1.4), arrowstyle="->", mutation_scale=14, linewidth=1.5, color=DARK))
    axis.add_patch(FancyArrowPatch((8.4, 0.95), (0.9, 0.95), arrowstyle="->", mutation_scale=15, linewidth=1.8, color=ORANGE))
    axis.text(4.6, 1.25, "rétropropagation à travers les $T$ pas enregistrés", color=ORANGE, ha="center", fontsize=10)
    axis.text(5.1, 3.55, "Les $T$ pas forment un graphe de calcul ; la mémoire croît avec la durée de l'épisode.", ha="center", fontsize=10.5, color=DARK)
    save(figure, "differentiable-rollout.png")


def episode_cycle() -> None:
    """Cycle de vie d'un épisode : tirage, boucle de pas, mesure finale."""
    figure, axis = plt.subplots(figsize=(10.2, 4.2))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 4.2)
    axis.axis("off")
    rounded_box(axis, 0.3, 2.9, 2.2, 0.95, "tirage initial\npositions, graine", "#d5f5e3")
    rounded_box(axis, 3.15, 2.9, 2.2, 0.95, "graphe de\nvoisinage", "#d6eaf8")
    rounded_box(axis, 6.0, 2.9, 2.2, 0.95, "GNN partagé\ncommandes", "#fdebd0")
    rounded_box(axis, 8.55, 2.9, 1.4, 0.95, "intégration\net clip", "#e8daef")
    for start, end in [(2.5, 3.15), (5.35, 6.0), (8.2, 8.55)]:
        axis.add_patch(FancyArrowPatch((start, 3.37), (end, 3.37), arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DARK))
    axis.add_patch(FancyArrowPatch((9.25, 2.85), (4.25, 2.35), connectionstyle="arc3,rad=0.18", arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DARK))
    axis.text(6.8, 2.28, "$T$ fois", fontsize=9.5, color=DARK)
    rounded_box(axis, 3.15, 0.6, 2.6, 0.95, "mesure finale\nChamfer, collisions", "#f9e79f")
    axis.add_patch(FancyArrowPatch((3.05, 2.6), (3.6, 1.6), arrowstyle="->", mutation_scale=14, linewidth=1.4, color=DARK, connectionstyle="arc3,rad=-0.25"))
    axis.add_patch(FancyArrowPatch((1.4, 2.85), (1.4, 1.07), arrowstyle="->", mutation_scale=13, linewidth=1.2, color="#839192", linestyle="--"))
    axis.add_patch(FancyArrowPatch((1.4, 1.07), (3.1, 1.07), arrowstyle="->", mutation_scale=13, linewidth=1.2, color="#839192", linestyle="--"))
    axis.text(5.1, 3.85, "Un épisode : mêmes étapes pour l'entraînement et l'évaluation.", ha="center", fontsize=10.5, color=DARK)
    axis.text(2.0, 1.5, "forme cible", fontsize=9, color="#839192")
    save(figure, "episode-cycle.png")


def failure_timeline() -> None:
    """Frise d'un scénario de panne : retrait à mi-épisode, puis mesure."""
    figure, axis = plt.subplots(figsize=(10.0, 3.8))
    axis.set_xlim(0, 10.0)
    axis.set_ylim(0, 3.8)
    axis.axis("off")
    axis.add_patch(FancyArrowPatch((0.6, 2.2), (9.4, 2.2), arrowstyle="->", mutation_scale=15, linewidth=1.5, color=DARK))
    ticks = [(1.2, "t=0\n10 actifs"), (3.6, "t=10\nretrait de 2"), (6.2, "t=15\n8 actifs"), (8.7, "t=20\nmesure")]
    for x, label in ticks:
        axis.plot([x], [2.2], marker="o", markersize=8, color=BLUE, zorder=3)
        axis.text(x, 2.5, label.replace(chr(10), " "), ha="center", fontsize=9.5, color=DARK)
    for index in range(10):
        x = 1.2 + index * 0.24
        removed = index >= 8
        color = "#aab7b8" if removed else BLUE
        axis.add_patch(Circle((x, 1.35), 0.09, facecolor=color, edgecolor=DARK, linewidth=0.7))
    axis.text(3.6, 0.85, "les deux robots gris sont retirés", ha="center", fontsize=9, color="#839192")
    for index in range(8):
        x = 6.2 + index * 0.24
        axis.add_patch(Circle((x, 1.35), 0.09, facecolor=BLUE, edgecolor=DARK, linewidth=0.7))
    axis.text(5.0, 3.3, "Les robots retirés sont figés ; la mesure n'utilise que les robots actifs.", ha="center", fontsize=10.5, color=DARK)
    axis.text(7.4, 0.85, "8 robots actifs", ha="center", fontsize=9, color="#839192")
    save(figure, "failure-timeline.png")


def threshold_convergence() -> None:
    """Distance de forme au fil des pas avec le seuil de succès."""
    figure, axis = plt.subplots(figsize=(8.8, 4.0))
    steps = np.arange(1, 21)
    distance = 0.95 * np.exp(-0.16 * steps) + 0.42 + 0.03 * np.sin(steps * 1.7)
    axis.plot(steps, distance, marker="o", markersize=4, color=BLUE, label="distance de forme")
    axis.axhline(0.5, color=ORANGE, linestyle="--", linewidth=1.5, label="seuil $\\tau=0{,}5$")
    crossing = int(np.argmax(distance <= 0.5)) + 1 if (distance <= 0.5).any() else None
    if crossing:
        axis.axvline(crossing, color="#839192", linestyle=":", linewidth=1.2)
        axis.annotate(f"$t_{{conv}}={crossing}$", xy=(crossing, 0.52), xytext=(crossing + 1.2, 0.72), arrowprops=dict(arrowstyle="->", color=DARK), fontsize=9.5, color=DARK)
    axis.set(xlabel="Pas de simulation", ylabel="Distance de forme", xlim=(1, 20))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    save(figure, "threshold-convergence.png")


def message_depth() -> None:
    """Portée de l'information selon le nombre de tours de messages."""
    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    axis.set_xlim(0, 9.0)
    axis.set_ylim(0, 4.4)
    axis.set_aspect("equal")
    axis.axis("off")
    center = (4.5, 2.3)
    rings = [(1.0, 5, "tour 1", "#d6eaf8"), (2.0, 9, "tour 2", "#eaf2f8")]
    for radius, count, label, color in rings:
        axis.add_patch(Circle(center, radius, facecolor=color, edgecolor="#839192", linestyle="--", linewidth=1.0, alpha=0.65))
        for k in range(count):
            angle = 2 * np.pi * k / count + radius
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            axis.add_patch(Circle((x, y), 0.13, facecolor=BLUE, edgecolor=DARK, linewidth=0.9))
        axis.text(center[0] + radius * 0.5, center[1] + radius * 0.62, label, fontsize=9, color=DARK, ha="center")
    draw_robot(axis, center[0], center[1], "$i$", highlighted=True)
    axis.text(4.5, 0.22, "Deux tours portent l'information jusqu'à deux sauts ; au-delà, elle est absente.", ha="center", fontsize=10, color=DARK)
    save(figure, "message-depth.png")


if __name__ == "__main__":
    swarm_local_communication()
    message_passing()
    experimental_pipeline()
    mean_aggregation()
    pilot_summary_grid()
    curriculum_losses()
    shape_pipeline()
    module_architecture()
    consensus_convergence()
    flocking_rules()
    loss_terms()
    chamfer_geometry()
    curriculum_stages()
    differentiable_rollout()
    episode_cycle()
    failure_timeline()
    threshold_convergence()
    message_depth()
