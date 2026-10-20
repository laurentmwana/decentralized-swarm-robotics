"""Quatre figures de données du mémoire, régénérées dans la charte `latex/images/`.

Reprend à l'identique les données et les déroulements des sorties historiques
de `latex/figures/` (supprimées), avec le rendu de la charte uniforme :
police Latin Modern (via les OTF de MiKTeX), 300 dpi, sans titre interne.

Sources, toutes versionnées :
- perte du cercle        <- results/training-circle/loss.csv
- configuration finale   <- rollout déterministe de circle-best.pt
                            (mêmes graines que src/train/visualize.py)
- pertes du curriculum   <- results/curriculum/<étape>/loss.csv
- planche curriculum     <- rollout déterministe de results/curriculum/numbers/model.pt
                            (mêmes graines que src/train/visualize_curriculum.py)

Sortie vers `latex/images/` :
- fig_4_1_perte_cercle.png
- fig_4_3_configuration_cercle.png
- fig_4_10_pertes_curriculum.png
- fig_4_11_planche_curriculum.png
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import GNNController  # noqa: E402
from src.train.training import TrainingConfig, adjacency_from_positions, local_target_features  # noqa: E402

OUTPUT = ROOT / "latex" / "images"
BLUE = "#1f5fbf"


def _setup_style() -> None:
    """Charte commune : Latin Modern si disponible, sinon DejaVu Serif."""
    family = "DejaVu Serif"
    try:
        path = subprocess.run(
            ["kpsewhich", "lmroman10-regular.otf"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        if path:
            font_manager.fontManager.addfont(path)
            family = "Latin Modern Roman"
    except Exception:  # kpsewhich absent -> police de repli
        pass
    plt.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "cm",
            "font.size": 11,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )


def _save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"{name} -> {OUTPUT / name}")


def _load_circle_model() -> tuple[TrainingConfig, GNNController]:
    config = TrainingConfig.from_yaml(ROOT / "configs" / "training-circle.yaml")
    torch.manual_seed(config.seed)
    model = GNNController(config.hidden_dim, config.message_passing_steps, 1, config.max_speed)
    state = torch.load(ROOT / "results" / "training-circle" / "circle-best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return config, model


def circle_loss() -> None:
    values = [float(v) for v in (ROOT / "results" / "training-circle" / "loss.csv").read_text().split()]
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    axis.plot(range(1, len(values) + 1), values, color=BLUE, linewidth=1.4)
    axis.set(xlabel="Itération", ylabel="Perte")
    figure.tight_layout()
    _save(figure, "fig_4_1_perte_cercle.png")


def circle_final() -> None:
    config, model = _load_circle_model()
    target = np.load(ROOT / "data" / "shapes" / "geometry" / "geometry_circle.npy") * config.target_radius
    target = target + np.asarray(config.arena_size) / 2
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
    generator = torch.Generator().manual_seed(config.seed + 1)
    positions = torch.rand((1, config.robot_count, 2), generator=generator) * torch.tensor(config.arena_size)
    velocities = torch.zeros_like(positions)
    center = torch.tensor(config.arena_size) / 2
    with torch.no_grad():
        for _ in range(config.episode_steps):
            adjacency = adjacency_from_positions(positions, config.communication_radius)
            geometry = (
                local_target_features(positions, target_tensor, config.target_radius)
                if config.use_target_geometry
                else None
            )
            velocities = model(positions, velocities, adjacency, torch.tensor([0]), positions - center, geometry)
            positions = torch.minimum(
                torch.maximum(positions + velocities * config.time_step, torch.zeros_like(center)), center * 2
            )
    figure, axis = plt.subplots(figsize=(5.4, 5.4))
    axis.grid(False)
    axis.scatter(target[:, 0], target[:, 1], color="black", marker="x", s=45, label="cible")
    axis.scatter(positions[0, :, 0], positions[0, :, 1], color=BLUE, s=45, label="robots", zorder=3)
    axis.set(xlim=(0, config.arena_size[0]), ylim=(0, config.arena_size[1]), aspect="equal")
    axis.legend(loc="upper right")
    figure.tight_layout()
    _save(figure, "fig_4_3_configuration_cercle.png")


def curriculum_losses() -> None:
    stages = [("geometry", "Géométrie"), ("digits", "Chiffres"), ("symbols", "Symboles"), ("numbers", "Nombres")]
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 5.6), sharey=True)
    for axis, (directory, label) in zip(axes.flat, stages):
        path = ROOT / "results" / "curriculum" / directory / "loss.csv"
        values = [float(v) for v in path.read_text().split()]
        axis.plot(range(1, len(values) + 1), values, color=BLUE, linewidth=1.4)
        axis.set(title=label, xlabel="Itération")
    axes[0, 0].set_ylabel("Perte")
    axes[1, 0].set_ylabel("Perte")
    figure.tight_layout()
    _save(figure, "fig_4_10_pertes_curriculum.png")


def _french_label(stem: str) -> str:
    geometry = {"circle": "cercle", "square": "carré", "triangle": "triangle"}
    if stem.startswith("geometry_"):
        return geometry[stem.split("_", 1)[1]]
    if stem.startswith("digit_"):
        return stem.split("_", 1)[1]
    if stem.startswith("alphabet_"):
        return stem.split("_", 1)[1]
    if stem.startswith("number_"):
        return stem.split("_", 1)[1]
    return stem


def curriculum_sheet() -> None:
    with (ROOT / "configs" / "curriculum.yaml").open(encoding="utf-8") as stream:
        curriculum = yaml.safe_load(stream)
    stage = next(item for item in curriculum["stages"] if item["name"] == "numbers")
    base = dict(curriculum["base"])
    base["arena_size"] = tuple(base["arena_size"])
    config = TrainingConfig(iterations=1, **base)
    target_paths = stage["targets"]
    targets = [np.load(p) * config.target_radius + np.asarray(config.arena_size) / 2 for p in target_paths]
    torch.manual_seed(config.seed)
    model = GNNController(config.hidden_dim, config.message_passing_steps, len(targets), config.max_speed)
    state = torch.load(ROOT / "results" / "curriculum" / "numbers" / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 99)
    columns = 4
    rows = (len(targets) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.1 * rows))
    for target_id, (axis, target, path) in enumerate(zip(axes.flat, targets, target_paths)):
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        positions = torch.rand((1, config.robot_count, 2), generator=generator) * torch.tensor(config.arena_size)
        velocities = torch.zeros_like(positions)
        center = torch.tensor(config.arena_size) / 2
        with torch.no_grad():
            for _ in range(config.episode_steps):
                adjacency = adjacency_from_positions(positions, config.communication_radius)
                geometry = (
                    local_target_features(positions, target_tensor, config.target_radius)
                    if config.use_target_geometry
                    else None
                )
                velocities = model(
                    positions, velocities, adjacency, torch.tensor([target_id]), positions - center, geometry
                )
                positions = torch.minimum(
                    torch.maximum(positions + velocities * config.time_step, torch.zeros_like(center)), center * 2
                )
        axis.grid(False)
        axis.scatter(target[:, 0], target[:, 1], color="black", marker="x", s=25)
        axis.scatter(positions[0, :, 0], positions[0, :, 1], color=BLUE, s=25, zorder=3)
        axis.set(
            title=_french_label(Path(path).stem),
            xlim=(0, 10),
            ylim=(0, 10),
            aspect="equal",
        )
    for axis in axes.flat[len(targets) :]:
        axis.axis("off")
    figure.tight_layout()
    _save(figure, "fig_4_11_planche_curriculum.png")


def main() -> None:
    _setup_style()
    circle_loss()
    circle_final()
    curriculum_losses()
    curriculum_sheet()


if __name__ == "__main__":
    main()
