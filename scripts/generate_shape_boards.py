"""Planches des formes du jeu standard (20 et 50 points) pour le mémoire.

Reconstruit les deux planches à partir de `data/shapes` : sept formes
géométriques, les 26 lettres, les dix chiffres et deux nombres à trois
caractères du curriculum. Les nuages à 20 points proviennent des fichiers
versionnés ; les nuages à 50 points sont ré-échantillonnés par
farthest-point sampling à partir des masques, avec la graine de
configuration. Sortie déterministe vers `latex/images/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shapes.generator import ShapeDatasetConfig, image_to_points  # noqa: E402

DATA = ROOT / "data" / "shapes"
OUTPUT = ROOT / "latex" / "images"
CONFIG = ShapeDatasetConfig.from_yaml(ROOT / "configs" / "shapes.yaml")

GEOMETRY_LABELS = {
    "circle": "cercle",
    "square": "carré",
    "triangle": "triangle",
    "line": "ligne",
    "cross": "croix",
    "star": "étoile",
    "ring": "anneau",
}


def _entries() -> list[tuple[str, Path, Path]]:
    """ liste (étiquette, fichier de points 20, masque image) pour les 45 formes. """
    items: list[tuple[str, Path, Path]] = []
    for name, label in GEOMETRY_LABELS.items():
        items.append((label, DATA / "geometry" / f"geometry_{name}.npy", DATA / "geometry" / f"geometry_{name}.png"))
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        items.append((letter, DATA / "alphabet" / f"alphabet_{letter}.npy", DATA / "alphabet" / f"alphabet_{letter}.png"))
    for digit in "0123456789":
        items.append((digit, DATA / "digits" / f"digit_{digit}.npy", DATA / "digits" / f"digit_{digit}.png"))
    for number in ("010", "123"):
        items.append((number, DATA / "numbers" / f"number_{number}.npy", DATA / "numbers" / f"number_{number}.png"))
    return items


def _points(label: str, cloud_path: Path, mask_path: Path, count: int) -> np.ndarray:
    if count == 20:
        return np.load(cloud_path)
    return image_to_points(Image.open(mask_path), count, CONFIG.seed)


def _board(count: int) -> None:
    entries = _entries()
    columns, rows = 9, 5
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.15, rows * 1.35))
    for axis, (label, cloud_path, mask_path) in zip(axes.flat, entries):
        points = _points(label, cloud_path, mask_path, count)
        axis.scatter(points[:, 0], -points[:, 1], s=7, color="#1f5fbf")
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#9aa5b8")
            spine.set_linewidth(0.8)
        axis.set_xlabel(label, fontsize=8, labelpad=2)
        axis.xaxis.set_label_position("top")
    for axis in axes.flat[len(entries):]:
        axis.axis("off")
    figure.subplots_adjust(wspace=0.12, hspace=0.35)
    figure.savefig(OUTPUT / f"fig_extra_planche_formes_{count}pts.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for count in (20, 50):
        _board(count)
        print(f"planche {count} points -> {OUTPUT / f'fig_extra_planche_formes_{count}pts.png'}")


if __name__ == "__main__":
    main()
