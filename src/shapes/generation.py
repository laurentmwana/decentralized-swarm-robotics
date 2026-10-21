"""Génération des livrables de formes : outputs/shapes/{index.csv, images/, point_clouds/, sheet_<type>.png}.

Usage : `python -m src.shapes.generation`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.config import ConfigLoader

from .index import generate_index

SHEET_RESOLUTION = 50


def contact_sheets(index, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    for shape_type in ("geometries", "digits", "letters", "numbers"):
        rows = [l for l in index.rows if l["type"] == shape_type and l["resolution"] == SHEET_RESOLUTION]
        if not rows:
            continue
        n = len(rows)
        ncol = 6
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.4 * nrow), squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, row in zip(axes.ravel(), rows):
            points = index.point_cloud_by_key(f"{row['type']}_{row['name']}@{SHEET_RESOLUTION}")
            axis.axis("equal")
            axis.scatter(points[:, 0], points[:, 1], s=8)
            axis.set_title(f"{row['name']} ({row['split']})", fontsize=9)
        fig.suptitle(f"Planche de controle — {shape_type} (resolution {SHEET_RESOLUTION})")
        fig.tight_layout()
        path = output / f"sheet_{shape_type}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def main(output: Path | str = "outputs/shapes") -> dict:
    output = Path(output)
    cfg = ConfigLoader().load_shapes()
    index, split = generate_index(cfg, output)
    written = contact_sheets(index, output)
    summary = {
        "shape_count": len({(l["type"], l["name"]) for l in index.rows}),
        "index_row_count": len(index.rows),
        "shapes_per_split": split.count(),
        "contact_sheets": [str(p) for p in written],
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
