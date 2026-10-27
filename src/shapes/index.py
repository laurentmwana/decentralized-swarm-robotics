"""Index des formes et découpage verrouillé (docs/decisions/04-shapes.md)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.common.config import ShapesConfig
from src.common.seeds import SeedManager

from .sampling import ShapeSampler
from .sources import GeometryShape, GlyphShape, NumberShape, ShapeSource

SPLITS = ("train", "val", "test")


def build_sources(cfg: ShapesConfig) -> list[ShapeSource]:
    """Une source par forme déclarée, dans l'ordre stable geometries -> digits -> letters -> numbers."""
    sources: list[ShapeSource] = []
    for name in cfg.geometries:
        sources.append(GeometryShape(name))
    for digit in cfg.digits:
        sources.append(GlyphShape(str(digit), cfg.font_path))
    for letter in cfg.letters:
        sources.append(GlyphShape(letter, cfg.font_path))
    for number in cfg.numbers:
        sources.append(NumberShape(number, cfg.font_path))
    return sources


@dataclass(frozen=True)
class ShapeSplit:
    mapping: dict[str, str]  # identifiant de forme -> split

    def split_of(self, source: ShapeSource) -> str:
        try:
            return self.mapping[source.identifier]
        except KeyError as exc:
            raise ValueError(f"forme absente du découpage figé : {source.identifier}") from exc

    def validate_request(self, source: ShapeSource, requested_split: str) -> None:
        """Refuse d'exposer une forme d'un split non demandé (protection du jeu de test)."""
        actual = self.split_of(source)
        if actual != requested_split:
            raise ValueError(
                f"forme {source.identifier!r} est du split {actual!r}, pas {requested_split!r} : accès refusé"
            )

    def count(self) -> dict[str, int]:
        counts = {s: 0 for s in SPLITS}
        for split in self.mapping.values():
            counts[split] += 1
        return counts


def build_split(cfg: ShapesConfig) -> ShapeSplit:
    mapping: dict[str, str] = {}
    for kind, parts in cfg.splits.items():
        for split, names in parts.items():
            for name in names:
                key = f"{kind}_{name}"
                if key in mapping:
                    raise ValueError(f"forme dans plusieurs splits : {key}")
                mapping[key] = split
    return ShapeSplit(mapping=mapping)


@dataclass
class ShapeIndex:
    """Index en mémoire + écriture de outputs/shapes/index.csv (une ligne par figure).

    L'identifiant de forme (`shape_id`) est l'index de la forme dans l'ordre stable de
    `build_sources` (geometries -> digits -> letters -> numbers) : partagé par le
    GNN à l'entraînement et à l'évaluation, reproductible tant que config/shapes.yaml est stable.
    """

    rows: list[dict[str, object]]
    point_clouds: dict[str, np.ndarray]  # identifiant de figure (letters_A) -> points de référence normalisés
    identifiers: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, directory: Path) -> "ShapeIndex":
        with (directory / "index.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        identifiers = {}
        clouds = {}
        for row in rows:
            if "resolution" in row:
                raise ValueError("catalogue de formes d'une ancienne version (une ligne par résolution) : relancer make shapes")
            for column in ("shape_id", "n_points"):
                row[column] = int(row[column])
            identifier = f"{row['type']}_{row['name']}"
            if identifier in clouds:
                raise ValueError(f"figure dupliquée dans l'index : {identifier}")
            identifiers[row["shape_id"]] = identifier
            clouds[identifier] = np.load(directory / "point_clouds" / f"{identifier}.npy", allow_pickle=False)
        if sorted(identifiers) != list(range(len(identifiers))):
            raise ValueError("identifiants de formes non contigus")
        return cls(rows, clouds, [identifiers[index] for index in range(len(identifiers))])

    def shape_id(self, key: str) -> int:
        try:
            return self.identifiers.index(key)
        except ValueError as exc:
            raise KeyError(f"forme absente de l'index : {key}") from exc

    def point_cloud(self, source: ShapeSource) -> np.ndarray:
        return self.point_cloud_by_key(source.identifier)

    def point_cloud_by_key(self, key: str) -> np.ndarray:
        try:
            return self.point_clouds[key]
        except KeyError as exc:
            raise KeyError(f"forme absente de l'index : {key}") from exc

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            raise ValueError("index vide")
        columns = ["name", "type", "shape_id", "split", "image_path", "n_points"]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)


def generate_index(
    cfg: ShapesConfig,
    output: Path,
) -> tuple[ShapeIndex, ShapeSplit]:
    """Rend, échantillonne et normalise toutes les formes ; écrit images + index.csv + nuages .npy.

    Une figure = un nuage de `cfg.reference_points` points de référence, servant à mesurer la
    distance essaim -> figure ; il n'existe pas plusieurs résolutions d'une même figure.
    """
    split = build_split(cfg)
    sampler = ShapeSampler(SeedManager(cfg.sampler_seed))
    index = ShapeIndex(rows=[], point_clouds={})
    image_root = output / "images"
    point_cloud_root = output / "point_clouds"
    image_root.mkdir(parents=True, exist_ok=True)
    point_cloud_root.mkdir(parents=True, exist_ok=True)

    for source in build_sources(cfg):
        index.identifiers.append(source.identifier)
        shape_identifier = len(index.identifiers) - 1
        image = source.render(cfg.render_size)
        mask = source.binarize(image)
        image_path = image_root / f"{source.identifier}.png"
        _write_png(str(image_path), mask.astype(np.uint8) * 255)
        image_points = sampler.sample(mask, cfg.reference_points)
        normalized = ShapeSampler.normalize(image_points, cfg.target_radius)
        index.point_clouds[source.identifier] = normalized
        np.save(point_cloud_root / f"{source.identifier}.npy", normalized)
        index.rows.append(
            {
                "name": source.name,
                "type": source.shape_type,
                "shape_id": shape_identifier,
                "split": split.split_of(source),
                "image_path": image_path.as_posix(),
                "n_points": len(normalized),
            }
        )
    index.write_csv(output / "index.csv")
    return index, split


def _write_png(path: str, grid: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(grid.astype(np.uint8), mode="L").save(path)
