"""Images de formes et nuages de points normalisés pour l'essaim."""
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ShapeDatasetConfig:
    seed: int
    image_size: int
    point_count: int
    font_path: str
    number_start: int = 10
    number_end: int = 999

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ShapeDatasetConfig":
        with Path(path).open(encoding="utf-8") as stream:
            return cls(**yaml.safe_load(stream))


def _canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (size, size), 255)
    return image, ImageDraw.Draw(image)


def draw_geometry(name: str, size: int) -> Image.Image:
    """Dessine une forme géométrique noire sur fond blanc."""
    image, draw = _canvas(size)
    margin, width, center = size // 6, max(3, size // 18), size // 2
    left, top, right, bottom = margin, margin, size - margin, size - margin
    if name == "circle": draw.ellipse((left, top, right, bottom), outline=0, width=width)
    elif name == "square": draw.rectangle((left, top, right, bottom), outline=0, width=width)
    elif name == "triangle": draw.line([(center, top), (right, bottom), (left, bottom), (center, top)], fill=0, width=width, joint="curve")
    elif name == "line": draw.line((left, center, right, center), fill=0, width=width)
    elif name == "cross":
        draw.line((left, center, right, center), fill=0, width=width)
        draw.line((center, top, center, bottom), fill=0, width=width)
    elif name == "star":
        angles, radii = np.linspace(-np.pi / 2, 3 * np.pi / 2, 11)[:-1], np.tile((size * .36, size * .16), 5)
        points = [(center + radii[i] * np.cos(a), center + radii[i] * np.sin(a)) for i, a in enumerate(angles)]
        draw.line(points + [points[0]], fill=0, width=width, joint="curve")
    elif name == "ring": draw.ellipse((left, top, right, bottom), outline=0, width=width * 2)
    else: raise ValueError(f"Forme géométrique inconnue : {name}")
    return image


def draw_text(text: str, config: ShapeDatasetConfig) -> Image.Image:
    """Dessine un symbole ou un nombre centré avec une police épaisse."""
    image, draw = _canvas(config.image_size)
    factor = .68 if len(text) == 1 else .45 if len(text) == 2 else .34
    font = ImageFont.truetype(config.font_path, int(config.image_size * factor))
    box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
    position = ((config.image_size - (box[2] - box[0])) / 2 - box[0], (config.image_size - (box[3] - box[1])) / 2 - box[1])
    draw.text(position, text, font=font, fill=0, stroke_width=1)
    return image


def image_to_points(image: Image.Image, count: int, seed: int) -> np.ndarray:
    """Échantillonne les pixels noirs par farthest-point sampling."""
    pixels = np.argwhere(np.asarray(image) < 128)[:, ::-1].astype(float)
    if len(pixels) < count: raise ValueError("La forme ne contient pas assez de pixels.")
    rng, points, distances = np.random.default_rng(seed), np.empty((count, 2)), np.full(len(pixels), np.inf)
    selected = rng.integers(len(pixels))
    for index in range(count):
        points[index] = pixels[selected]
        distances = np.minimum(distances, ((pixels - pixels[selected]) ** 2).sum(axis=1))
        selected = int(np.argmax(distances))
    center = (np.asarray(image.size) - 1) / 2
    return (points - center) / center


def _split(category: str, label: str) -> str:
    if category == "alphabet": return "test" if label in "QXYZ" else "validation" if label in "MW" else "train"
    if category == "digits": return "test" if label in "89" else "validation" if label == "6" else "train"
    if category == "numbers": return "test" if int(label) % 10 == 9 else "validation" if int(label) % 10 == 8 else "train"
    return "train"


def _entries(config: ShapeDatasetConfig):
    for name in ("circle", "square", "triangle", "line", "cross", "star", "ring"): yield "geometry", name, draw_geometry(name, config.image_size)
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ": yield "alphabet", letter, draw_text(letter, config)
    for digit in "0123456789": yield "digits", digit, draw_text(digit, config)
    for number in range(config.number_start, config.number_end + 1): yield "numbers", f"{number:03d}", draw_text(f"{number:03d}", config)


def build_dataset(config: ShapeDatasetConfig, output_directory: str | Path) -> Path:
    """Génère les images, points, index, planche contact et archive ZIP."""
    output = Path(output_directory)
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    rows, previews = [], []
    for index, (category, label, image) in enumerate(_entries(config)):
        stem = f"{'number' if category == 'numbers' else category.rstrip('s')}_{label}"
        folder = output / category
        folder.mkdir(exist_ok=True)
        image.save(folder / f"{stem}.png")
        np.save(folder / f"{stem}.npy", image_to_points(image, config.point_count, config.seed + index))
        rows.append({"name": stem, "category": category, "split": _split(category, label), "point_count": config.point_count})
        if category == "geometry" or (category == "alphabet" and label in "ACEGIK") or (category == "digits" and label in "02468") or (category == "numbers" and label in {"010", "123", "999"}):
            previews.append(image)
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "category", "split", "point_count"])
        writer.writeheader(); writer.writerows(rows)
    sheet = Image.new("L", (6 * config.image_size, 4 * config.image_size), 255)
    for index, image in enumerate(previews): sheet.paste(image, ((index % 6) * config.image_size, (index // 6) * config.image_size))
    sheet.save(output / "contact-sheet.png")
    archive = output.with_suffix(".zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for item in output.rglob("*"):
            if item.is_file(): zipped.write(item, item.relative_to(output))
    return archive
