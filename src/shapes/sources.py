"""Sources de formes : rendu binaire 128x128 (docs/decisions/04-shapes.md).

`ShapeSource` est le contrat abstrait ; chaque type de forme est une sous-classe qui ne
connaît que son rendu, jamais l'échantillonnage ni la normalisation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont


class ShapeSource(ABC):
    def __init__(self, name: str, shape_type: str) -> None:
        self.name = name
        self.shape_type = shape_type

    @property
    def identifier(self) -> str:
        return f"{self.shape_type}_{self.name}"

    @abstractmethod
    def render(self, size: int) -> np.ndarray:
        """Image en niveaux de gris (size, size), float 0..255."""

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Masque d'encre au milieu de la plage de gris observée (jamais un seuil fixe)."""
        threshold = (image.min() + image.max()) / 2.0
        return image < threshold


def _grayscale_figure(size: int, draw) -> np.ndarray:
    import io

    fig = Figure(figsize=(size / 100.0, size / 100.0), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    draw(ax)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt_close(fig)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L"), dtype=np.float64)


def plt_close(fig) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


class GeometryShape(ShapeSource):
    """Géométries tracées en contour (les robots doivent suivre un contour, pas un disque)."""

    _RADIUS = 0.9

    def __init__(self, name: str) -> None:
        super().__init__(name, "geometries")

    def _regular_polygon(self, n: int, radius: float, rotation: float = np.pi / 2):
        angles = rotation + np.arange(n) * 2.0 * np.pi / n
        return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    def _star(self, points: int = 5, outer_radius: float = 0.95, inner_radius: float = 0.42):
        vertices = []
        for k in range(2 * points):
            radius = outer_radius if k % 2 == 0 else inner_radius
            angle = np.pi / 2 + k * np.pi / points
            vertices.append([radius * np.cos(angle), radius * np.sin(angle)])
        return np.array(vertices)

    def _cross(self, arm_length: float = 0.9, half_width: float = 0.28):
        b, w = arm_length, half_width
        return np.array(
            [
                [-w, b], [w, b], [w, w], [b, w], [b, -w], [w, -w],
                [w, -b], [-w, -b], [-w, -w], [-b, -w], [-b, w], [-w, w],
            ]
        )

    def render(self, size: int) -> np.ndarray:
        from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

        if self.name == "circle":
            draw_shape = lambda ax: ax.add_patch(Circle((0, 0), self._RADIUS, fill=False, lw=2))
        elif self.name == "ellipse":
            draw_shape = lambda ax: ax.add_patch(Ellipse((0, 0), 1.9, 1.1, fill=False, lw=2))
        elif self.name == "square":
            draw_shape = lambda ax: ax.add_patch(Rectangle((-0.75, -0.75), 1.5, 1.5, fill=False, lw=2))
        elif self.name == "triangle":
            draw_shape = lambda ax: ax.add_patch(Polygon(self._regular_polygon(3, 0.95), fill=False, lw=2))
        elif self.name == "pentagon":
            draw_shape = lambda ax: ax.add_patch(Polygon(self._regular_polygon(5, 0.95), fill=False, lw=2))
        elif self.name == "star":
            draw_shape = lambda ax: ax.add_patch(Polygon(self._star(), fill=False, lw=2))
        elif self.name == "cross":
            draw_shape = lambda ax: ax.add_patch(Polygon(self._cross(), fill=False, lw=2))
        else:
            raise ValueError(f"géométrie inconnue : {self.name!r}")
        return _grayscale_figure(size, draw_shape)


class GlyphShape(ShapeSource):
    """Caractères rendus par police (lettres et chiffres)."""

    def __init__(self, character: str, font_path: str) -> None:
        super().__init__(str(character), "letters" if len(str(character)) == 1 and str(character).isalpha() else "digits")
        self.character = str(character)
        self.font_path = font_path

    def render(self, size: int) -> np.ndarray:
        return _render_text(self.character, size, self.font_path)


class NumberShape(ShapeSource):
    """Nombres (plusieurs chiffres), même moteur de rendu que les glyphes."""

    def __init__(self, value: int, font_path: str) -> None:
        super().__init__(str(value), "numbers")
        self.value = int(value)
        self.font_path = font_path

    def render(self, size: int) -> np.ndarray:
        return _render_text(str(self.value), size, self.font_path)


def _render_text(text: str, size: int, font_path: str) -> np.ndarray:
    margin = max(2, size // 16)
    img = Image.new("L", (size + 2 * margin, size + 2 * margin), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError as exc:
        raise RuntimeError(f"police introuvable pour le rendu des glyphes : {font_path!r}") from exc
    draw.text((margin, margin), text, font=font, fill=0)
    bbox = img.getbbox()
    if bbox is None:
        raise ValueError(f"glyphe rendu vide : {text!r}")
    crop = img.crop(bbox)
    ratio = size / max(crop.size)
    result = crop.resize((max(1, int(crop.width * ratio)), max(1, int(crop.height * ratio))), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    corner = ((size - result.width) // 2, (size - result.height) // 2)
    canvas.paste(result, corner)
    return np.asarray(canvas, dtype=np.float64)
