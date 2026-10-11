import numpy as np
from pathlib import Path

from src.shapes.generator import ShapeDatasetConfig, build_dataset, draw_geometry, image_to_points


def test_geometric_points_are_normalized_and_reproducible() -> None:
    image = draw_geometry("circle", 64)
    first = image_to_points(image, count=12, seed=4)
    assert first.shape == (12, 2)
    assert np.all(np.abs(first) <= 1.0)
    assert np.array_equal(first, image_to_points(image, count=12, seed=4))


def test_dataset_contains_images_points_index_and_archive() -> None:
    config = ShapeDatasetConfig(3, 48, 8, "C:/Windows/Fonts/arialbd.ttf", 10, 11)
    output = Path("results/test-shapes")
    archive = build_dataset(config, output)
    assert archive.is_file()
    assert (output / "index.csv").is_file()
    assert len(list(output.rglob("*.png"))) >= 44
    assert len(list(output.rglob("*.npy"))) == 45
