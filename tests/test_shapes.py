"""Tests obligatoires de la phase 1 — formes (PLAN, section Tests), dont le test anti-doublon."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.config import ConfigLoader
from src.common.seeds import SeedManager
from src.shapes import ShapeSampler, generate_index
from src.shapes.index import build_sources, build_split


@pytest.fixture(scope="module")
def cfg_shapes():
    return ConfigLoader().load_shapes()


@pytest.fixture(scope="module")
def index_and_split(cfg_shapes, tmp_path_factory):
    output = tmp_path_factory.mktemp("shapes")
    return generate_index(cfg_shapes, output)


class TestNoDuplicates:
    def test_distinct_points_match_reference_points(self, cfg_shapes, index_and_split):
        index, _ = index_and_split
        assert len(index.rows) == len(index.identifiers)  # une ligne par figure, aucune résolution
        for row in index.rows:
            point_cloud = index.point_cloud_by_key(f"{row['type']}_{row['name']}")
            distinct = np.unique(np.round(point_cloud, 12), axis=0)
            assert len(distinct) == cfg_shapes.reference_points, (
                f"doublons sur {row['type']}_{row['name']} : {len(distinct)} != {cfg_shapes.reference_points}"
            )

    def test_reject_coarse_render(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2, 3] = True
        mask[5, 6] = True
        sampler = ShapeSampler(SeedManager(1))
        with pytest.raises(ValueError):
            sampler.sample(mask, 20)

    def test_sampling_reproducibility(self, cfg_shapes):
        sources = build_sources(cfg_shapes)
        circle = next(s for s in sources if s.name == "circle")
        mask = circle.binarize(circle.render(cfg_shapes.render_size))
        a = ShapeSampler(SeedManager(cfg_shapes.sampler_seed)).sample(mask, 50)
        b = ShapeSampler(SeedManager(cfg_shapes.sampler_seed)).sample(mask, 50)
        assert np.array_equal(a, b)


class TestNormalization:
    def test_all_shapes_in_common_frame(self, cfg_shapes, index_and_split):
        index, _ = index_and_split
        for row in index.rows:
            point_cloud = index.point_cloud_by_key(f"{row['type']}_{row['name']}")
            assert np.abs(point_cloud).max() <= 1.0 + 1e-9
            radius = np.linalg.norm(point_cloud, axis=1).max()
            assert radius == pytest.approx(cfg_shapes.target_radius, abs=1e-9)

    def test_centered_centroid(self, index_and_split):
        index, _ = index_and_split
        point_cloud = index.point_cloud_by_key("geometries_circle")
        assert np.linalg.norm(point_cloud.mean(axis=0)) < 1e-9


class TestIndex:
    def test_every_shape_present(self, cfg_shapes, index_and_split):
        index, _ = index_and_split
        seen = {(l["type"], str(l["name"])) for l in index.rows}
        expected = (
            {("geometries", g) for g in cfg_shapes.geometries}
            | {("digits", str(d)) for d in cfg_shapes.digits}
            | {("letters", l) for l in cfg_shapes.letters}
            | {("numbers", str(n)) for n in cfg_shapes.numbers}
        )
        assert expected <= seen

    def test_every_shape_has_split(self, cfg_shapes, index_and_split):
        _, split = index_and_split
        counts = split.count()
        assert all(counts[s] > 0 for s in ("train", "val", "test"))

    def test_reject_access_outside_requested_split(self, cfg_shapes):
        split = build_split(cfg_shapes)
        sources = {s.identifier: s for s in build_sources(cfg_shapes)}
        star = sources["geometries_star"]  # réservée au test
        assert split.split_of(star) == "test"
        with pytest.raises(ValueError):
            split.validate_request(star, "train")


class TestBinarization:
    def test_ink_present_background_absent(self, cfg_shapes):
        sources = build_sources(cfg_shapes)
        for source in sources:
            image = source.render(cfg_shapes.render_size)
            mask = source.binarize(image)
            assert mask.any(), f"aucune encre pour {source.identifier}"
            assert (~mask).any(), f"image entièrement pleine pour {source.identifier}"


class TestGlyphRendering:
    @pytest.mark.parametrize("value", [12, 314, 999])
    def test_numbers_are_not_clipped(self, value):
        # Régression : « 314 » était rendu « 31 » (texte plus large que le canevas).
        from src.shapes.sources import NumberShape

        shape = NumberShape(value, ConfigLoader().load_shapes().font_path)
        mask = shape.binarize(shape.render(128))
        columns = np.flatnonzero(mask.any(axis=0))
        # Le texte cadré occupe toute la largeur, avec une marge blanche à gauche et à droite.
        assert columns.min() <= 2 and columns.max() >= 125
        assert not mask[:, 0].all() and not mask[:, -1].all()
        # Autant de groupes de colonnes encrées que de chiffres (chiffres séparés par un blanc).
        groups = 1 + int((np.diff(columns) > 1).sum())
        assert groups == len(str(value))
