"""Couverture du catalogue complet ; aucune commande de robots ni optimisation."""
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.reference.catalog import LETTERS, NUMBERS, GEOMETRIES, catalog_entries, export_catalog
from src.reference.shapes import named_shape


def test_catalog_has_complete_requested_scope_without_duplicate_identifiers():
    entries = catalog_entries()
    assert LETTERS == tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    assert NUMBERS == tuple(str(i) for i in range(1, 11))
    assert len(GEOMETRIES) >= 15
    assert {'square', 'rectangle', 'circle', 'ellipse', 'triangle', 'star', 'cross', 'arrow'} <= set(GEOMETRIES)
    assert len({e['identifier'] for e in entries}) == len(entries) == 56


@pytest.mark.parametrize('identifier', [e['identifier'] for e in catalog_entries()])
@pytest.mark.parametrize('count', [12, 24])
def test_every_catalog_entry_has_exact_distinct_on_shape_points(identifier, count):
    mission = named_shape(identifier, count)
    assert mission.identifier == identifier
    points = mission.reference_points()
    assert points.shape == (count, 2)
    assert np.isfinite(points).all()
    assert len(np.unique(np.round(points, 10), axis=0)) == count
    assert max(mission.project(point)[1].distance for point in points) < 1e-10
    np.testing.assert_array_equal(points, named_shape(identifier, count).reference_points())
    vertices = np.concatenate(mission.strokes)
    assert (np.abs(vertices) <= 3+1e-12).all()


def test_ten_is_two_digits_with_both_components_present():
    mission = named_shape(10, 24)
    assert mission.identifier == 'numbers_10'
    assert len(mission.strokes) == 3
    assert mission.closed == (False, False, True)
    points = mission.reference_points()
    assert (points[:, 0] < 0).any() and (points[:, 0] > 0).any()


def test_letter_and_geometry_v_have_explicit_categories():
    assert named_shape('V', 12).identifier == 'letters_V'
    assert named_shape('geometries_v', 12).identifier == 'geometries_v'
    assert named_shape('carre', 12).identifier == 'geometries_square'
    assert named_shape('cœur', 12).identifier == 'geometries_heart'


@pytest.mark.parametrize('name', ['0', '11', 'AA', 'letters_10', 'numbers_A', 'geometries_A'])
def test_names_outside_the_explicit_catalog_are_rejected(name):
    with pytest.raises(ValueError, match='inconnue'):
        named_shape(name, 12)


def test_export_is_complete_and_records_physical_infeasibility(tmp_path):
    folder = tmp_path/'catalog'
    result = export_catalog(folder, 24)
    assert result['counts'] == dict(letters=26, numbers=10, geometries=20)
    assert json.loads((folder/'catalog.json').read_text(encoding='utf-8')) == result
    assert len(result['shapes']) == 56
    line = next(s for s in result['shapes'] if s['identifier'] == 'geometries_line')
    assert line['feasible'] is False
    assert line['infeasibility_reason']
    root = ET.parse(folder/'catalog.svg').getroot()
    ns = {'s': 'http://www.w3.org/2000/svg'}
    assert len(root.findall('s:g', ns)) == 56
    assert len(root.findall('.//s:circle', ns)) == 56*24
    with pytest.raises(FileExistsError):
        export_catalog(folder, 24)
