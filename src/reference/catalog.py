"""Catalogue vectoriel métrique : A–Z, 1–10 et vingt géométries."""
import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


LETTERS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
NUMBERS = tuple(str(i) for i in range(1, 11))
GEOMETRIES = ('circle', 'ellipse', 'square', 'rectangle', 'triangle', 'pentagon',
              'hexagon', 'heptagon', 'octagon', 'decagon', 'diamond', 'trapezoid',
              'parallelogram', 'star', 'cross', 'arrow', 'semicircle', 'heart', 'line', 'v')
ALIASES = {'ligne': 'line', 'carre': 'square', 'carré': 'square', 'cercle': 'circle',
           'losange': 'diamond', 'trapeze': 'trapezoid', 'trapèze': 'trapezoid',
           'parallelogramme': 'parallelogram', 'etoile': 'star', 'étoile': 'star',
           'croix': 'cross', 'fleche': 'arrow', 'flèche': 'arrow', 'coeur': 'heart',
           'cœur': 'heart', 'demi-cercle': 'semicircle', 'pentagone': 'pentagon',
           'hexagone': 'hexagon', 'heptagone': 'heptagon', 'octogone': 'octagon', 'decagone': 'decagon'}


def catalog_entries():
    return tuple(dict(identifier=f'{category}_{name}', category=category, name=name)
                 for category, names in (('letters', LETTERS), ('numbers', NUMBERS), ('geometries', GEOMETRIES))
                 for name in names)


def arc(center=(0., 0.), radii=(.5, .5), start=0., stop=2*math.pi, count=64, closed=False):
    angles = np.linspace(start, stop, count, endpoint=not closed)
    points = np.asarray(center)+np.column_stack((np.cos(angles), np.sin(angles)))*radii
    points[np.abs(points) < 1e-14] = 0.
    return points


def _letter(name):
    left = [(-.5, -.5), (-.5, .5)]
    right = [(.5, -.5), (.5, .5)]
    top = [(-.5, .5), (.5, .5)]
    middle = [(-.5, 0), (.5, 0)]
    bottom = [(-.5, -.5), (.5, -.5)]
    upper = arc((-.5, .25), (1., .25), math.pi/2, -math.pi/2, 33)
    lower = arc((-.5, -.25), (1., .25), math.pi/2, -math.pi/2, 33)
    c = arc(start=math.pi/4, stop=7*math.pi/4, count=65)
    definitions = {
        'A': [[(-.5, -.5), (0, .5), (.5, -.5)], [(-.25, 0), (.25, 0)]],
        'B': [left, upper, lower], 'C': [c],
        'D': [left, arc((-.5, 0), (1., .5), math.pi/2, -math.pi/2, 65)],
        'E': [left, top, middle, bottom], 'F': [left, top, middle],
        'G': [c, [tuple(c[-1]), (.5, -.25), (.5, 0), (0, 0)]],
        'H': [left, right, middle], 'I': [top, [(0, .5), (0, -.5)], bottom],
        'J': [top, [(.5, .5), (.5, -.25)], arc((0, -.25), (.5, .25), 0, -math.pi, 33)],
        'K': [left, [(.5, .5), (-.5, 0), (.5, -.5)]],
        'L': [[(-.5, .5), (-.5, -.5), (.5, -.5)]],
        'M': [[(-.5, -.5), (-.5, .5), (0, 0), (.5, .5), (.5, -.5)]],
        'N': [[(-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)]],
        'O': [arc(closed=True)], 'P': [left, upper],
        'Q': [arc(closed=True), [(.1, -.2), (.5, -.5)]],
        'R': [left, upper, [(-.5, 0), (.5, -.5)]],
        'S': [[(.5, .5), (-.25, .5), (-.5, .25), (-.25, 0), (.25, 0), (.5, -.25), (.25, -.5), (-.5, -.5)]],
        'T': [top, [(0, .5), (0, -.5)]],
        'U': [[(-.5, .5), (-.5, -.25)], arc((0, -.25), (.5, .25), math.pi, 2*math.pi, 33), [(.5, -.25), (.5, .5)]],
        'V': [[(-.5, .5), (0, -.5), (.5, .5)]],
        'W': [[(-.5, .5), (-.25, -.5), (0, 0), (.25, -.5), (.5, .5)]],
        'X': [[(-.5, .5), (.5, -.5)], [(-.5, -.5), (.5, .5)]],
        'Y': [[(-.5, .5), (0, 0), (.5, .5)], [(0, 0), (0, -.5)]],
        'Z': [[(-.5, .5), (.5, .5), (-.5, -.5), (.5, -.5)]],
    }
    strokes = definitions[name]
    return strokes, tuple(name in ('O', 'Q') and i == 0 for i in range(len(strokes)))


def _number(name):
    if name == '10':
        strokes, flags = [], []
        for digit, shift in (('1', -.3), ('0', .3)):
            paths, closed = _number(digit)
            strokes.extend(np.asarray(p)*[.4, 1.]+[shift, 0.] for p in paths)
            flags.extend(closed)
        return strokes, tuple(flags)
    if name == '0': return [arc(radii=(.45, .5), closed=True)], (True,)
    if name == '1': return [[(-.25, .25), (0, .5), (0, -.5)], [(-.25, -.5), (.25, -.5)]], (False, False)
    if name == '3':
        return [arc((-.5, .25), (1, .25), math.pi/2, -math.pi/2, 33),
                arc((-.5, -.25), (1, .25), math.pi/2, -math.pi/2, 33)], (False, False)
    if name == '4': return [[(.3, -.5), (.3, .5)], [(.3, .5), (-.5, -.1), (.5, -.1)]], (False, False)
    if name == '6': return [arc((0, -.2), (.5, .3), closed=True), [(-.5, -.2), (-.5, .1), (-.25, .5), (.5, .5)]], (True, False)
    if name == '8': return [arc((0, .25), (.5, .25), closed=True), arc((0, -.25), (.5, .25), closed=True)], (True, True)
    if name == '9': return [arc((0, .2), (.5, .3), closed=True), [(.5, .2), (.5, -.5)]], (True, False)
    paths = {
        '2': [(-.5, .25), (-.25, .5), (.25, .5), (.5, .25), (.5, 0), (-.5, -.5), (.5, -.5)],
        '5': [(.5, .5), (-.5, .5), (-.5, 0), (.25, 0), (.5, -.25), (.25, -.5), (-.5, -.5)],
        '7': [(-.5, .5), (.5, .5), (-.25, -.5)],
    }
    return [paths[name]], (False,)


def _geometry(name):
    if name == 'line': return [[(-.5, 0), (.5, 0)]], (False,)
    if name == 'v': return _letter('V')
    if name in ('circle', 'ellipse'):
        return [arc(radii=(.5, .5 if name == 'circle' else .3), closed=True)], (True,)
    polygons = {'triangle': 3, 'pentagon': 5, 'hexagon': 6, 'heptagon': 7, 'octagon': 8, 'decagon': 10}
    if name in polygons:
        angles = math.pi/2+np.arange(polygons[name])*2*math.pi/polygons[name]
        return [.5*np.column_stack((np.cos(angles), np.sin(angles)))], (True,)
    if name == 'star':
        angles = math.pi/2+np.arange(10)*math.pi/5
        radii = np.where(np.arange(10)%2 == 0, .5, .22)
        return [radii[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))], (True,)
    if name == 'semicircle': return [arc(start=0, stop=math.pi, count=65)], (True,)
    if name == 'heart':
        t = np.linspace(0, 2*math.pi, 128, endpoint=False)
        x = .5*np.sin(t)**3
        y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
        y = (y-y.min())/np.ptp(y)-.5
        return [np.column_stack((x, y))], (True,)
    paths = {
        'square': [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)],
        'rectangle': [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)],
        'diamond': [(0, .5), (.5, 0), (0, -.5), (-.5, 0)],
        'trapezoid': [(-.5, -.5), (.5, -.5), (.3, .5), (-.3, .5)],
        'parallelogram': [(-.5, -.5), (.3, -.5), (.5, .5), (-.3, .5)],
        'cross': [(-.17, .5), (.17, .5), (.17, .17), (.5, .17), (.5, -.17), (.17, -.17),
                  (.17, -.5), (-.17, -.5), (-.17, -.17), (-.5, -.17), (-.5, .17), (-.17, .17)],
        'arrow': [(.5, 0), (.1, .5), (.1, .18), (-.5, .18), (-.5, -.18), (.1, -.18), (.1, -.5)],
    }
    return [paths[name]], (True,)


def shape_definition(name):
    token = str(name).strip().lower()
    category = None
    for prefix in ('letters_', 'numbers_', 'geometries_'):
        if token.startswith(prefix):
            category, token = prefix[:-1], token[len(prefix):]
            break
    token = ALIASES.get(token, token)
    if category is None:
        category = 'letters' if token.upper() in LETTERS else 'numbers' if token in NUMBERS else 'geometries'
    if category == 'letters' and token.upper() in LETTERS:
        strokes, closed = _letter(token.upper())
    elif category == 'numbers' and token in NUMBERS:
        strokes, closed = _number(token)
    elif category == 'geometries' and token in GEOMETRIES:
        strokes, closed = _geometry(token)
    else:
        raise ValueError(f'Forme inconnue : {name}')
    return category, token, strokes, closed


def export_catalog(output, robot_count=24):
    """Exporter les consignes et une planche SVG native, sans entraînement."""
    from .shapes import named_shape
    from src.simulator.acceleration_episode import AccelerationEpisodeConfig
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    records, panels = [], []
    entries = catalog_entries()
    for index, entry in enumerate(entries):
        # Un rectangle par défaut est distinct du carré dans la planche.
        height = 4. if entry['identifier'] == 'geometries_rectangle' else 6.
        mission = named_shape(entry['identifier'], robot_count, width=6., height=height)
        points = mission.reference_points()
        feasible, reason = True, None
        try:
            mission.validate_environment(config, .4)
        except ValueError as error:
            feasible, reason = False, str(error)
        records.append(dict(**entry, robot_count=robot_count,
                            strokes=[[list(point) for point in stroke] for stroke in mission.strokes], closed=list(mission.closed),
                            reference_points=points.tolist(), feasible=feasible, infeasibility_reason=reason))
        x, y = (index%7)*200, (index//7)*190
        panel = [f'<g transform="translate({x},{y})"><text x="100" y="20" text-anchor="middle" font-size="14">{escape(entry["identifier"])}</text>']
        for stroke, closed in zip(mission.strokes, mission.closed):
            coordinates = ' '.join(f'{100+px*20:.3f},{100-py*20:.3f}' for px, py in stroke)
            tag = 'polygon' if closed else 'polyline'
            panel.append(f'<{tag} points="{coordinates}" fill="none" stroke="#334155" stroke-width="2"/>')
        panel.extend(f'<circle cx="{100+px*20:.3f}" cy="{100-py*20:.3f}" r="2.5" fill="#0d9488"/>' for px, py in points)
        panel.append(f'<text x="100" y="180" text-anchor="middle" font-size="11" fill="{("#166534" if feasible else "#9a3412")}">{("config. réalisable" if feasible else "config. à adapter")}</text></g>')
        panels.append(''.join(panel))
    payload = dict(schema_version=1, units='m', counts=dict(letters=len(LETTERS), numbers=len(NUMBERS), geometries=len(GEOMETRIES)),
                   robot_count=robot_count, feasibility_config=dict(arena_size_m=config.physics.arena_size,
                       communication_radius_m=config.communication_radius, safety_distance_m=.4), shapes=records)
    (output/'catalog.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{math.ceil(len(entries)/7)*190}" viewBox="0 0 1400 {math.ceil(len(entries)/7)*190}"><rect width="100%" height="100%" fill="white"/>{"".join(panels)}</svg>'
    (output/'catalog.svg').write_text(svg, encoding='utf-8')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--robots', type=int, default=24)
    args = parser.parse_args(argv)
    result = export_catalog(args.output, args.robots)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
