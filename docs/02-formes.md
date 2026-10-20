# Phase 2 — Formes Cibles (Target Shapes)

## Objectif

Génération de nuages de points normalisés pour les formations cibles de l'essaim.

## Formes Géométriques (7 primitives)

| Forme | Fonction | Description |
|-------|----------|-------------|
| Cercle | `circle_points(n)` | n points sur cercle unité |
| Carré | `square_points(n)` | n points sur périmètre carré [-1,1]² |
| Triangle | `triangle_points(n)` | n points sur triangle équilatéral |
| Ligne | `line_points(n)` | n points sur segment horizontal [-1,1] |
| Croix | `cross_points(n)` | n points sur axes horizontaux + verticaux |
| Étoile | `star_points(n)` | n points sur étoile 5 branches |
| Anneau | `ring_points(n)` | n points sur anneau (rayon 0.5 à 1.0) |

## API

```python
from src.swarm_gnn.shapes import (
    circle_points, square_points, triangle_points,
    line_points, cross_points, star_points, ring_points,
    generate_geometry, save_shape, load_shape
)

# Génération directe
pts = circle_points(20)  # (20, 2)

# Par nom
pts = generate_geometry("star", 30)

# I/O
save_shape(pts, Path("data/shapes/my_shape.npy"))
loaded = load_shape(Path("data/shapes/my_shape.npy"))
```

## Propriétés

- Tous les points dans `[-1, 1] × [-1, 1]` (normalisé)
- Distribution uniforme sur le périmètre/surface
- Reproductible (sauf `ring_points` qui utilise `np.random`)
- Extensible via `generate_geometry()`

## Utilisation dans l'entraînement

```python
# Chargement pour l'entraînement
target = np.load("data/shapes/geometry/geometry_circle.npy")  # (20, 2)
target = target * config.target_radius + np.array(config.arena_size) / 2
# target maintenant dans coordonnées arena, centré, rayon target_radius
```

## Tests

```bash
python -m pytest tests/test_shapes.py -v
```

12 tests :
- 7 formes géométriques (forme, bornes, propriétés)
- `generate_geometry()` pour tous les noms
- Erreur sur nom inconnu
- I/O save/load roundtrip
- Fichier inexistant → FileNotFoundError
- Création répertoires parents