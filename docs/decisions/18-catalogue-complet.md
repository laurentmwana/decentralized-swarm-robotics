# 18 — Catalogue A–Z, 1–10 et vingt géométries

Date : 2026-10-29. Périmètre explicitement demandé par l'utilisateur : alphabet complet, nombres de 1 à 10 et au moins quinze figures géométriques.

## Catalogue livré : 56 entrées

- 26 lettres majuscules A à Z, en traits centraux vectoriels.
- 10 nombres : 1, 2, 3, 4, 5, 6, 7, 8, 9 et 10. Le nombre 10 associe les glyphes 1 et 0 dans une seule formation.
- 20 géométries : cercle, ellipse, carré, rectangle, triangle, pentagone, hexagone, heptagone, octogone, décagone, losange, trapèze, parallélogramme, étoile, croix, flèche, demi-cercle, cœur, ligne et V.

La représentation est indépendante d'une police installée ou du rendu raster historique. Elle conserve les traits séparés, les ouvertures et les boucles. Les courbes sont approximées par des polylignes ; il s'agit d'un alphabet vectoriel simplifié, pas d'une police typographique. La structure multitraits permet d'étendre le catalogue.

## Identifiants et compatibilité

Identifiants explicites : `letters_A` à `letters_Z`, `numbers_1` à `numbers_10`, `geometries_square`, etc. Les noms courts `A`, `10`, `rectangle` et les principaux alias français sont acceptés. `V` désigne la lettre ; `geometries_v` désigne la géométrie. Ces deux entrées partagent le même tracé mais gardent des identifiants distincts.

Un rectangle sans hauteur explicite utilise les deux tiers de sa largeur pour être visuellement distinct du carré. Cercle et carré refusent des dimensions inégales. Chaque mission fournit ses traits, son identifiant, ses points de référence, ses projections et le contrôle de faisabilité du simulateur. Aucune identité de robot n'est attribuée aux places par cette API.

```python
from src.reference.catalog import catalog_entries
from src.reference.shapes import named_shape

catalogue = catalog_entries()  # 56 identifiants stables
mission = named_shape('letters_R', robot_count=24, width=6)
ten = named_shape('numbers_10', robot_count=24, width=6)
rectangle = named_shape('geometries_rectangle', robot_count=24, width=6, height=4)
```

## Export et vérification

```powershell
python -m src.reference.catalog --output outputs/formations/catalog-complete-v1 --robots 24
```

Le dossier neuf reçoit `catalog.json` et `catalog.svg`. Le JSON conserve les 56 définitions en mètres, les points, les indicateurs de fermeture et le résultat du contrôle de faisabilité. La planche SVG montre tous les tracés et points cibles ; elle étiquette les configurations à adapter. Le catalogue ne supprime pas une forme dont le nombre de robots choisi est incompatible avec la taille, la sécurité ou la portée.

147 tests géométriques réussis : couverture de toutes les entrées à 12 et 24 places, unicité des points, appartenance aux traits, reproductibilité, dimensions, identifiants, nombre composé 10, export JSON et structure de la planche SVG. Les tests géométriques ne constituent pas une réussite de contrôle en essaim.

## Suite de la tâche 008

Le catalogue est terminé pour ce périmètre. La répartition locale, les métriques de couverture par trait, les professeurs, les transitions, la collecte et l'entraînement multi-formes restent à développer. Les poids actuels du cercle restent spécifiques au cercle. Les partitions d'apprentissage/validation/test des formes devront être définies avant tout entraînement sur ce nouveau catalogue ; aucun découpage historique n'est modifié silencieusement.
