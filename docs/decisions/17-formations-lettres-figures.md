# 17 — Base géométrique extensible aux lettres et figures

Date : 2026-10-29. Demande utilisateur : inclure A, B, C, rectangle, carré et d'autres formes, au-delà du V prévu dans la progression initiale.

## Périmètre livré

`src.reference.path` décrit des chemins ouverts ou fermés en unités SI : taille, orientation, sommets, longueur, abscisse curviligne, projection locale et tangente. Un chemin fermé ne répète pas sa première place. Les constructeurs ligne/V demeurent disponibles. La distance euclidienne au coin du V est vérifiée séparément de l'espacement curviligne.

`src.reference.shapes` ajoute une mission multitraits et les formes nommées ligne, V, carré, rectangle, A, B, C. A conserve sa barre transversale ; B conserve sa verticale et ses deux boucles ; C reste ouvert. Les lettres sont des traits centraux vectoriels, pas des contours de police remplis. B/C utilisent une approximation polygonale de courbes. Les dimensions règlent le cadre de construction ; une lettre ouverte comme C peut occuper moins que toute la largeur nominale. Le centre et l'orientation sont appliqués dans le même repère métrique que le simulateur.

Un chemin unique est échantillonné uniformément en longueur. Une forme multitraits utilise un ensemble de candidats proportionnel aux longueurs puis un échantillonnage déterministe par points éloignés, avec déduplication des intersections. Il s'agit d'une couverture géométrique approximative, pas d'une preuve d'espacement optimal ni d'une répartition parfaite entre traits. Le nombre demandé de places distinctes doit être respecté.

La faisabilité vérifie le confinement des disques, la séparation euclidienne des places et la connectivité du graphe cible à la portée demandée. Un B de cadre 6 m avec 12 places est rejeté à portée 2 m dans l'échantillonnage livré ; portée 3 m permet sa connectivité. Une ligne de 4 m ne peut pas recevoir 12 robots séparés de 0,4 m. Ces refus restent explicites ; les limites physiques ne sont pas assouplies silencieusement.

## Distinction avec les formes historiques

Le projet sait déjà rasteriser des glyphes et des géométries dans `src.shapes`, puis produire des nuages normalisés pour le parcours historique. Ces ressources ne sont pas supprimées. Le nouveau parcours à accélérations exige une consigne en mètres et un contrat de répartition locale avant imitation. Les poids du cercle ne sont pas appliqués directement à A/B/C et ne sont pas présentés comme un modèle multi-formes.

## Travail restant avant apprentissage

Une liste de places partagée définit une géométrie ; elle ne résout pas les conflits d'affectation des robots. L'étape suivante doit spécifier et tester une distribution décentralisée utilisant seulement les observations et communications accessibles, notamment aux intersections et entre traits. Il faudra vérifier les places vides, les regroupements, les permutations, les collisions et les transitions.

La collecte et l'architecture de consigne du GAT devront ensuite être adaptées : le contrat actuel centre/rayon/effectif ne suffit pas à identifier une lettre. Aucun entraînement des nouvelles formes n'est lancé dans cette livraison. La robustesse résiduelle du cercle reste explicitement documentée en 007c.

## Vérification et utilisation

25 tests réussis : géométrie en mètres, projections, coins, absence de place répétée sur contour fermé, A/B/C, dimensions et transformations, alias français, déterminisme, arène, densité et portée.

```python
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig

mission = named_shape('A', robot_count=12, width=6, height=6)
config = AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
mission.validate_environment(config, safety_distance=0.4)
places = mission.reference_points()
stroke_index, projection = mission.project((1.0, 0.0))
```

`places` n'attribue aucune identité de robot. Modifier la taille, l'effectif et la portée exige une nouvelle vérification de faisabilité.
