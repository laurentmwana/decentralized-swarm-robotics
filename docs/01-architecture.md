# 01 — Architecture

## Flux général
Forme cible (image) → nuage de points → contrôleur local partagé → simulateur 2D → métriques et perte. Pendant l'entraînement, la perte globale est rétropropagée à travers tous les pas de simulation jusqu'aux paramètres partagés. À l'exécution, chaque robot applique la même politique à partir de son observation locale, de ses voisins et de la consigne.

## Modules et artefacts

| Module | Responsabilité | Artefacts |
|---|---|---|
| Simulation | dynamique, voisinage, arène, pannes, animation | positions, graphe, animation |
| Formes cibles | rendu et échantillonnage | images, tableaux de points, index |
| Contrôleurs | GNN et contrôleurs de référence | commandes, vérification de vitesse |
| Entraînement | boucle, pertes, curriculum | modèles, courbes, métriques |
| Évaluation | scénarios de pannes et d'échelle | CSV bruts, agrégats, figures |
| Démonstration Web | API locale | réponses JSON de trajectoires |
| Génération des figures | figures du mémoire | PNG/PDF |

## Contraintes de conception
- Un même épisode doit pouvoir être rejoué avec un autre contrôleur sans toucher à la dynamique (règle manuelle et GNN passent par la même interface d'épisode).
- Aucun résultat ne dépend d'un réglage caché : tout est dans les configurations.
- Une fonction de visualisation ne peut pas influencer la simulation.

## Structure du dépôt proposée (à aligner)
```
configs/        fichiers de configuration versionnés
src/            simulation, formes, contrôleurs, entraînement, évaluation, api
data/shapes/    images et nuages générés (non versionnés)
docs/           cette documentation
tests/          tests unitaires
outputs/        <expérience>/<niveau>/ : modèles, pertes, métriques (non versionnés)
figures/        sorties du module de figures
```

## Cycle d'un épisode
1. Tirer la cible et les positions initiales (vitesses nulles), selon la graine.
2. Pour chaque pas : construire le graphe, faire passer les messages (2 tours), décoder une vitesse bornée, appliquer les commandes, appliquer les pannes éventuelles.
3. Après le dernier pas : mesurer sur les robots actifs uniquement.
4. Entraînement : perte + rétropropagation + Adam. Évaluation : poids gelés, métriques seulement.

Le même mécanisme sert à l'entraînement et à l'évaluation.

## Coût de calcul
La mémoire croît avec la longueur des épisodes, le nombre de robots et le nombre de paires de voisins, car l'historique des pas est conservé pour la rétropropagation. D'où un profil de vérification rapide (« smoke ») qui teste le pipeline sans permettre de conclure sur la qualité.
