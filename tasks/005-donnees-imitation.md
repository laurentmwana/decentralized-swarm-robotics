# 005 — Collecte du professeur et séparation par épisodes

Statut : validée par l’utilisateur ; commit autorisé.
Décisions : docs/decisions/12-donnees-imitation.md.

## Livraison

- Initialisation déterministe sans chevauchement, distance de sécurité et budget borné.
- Configuration explicite de trois ensembles avec graines disjointes.
- Archives contenant observations propres, consigne, voisins à portée, graphe, commandes bornées/brutes et trajectoires.
- Manifeste complet, contrôle des duplications exactes et empreintes des fichiers.
- Chargement vérifiant ensemble et intégrité ; refus des collectes incomplètes et de qualité non validée.
- Commande de collecte sans entraînement : `python -m src.imitation.collect --output outputs/imitation/circle-v1`.

## Résultats et limites

30 épisodes et 36 000 exemples produits dans outputs/imitation/circle-task005/.
train : 19/20 réussites du professeur ; validation : 5/5 ; test : 5/5.
La graine 113 contient deux collisions pendant le contrôle : conservée et marquée comme échec. Le jeu porte le statut review_required ; les données ne sont pas déclarées prêtes à entraîner. Le manifeste complet est versionné dans tasks/evidence/005-circle-manifest.json.

## Vérification

155 tests réussis : 128 précédents et 27 nouveaux, couvrant initialisation, partition, alignement états/commandes, voisinages masqués, duplication, corruption, sorties existantes, reproductibilité, qualité et commande de collecte.
Les tests historiques nécessitant PyTorch restent hors du périmètre vérifié.
Aucun entraînement neuronal exécuté ; la collecte consiste uniquement en simulations du professeur.

## Suite après validation et commit

Tâche 006 : architecture GAT et MSE supervisée, sans entraînement. Avant la tâche 007, traiter le défaut de qualité du professeur observé sur la graine 113 et revalider/versionner la collecte si nécessaire. Les commandes d'entraînement seront remises à l'utilisateur pour exécution.
