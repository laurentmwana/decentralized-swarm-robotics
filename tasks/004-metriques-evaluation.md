# 004 — Métriques et évaluation du professeur circulaire

Statut : validée par l’utilisateur ; commit autorisé.
Références : PLAN.md §8 ; docs/decisions/10-metriques-cercle.md.

## Livraison

- Erreur circulaire en mètres contrôlant rayon et répartition angulaire.
- Temps de stabilisation avec seuil de vitesse, durée de maintien et délai maximal.
- Collisions comptées par événements de paire, y compris initiales et traversées entre états.
- Réussite stricte sans collision ; taux de réussite agrégé ; temps null en cas d'échec.
- Protocole YAML et campagne reproductible : `python -m src.reference.evaluate`.
- Trajectoires, paramètres et rapport exportés ; preuve légère dans tasks/evidence/004-circle-pilot.json.

## Vérification

117 tests réussis : 85 existants de la migration et 32 nouveaux sur les métriques et la campagne.
Les tests historiques nécessitant PyTorch ne sont pas inclus dans cette vérification.
Aucun entraînement exécuté : uniquement des simulations du contrôleur classique.

## Pilote : échec explicite du jalon, métriques fonctionnelles

10 épisodes, graines 0 à 9, 12 robots, portée 2 m, délai 10 s.
0 % de réussite ; erreur finale moyenne 0,9116 m, écart-type 0,4914 m ; un événement de collision initiale (graine 9). Aucun temps de stabilisation imputé aux échecs.
Les fichiers bruts sont dans outputs/plan-v2/circle-evaluation-task004/ (ignorés par Git). Le rapport versionné conserve la configuration complète.

## Suite proposée après validation et commit

Tâche 004b : diagnostiquer et corriger le professeur depuis des positions dispersées ; ajouter des tests reproduisant les défauts et réévaluer sous le même protocole, puis vérifier sur d'autres graines. La collecte de données (005) attend un professeur validé. Le jalon du cercle classique n'est pas déclaré réussi.
