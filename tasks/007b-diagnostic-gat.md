# 007b — Diagnostiquer l'échec du cercle appris

Statut : entraînement réel et évaluation terminés sur autorisation utilisateur ; validation 5/5 sans collision, campagne indépendante 11/20 avec 5 événements de collision. Robustesse insuffisante pour clore le jalon cercle ; poursuite en 007c.

## Résultat réel de la reprise

200 époques exécutées, meilleur modèle à l'époque 197, MSE validation 0,18508326. Les cinq rollouts validation réussissent sans collision. Sur les nouvelles graines 400–419 : GAT 55 % de réussite, professeur 95 % ; erreurs moyennes 0,042090 m et 0,008261 m ; collisions 5 et 1. Les poids ont été figés avant la campagne et les seuils préservés.

Rapports versionnés : `tasks/evidence/007b-circle-relative-run.json`, `007b-circle-relative-diagnostic.json`, `007b-circle-relative-fresh-evaluation.json`. Décision de suite : `docs/decisions/16-campagne-independante-corrections.md`.

Les mentions d'entraînement en attente dans les sections historiques ci-dessous sont remplacées par ce résultat.

Décisions : `docs/decisions/15-diagnostic-geometrie-relative.md`.
Preuve : `tasks/evidence/007b-circle-diagnostic.json`.

## Autorisation du 2026-10-29

L'utilisateur demande explicitement : « lance l'entrainement et passe dans à la prochaine etape ». Cette instruction remplace la restriction antérieure pour cette reprise. Entraînement réel lancé dans `outputs/imitation/training-circle-relative-v1`, avec le YAML relatif et les paramètres initiaux inchangés (200 époques maximum, patience 20, graine 42). L'étape suivante est le contrôle en boucle fermée sur validation ; le test déjà examiné n'est pas chargé pour choisir le modèle.

## Résultat du diagnostic

- MSE validation : 0,297486 sur les états professeur, 1,743084 sur les états visités par le GAT.
- Erreur quadratique tangentielle 0,524832 contre 0,070140 radiale sur les états professeur validation.
- Couverture sous la distance de sécurité : 92/22 800 exemples train, 0/6 000 validation professeur, 2 665/6 000 validation GAT.
- Boucle fermée validation : 0/5 réussites, 19 événements de collision. Les trajectoires professeur réussissent toutes sans collision.
- Limite structurelle démontrée par le code et un test : le classement des voisins de l'attention additive initiale ne dépend pas de la distance au récepteur.

## Correction candidate livrée

Variante à géométrie relative par arête (position relative, vitesse relative, distance), toujours un échange direct. Activation explicite par `config/imitation-model-relative.yaml`, sauvegarde schéma 2, ancien modèle inchangé. MSE et données conservées pour isoler cette modification. L'efficacité après entraînement n'est pas encore démontrée ; la rareté des situations de sécurité demeure.

Tests : 68 réussites sur `test_imitation_model.py`, `test_imitation_integration.py`, `test_imitation_diagnostic.py` et `test_imitation_relative.py`. Mode à blanc relatif exécuté : 1 900 graphes train, 500 validation, aucun optimiseur et aucun dossier d'entraînement créé.

## Commandes pour l'utilisateur

Avec un Python disposant de torch, numpy et pyyaml, depuis la racine :

```powershell
python -m src.imitation.train --dataset outputs/imitation/circle-reviewed-task007 --model config/imitation-model-relative.yaml --output outputs/imitation/training-circle-relative-v1
```

Dans l'environnement Windows actuellement vérifié (le `.venv` transféré depuis Linux est inutilisable), utiliser :

```powershell
$env:PYTHONPATH=(Resolve-Path .codex-test-deps).Path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:/Users/padoda/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m src.imitation.train --dataset outputs/imitation/circle-reviewed-task007 --model config/imitation-model-relative.yaml --output outputs/imitation/training-circle-relative-v1
```

Cette commande lance un entraînement réel, à exécuter par l'utilisateur. Ajouter `--dry-run` pour une vérification sans optimisation. Le dossier de sortie doit être neuf. Les dépendances locales `.codex-test-deps` sont ignorées par Git et ne constituent pas un environnement portable.

Après l'entraînement, lancer le diagnostic train/validation, avant toute nouvelle évaluation finale :

```powershell
& 'C:/Users/padoda/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m src.imitation.diagnose --run outputs/imitation/training-circle-relative-v1 --dataset outputs/imitation/circle-reviewed-task007 --output outputs/imitation/diagnostic-circle-relative-v1
```

Conserver les variables d'environnement ci-dessus dans cette session. Le rapport est `diagnostic.json`, avec les trajectoires GAT validation. Le diagnostic initial se reproduit avec `training-circle-v1` et un nouveau dossier de sortie. Ne pas utiliser `compare` sur les anciens épisodes test pour sélectionner la nouvelle variante.

## Constat

Le premier modèle entraîné échoue sur les cinq tests réservés : erreur finale moyenne 1,149841 m et 22 événements de collision. Le professeur réussit les cinq épisodes sans collision. Les tests d'équivalence entre l'adaptateur local et le graphe global passent ; cela ne suffit pas à expliquer l'échec du modèle appris.

## Travail attendu

- Examiner les erreurs de commande sur train/validation selon le temps, le degré local, la distance de sécurité et les composantes radiale/tangentielle.
- Comparer sur validation les états visités en boucle fermée aux démonstrations du professeur et mesurer la dérive.
- Distinguer les hypothèses (couverture des données, pondération de la MSE, capacité d'attention) des causes effectivement démontrées.
- Proposer une correction justifiée, avec tests ciblés et commandes d'entraînement remises à l'utilisateur. Aucun entraînement lancé par l'agent.
- Conserver le modèle initial, son rapport et les seuils du protocole. Ne pas sélectionner de correction sur les graines test 300–304 déjà examinées ; prévoir un nouveau test indépendant pour toute nouvelle conclusion finale.

## Critère de fin

Diagnostic reproductible et correction vérifiée sur train/validation, puis nouveau modèle entraîné par l'utilisateur et évalué avec échecs explicites. Le cercle appris doit être validé avant les formations ligne/V.
