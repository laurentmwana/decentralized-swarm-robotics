# 007 — Intégration et commandes à exécuter par l'utilisateur

Statut : entraînement utilisateur retrouvé et comparaison réelle exécutée le 2026-10-29 ; jalon de qualité du GAT non atteint. Aucun commit effectué par l'agent.
Décisions : docs/decisions/14-integration-entrainement-utilisateur.md.

## Réalisation

## Reprise du 2026-10-29 : résultat réel

Le run `outputs/imitation/training-circle-v1/run.json` porte le statut trained : 69 époques, meilleur modèle à l'époque 49, MSE validation 0,29748584. Aucun nouvel entraînement lancé par l'agent.

Le déplacement Linux → Windows exige un paramètre `--dataset` pour la comparaison. Le manifeste est toujours vérifié contre l'empreinte enregistrée à l'entraînement ; le run d'origine reste intact. Le rapport enregistre le dossier réellement utilisé.

Commande portable avec un Python et les dépendances disponibles :

```powershell
python -m src.imitation.compare --run outputs/imitation/training-circle-v1 --dataset outputs/imitation/circle-reviewed-task007 --output outputs/imitation/comparison-circle-v1
```

Résultat sur les cinq épisodes test 300–304, protocole inchangé :

| Mesure | Professeur | GAT |
|---|---|---|
| Réussite | 5/5 (100 %) | 0/5 (0 %) |
| Erreur finale moyenne | 0,003160 m | 1,149841 m |
| Collisions (événements) | 0 | 22 |
| Temps moyen de stabilisation des réussites | 4,98 s | null (aucune réussite) |

MSE sur les états test du professeur : 0,37933957. Ce résultat ne démontre pas une maîtrise du cercle par le GAT. Les contacts apparaissent pendant le contrôle ; aucun épisode ne comporte de contact initial.

Preuves : `tasks/evidence/007-circle-trained-run.json` et `tasks/evidence/007-circle-comparison.json`. Trajectoires brutes : `outputs/imitation/comparison-circle-v1/`.

Vérification du correctif : 21 tests d'intégration réussis sous Python 3.12.14 / PyTorch 2.15.0 Windows, dont déplacement du jeu et refus d'une empreinte différente. Les 222 tests antérieurs ci-dessous constituent l'historique et n'ont pas tous été rejoués lors de cette reprise.

Suite : tâche 007b avant 008. Les sections suivantes décrivent la préparation antérieure ; leurs mentions « en attente » sont remplacées par le résultat réel ci-dessus.

## Préparation antérieure

- Revue explicite du jeu brut : exclusion de train/113, conservation intégrale de validation/test et provenance versionnée.
- Version revue prête à charger : outputs/imitation/circle-reviewed-task007 (29 épisodes, 34 800 exemples).
- Adaptateur GAT local dans les épisodes du nouveau simulateur.
- Entraînement MSE, lots de graphes, sélection sur validation, arrêt anticipé, sauvegardes et historique.
- Commande de comparaison sur test réservé, avec conditions initiales identiques professeur/GAT.
- Mode à blanc exécuté sans optimiseur ni écriture d'entraînement : 1 900 graphes train, 500 validation.

## Vérification

222 tests ciblés réussis : 203 précédents et 19 nouveaux. Aucun entraînement lancé.
Les tests de comparaison emploient un run fictif et des poids initiaux, uniquement pour vérifier le raccordement. Aucun résultat de modèle appris n'est encore disponible.
La boucle d'optimisation réelle n'a pas été exécutée ; elle attend l'utilisateur.

## Commandes depuis la racine du projet

Raccourci Make pour l'environnement Linux temporaire actuellement disponible :

```bash
make train-imitation-local
```

Avec un environnement Python persistant correctement activé : `make train-imitation`.
Les variables `IMITATION_DATASET` et `IMITATION_OUTPUT` permettent de choisir les dossiers.
Les chemins temporaires Linux sont surchargeables par `IMITATION_LOCAL_PYTHON` et `IMITATION_LOCAL_PACKAGES`.
Les deux cibles lancent un entraînement réel : elles doivent être exécutées par l'utilisateur.
Vérification du raccordement : `make -n train-imitation-local` et `make -n train-imitation` affichent les commandes sans entraîner.


Avec un environnement Python disposant des dépendances du projet :

```bash
python -m src.imitation.train --dataset outputs/imitation/circle-reviewed-task007 --output outputs/imitation/training-circle-v1
```

Dans l'environnement Linux actuellement disponible, les dépendances de vérification ont été installées dans /tmp. La commande exacte utilisable maintenant est :

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/swarm-gnn-test-deps:/tmp/swarm-gnn-torch \
/home/padoda/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
-m src.imitation.train \
--dataset outputs/imitation/circle-reviewed-task007 \
--output outputs/imitation/training-circle-v1
```

Ces chemins /tmp sont temporaires ; après nettoyage ou redémarrage, installer les dépendances dans un environnement Python persistant avant de reprendre. La commande standard ci-dessus fonctionne alors avec cet environnement activé.

À la fin, communiquer le fichier outputs/imitation/training-circle-v1/run.json et la disponibilité de best.pt. La tâche 007 ne sera déclarée terminée qu'après examen du résultat et comparaison.

La comparaison après entraînement utilisera :

```bash
python -m src.imitation.compare --run outputs/imitation/training-circle-v1 --output outputs/imitation/comparison-circle-v1
```

Sous l'environnement temporaire Linux, préfixer cette commande par les mêmes variables et remplacer python par l'interpréteur indiqué ci-dessus. Aucun dossier de sortie existant n'est écrasé : choisir un nouveau nom pour une nouvelle exécution.

## Reproduire la revue (facultatif, nouveau dossier requis)

```bash
python -m src.imitation.review --source outputs/imitation/circle-task005 --output outputs/imitation/circle-reviewed-v2
```

Cette commande ne modifie ni la collecte brute ni le jeu revu existant. Elle ne lance aucun entraînement.
