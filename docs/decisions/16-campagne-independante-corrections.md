# 16 — Campagne indépendante et corrections sur états visités

Date : 2026-10-29. Entraînements autorisés explicitement par l'utilisateur.

## Variante relative : entraînement et première campagne

200 époques exécutées, meilleur modèle à l'époque 197, MSE validation 0,18508326. Les cinq rollouts validation réussissent sans collision ; erreur finale moyenne 0,042244 m. L'ancien run et ses poids restent conservés.

Avant toute simulation indépendante, `evaluate_fresh` écrit le modèle figé (empreinte SHA256), le protocole, les graines et le domaine d'initialisation dans `protocol.json`. Les graines de la collecte source, y compris les épisodes exclus de train, sont refusées. L'outil ne peut pas vérifier à lui seul toutes les campagnes historiques : choisir et consigner des graines réellement inédites reste nécessaire. Les états initiaux sont identiques pour les deux contrôleurs et tous les échecs sont conservés. Aucun optimiseur ni lecture des archives de démonstration n'intervient dans cette campagne.

Graines 400–419, 12 robots, portée 2 m, horizon 10 s ; seuils précédents inchangés :

| Mesure | Professeur | GAT relatif |
|---|---:|---:|
| Réussite | 19/20 (95 %) | 11/20 (55 %) |
| Erreur finale moyenne | 0,008261 m | 0,042090 m |
| Événements de collision | 1 | 5 |
| Temps moyen, réussites seulement | 6,1263 s | 7,1091 s |

Les deux contrôleurs peuvent échouer malgré une initialisation sans chevauchement. La seule erreur finale de formation ne suffit pas à conclure au succès. Ce résultat de généralisation limite la conclusion obtenue sur les cinq épisodes de validation ; le jalon de robustesse du cercle reste ouvert.

Preuves : `tasks/evidence/007b-circle-relative-run.json`, `007b-circle-relative-diagnostic.json` et `007b-circle-relative-fresh-evaluation.json`.

## Cycle de corrections supervisées

Décision : ajouter un seul cycle de corrections, sans consulter les graines 400–419 pour sélectionner de nouveaux poids. Cette campagne reste le résultat indépendant de la variante relative initiale ; après ce constat elle ne doit pas servir de nouveau test final pour le modèle corrigé.

`src.imitation.corrections` rejoue uniquement les 19 épisodes train depuis leurs états initiaux archivés, sous le GAT relatif figé. Sur chaque état visité, le professeur local fournit l'accélération attendue bornée. Les observations du professeur sont limitées aux mêmes voisins directs. Les actions réellement exécutées par le GAT sont stockées séparément des labels du professeur. Aucun succès du professeur en boucle fermée n'est imputé à ces trajectoires ; les échecs et collisions du collecteur restent dans le manifeste.

Ce supplément est distinct du jeu de démonstrations revu et possède un type explicite `train_on_policy_teacher_labels`. Il n'est pas soumis au critère « trajectoires professeur réussies », puisqu'il contient précisément des états d'échec du collecteur. Les erreurs ne sont ni supprimées ni requalifiées. Le chargeur contrôle : version et achèvement, empreinte du jeu source, contrat physique/professeur, une archive par graine train, empreinte du parent, intégrité et métadonnées de chaque archive. Validation et test ne peuvent pas être ajoutés aux corrections.

Le nouvel entraînement conserve les 1 900 graphes train initiaux et ajoute les 1 900 graphes corrigés, avec le même poids par graphe. Les 500 graphes validation restent identiques. Même modèle relatif et même MSE ; initialisation depuis le meilleur checkpoint relatif, nouvel Adam à 0,0003, 100 époques maximum, patience 20, graine 42. Le modèle initial est inclus dans la sélection (époque 0) ; si la MSE validation n'est pas améliorée, ses poids sont conservés. La traçabilité du run inclut l'empreinte des poids initiaux et du manifeste de corrections.

Cette collecte n'est qu'un cycle d'agrégation de données ; elle ne constitue pas une convergence démontrée de DAgger. Le professeur reste heuristique et ses labels sur les états visités n'offrent pas une garantie de sécurité.

## Commandes

Dans un environnement Python disposant des dépendances :

```powershell
python -m src.imitation.corrections --run outputs/imitation/training-circle-relative-v1 --dataset outputs/imitation/circle-reviewed-task007 --output outputs/imitation/corrections-circle-relative-v1
python -m src.imitation.train --dataset outputs/imitation/circle-reviewed-task007 --model config/imitation-model-relative.yaml --training config/imitation-training-corrections.yaml --corrections outputs/imitation/corrections-circle-relative-v1 --initial-checkpoint outputs/imitation/training-circle-relative-v1/best.pt --output outputs/imitation/training-circle-corrections-v1
python -m src.imitation.diagnose --run outputs/imitation/training-circle-corrections-v1 --dataset outputs/imitation/circle-reviewed-task007 --output outputs/imitation/diagnostic-circle-corrections-v1
```

Les dossiers existants sont refusés ; choisir de nouveaux noms pour reproduire. Sous Windows dans la session vérifiée, employer le Python du runtime et `PYTHONPATH=.codex-test-deps` décrits dans la tâche 007b.

## Résultat réel du cycle de corrections

100 époques exécutées ; meilleur modèle à l'époque 80, MSE validation 0,16171945 contre 0,18508327 initialement. Les cinq rollouts validation réussissent sans collision ; erreurs finales 0,00951–0,01989 m. Les poids et le protocole ont ensuite été figés avant la campagne sur les nouvelles graines 600–619.

| Mesure, graines 600–619 | Professeur | Relatif initial | Relatif corrigé |
|---|---:|---:|---:|
| Réussite | 19/20 (95 %) | 14/20 (70 %) | 17/20 (85 %) |
| Erreur finale moyenne | 0,006681 m | 0,042657 m | 0,022025 m |
| Événements de collision | 1 | 3 | 2 |
| Stabilisation moyenne, réussites seulement | 5,8105 s | 6,8143 s | 6,5588 s |

Le modèle relatif initial, déjà figé avant le cycle de corrections, a été évalué aussi sur 600–619 pour obtenir une comparaison appariée des variantes. Aucun réglage ni nouvel entraînement n'a suivi l'examen de cette campagne. Les valeurs 55 % (400–419) et 85 % (600–619) portent sur des graines différentes : la comparaison contrôlée de l'effet des corrections est 70 % contre 85 % sur les mêmes graines 600–619. Les petites tailles de campagne et les échecs résiduels limitent la portée de la conclusion ; aucune sûreté universelle n'est affirmée.

Le cercle corrigé devient le prototype de référence de cette reprise ; ses trois échecs sont conservés. La préparation géométrique de nouvelles formes peut avancer indépendamment, mais la robustesse du cercle et la validation du contrôle des nouvelles formes restent des travaux distincts.

Preuves : `tasks/evidence/007c-circle-corrections-run.json`, `007c-circle-corrections-diagnostic.json`, `007c-circle-corrections-fresh-evaluation.json`, `007c-circle-relative-paired600-evaluation.json`. Le dernier `best.pt` est dans `outputs/imitation/training-circle-corrections-v1/`.

Vérification : 87 tests ciblés sur modèle, intégration, diagnostic, géométrie relative, campagne indépendante et corrections. Ils incluent le maintien du modèle initial si les nouvelles validations sont moins bonnes, sans mise à jour de poids dans ce test.
