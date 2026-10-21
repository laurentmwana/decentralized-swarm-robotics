# PLAN.md — Contrôle d'essaim par GNN (plan détaillé)

Ce document décrit, pour chaque phase : l'objectif, les classes à écrire (programmation orientée objet), les tâches, ce que l'on doit **produire** (fichiers, tableaux, figures), les **résultats attendus** (ce qu'on doit pouvoir lire à la fin), le critère d'acceptation et les points de vigilance.

Toutes les règles de travail (POO, conventions, configuration, tests, simulateur, évaluation) sont dans le dossier `skills/`, à lire avant de travailler sur le domaine concerné.

---

## 0. Principes et conventions

### 0.1 Légende
Cases :
- `[ ]` à faire
- `[~]` implémentation préparée, critère d'acceptation non vérifié
- `[x]` fait **et** critère validé

Priorités :
- Noyau : indispensable au mémoire
- Important : à faire si possible
- Bonus : à sacrifier en premier

### 0.2 Règles de fonctionnement
1. Une phase est terminée seulement quand son critère d'acceptation est validé et coché explicitement.
2. **Tout le code est orienté objet** (règles dans `skills/`) : une classe par responsabilité, classes abstraites pour les contrats, injection de dépendances, objets de valeur immuables.
3. Aucun paramètre dans le code : tout vit dans des configurations versionnées.
4. Chaque chiffre produit doit être traçable jusqu'à un fichier brut.
5. Deux sources de variabilité, toujours rapportées séparément :
   - les graines d'entraînement (5 au minimum) ;
   - les épisodes d'évaluation (50 répétitions par condition).
6. Une phase ne démarre pas tant que la précédente n'est pas validée, sauf mention contraire. **Exception explicite demandée par l'utilisateur pour cette réécriture : l'implémentation seule des phases suivantes peut être préparée avant la validation empirique des précédentes.** Cette exception ne valide aucun résultat : les campagnes et leurs critères d'acceptation restent soumis aux dépendances et aux preuves prévues.
7. Avant chaque phase, relire les skills concernés ; toute nouvelle règle est ajoutée dans `skills/` dans le cadre d'une tâche autorisée.
8. L'agent ne lance aucun entraînement local, y compris avec le profil `smoke`. L'utilisateur lance lui-même les commandes d'entraînement du `Makefile`. Les tests de propriétés, de gradients et d'orchestration simulée sont autorisés ; ils ne remplacent jamais l'entraînement ni la validation empirique.
9. Tous les noms de fonctions, variables, clés, dossiers et fichiers sont en anglais ; les commentaires et explications sont en français.

### 0.3 Organisation des fichiers
- `config/` : toutes les configurations versionnées (une par expérience).
- `src/` : le code (simulateur, formes, métriques, contrôleurs, entraînement, évaluation).
- `tests/` : les tests automatiques.
- `docs/decisions/` : les décisions écrites (`02-simulator.md`, `06-evaluation.md`, etc.).
- `outputs/` : sorties brutes, non versionnées, rangées par expérience : `outputs/<phase>/<run_name>/`.
- `figures/` : figures finales générées depuis `outputs/`, jamais retouchées à la main.

### 0.4 Contenu minimal de chaque dossier de run
Tout run (entraînement ou évaluation) doit contenir :
- `effective_config.yaml` : la configuration réellement utilisée (copie figée) ;
- `seeds.json` : toutes les graines utilisées ;
- `version.txt` : identifiant du code (commit) et provenance des dépendances à vérifier ;
- un journal brut anglais adapté : `losses.csv` pour l'entraînement, `episodes.csv` pour la campagne, `log.csv` pour les démonstrations ;
- `summary.json` : les chiffres clés du run (pour construire les tableaux sans tout relire).
Les entraînements prévoient aussi `targets.json` (points, identifiants et découpages) et `status.json` (état du run). Les campagnes figent les configurations sources et les empreintes SHA-256 des checkpoints. Ces noms décrivent les artefacts prévus, pas des sorties produites lors de cette mise à jour.

### 0.5 Décisions à figer avant de coder (à écrire dans `docs/decisions/`)
- Définition unique du succès et valeur du seuil (`docs/decisions/06-evaluation.md`).
- Règle des robots retirés : supprimés du graphe ou figés et présents (`docs/decisions/02-simulator.md`).
- Distance comparable entre tailles d'essaim (normalisation, ou résolution de cible fixe).
- Repère commun 2D et consigne de forme (identifiant de forme partagé ; vecteur vers le contour en variante séparée).
- Découpage entraînement / validation / test des formes.

### 0.6 Leçons du premier prototype (historique)
Une figure du premier prototype (100 robots, graine 142) avait montré deux anomalies à corriger dès la phase A :
- La cible contenait 100 points mais seulement 20 étaient distincts (répétés cinq fois) : la forme visée était trop grossière et l'essaim formait un amas au lieu de suivre un contour.
- Le succès était enregistré au pas 1 alors que les robots étaient encore loin de la cible et bougeaient jusqu'au pas 20 : la distance ou le seuil était mal défini.
Ces observations historiques imposent deux tests obligatoires (voir phases 1 et 2), mais ne prouvent rien sur les résultats de la réécriture actuelle.

**Références actuelles (2026-10-21)** : `validate_target` dans `src/training/train.py`, `ShapeSampler` dans `src/shapes/sampling.py`, `SuccessCriterion` dans `src/metrics/success.py` ; tests dans `tests/test_training.py`, `tests/test_shapes.py` et `tests/test_metrics.py`. Les anciennes références à des fonctions et lignes du prototype sont périmées. Les flux aléatoires sont désormais dérivés des noms anglais : le déterminisme est conservé, mais les trajectoires historiques utilisant les anciens noms ne sont pas identiques.

### 0.7 État de la réécriture (2026-10-21)
- Phases 5 à 13 : `[~]`, implémentation préparée ; acceptation empirique non vérifiée. Aucun entraînement réel ni aucune campagne formelle n'ont été exécutés par l'agent. L'utilisateur réalise les entraînements via les commandes explicites du `Makefile`.
- Suite complète sans entraînement : **371 tests réussis, 1 test d'optimisation réelle exclu**, sans avertissement (`python -m pytest tests/ -q --tb=short`, 154,97 s). Syntaxe Python, espaces sur le périmètre modifié, huit aides CLI et plans sans optimisation vérifiés. Les tests synthétiques, de propriétés et d'orchestration ne constituent pas des résultats expérimentaux. Le test marqué `training` est exclu par défaut ; `make test-training` est réservé à l'utilisateur.
- Curriculum cumulatif, évaluations des formes vues/réservées, protocole humain de lisibilité, perturbations de robustesse et démonstration Web implémentés. Rapports issus des artefacts sauvegardés ; aucun score de modèle entraîné ou réponse humaine inventé.
- Identifiants actuels : clé `model.message_rounds`, `Mission.target`, `Controller.command`, `GNNController.model`, méthode `parameter_count` ; métriques accessibles par `MetricSuite.compute(...).values` (`MetricReport.values`).
- Les statuts de validation déjà consignés pour les phases 1 à 4 sont conservés ; leurs tests automatiques ont été relancés, pas leurs campagnes ni leurs inspections visuelles. Les anciens résultats des phases 5, 9 et 10 sont historiques, non recevables pour accepter la réécriture.
- Démonstration utilisée dans le navigateur avec catalogue synthétique et contrôleur manuel : réponses déterministes et erreurs vérifiées. Capture native indisponible, donc inspection visuelle différée. Le catalogue historique incompatible n'a pas été modifié ; lancer `make shapes` avant d'utiliser les nouveaux modules.

---

## Phase A — Audit de l'existant (Noyau) [ ]

**Objectif** : savoir exactement ce qui existe, ce qui marche et ce qui est à refaire, avant d'ajouter quoi que ce soit.

**Classes** : aucune nouvelle ; vérifier que le code existant respecte `skills/` et lister les modules écrits en fonctions isolées à convertir en classes.

**Tâches**
- [ ] Lister, phase par phase, ce qui existe réellement dans le dépôt (modules, config, tests, résultats).
- [ ] Lancer tous les tests ; noter ceux qui échouent et pourquoi.
- [ ] Décocher tout ce dont le critère d'acceptation n'a pas été vérifié.
- [ ] Vérifier les deux anomalies de la section 0.6 sur les sorties existantes.
- [ ] Corriger l'ordre des dépendances (le curriculum précède l'entraînement des formes complexes).
- [ ] Inventorier les paramètres codés en dur dans le code et les déplacer vers les configurations.

**Livrables**
- `docs/decisions/00-audit.md` : tableau daté « fait / à vérifier / à refaire » par phase.
- `outputs/audit/tests.txt` : sortie complète des tests (réussites et échecs).
- Liste des paramètres en dur à externaliser.

**Résultats attendus**
- Une page de synthèse : nombre de tests réussis et échoués, liste des phases réellement terminées, liste des défauts connus.
- Un plan de travail corrigé, avec les cases mises à jour.

**Critère d'acceptation** : le tableau d'audit est daté et chaque ligne a un statut justifié par une preuve (test, fichier, figure).

**Points de vigilance** : ne pas corriger en cours d'audit ; d'abord constater, ensuite décider.

**Audit du prototype (historique, consigné le 2026-10-21)** — les anciennes corrections et déclarations d'absence ne valent pas inventaire de la réécriture :
1. **Clés mortes** : les poids de couverture et de transport étaient auparavant déclarés sans être utilisés. Vérifier désormais les quatre termes de `src/training/losses.py` et les clés `losses.weights` de `config/training.yaml` ; ne pas confondre l'ancien transport avec un terme actuellement implémenté. Aucune affectation globale robot→point ne doit entrer dans le contrôleur.
2. **Consigne de contour absente ou figée** : ancien défaut du prototype ; références actuelles `use_target_geometry` dans `config/model.yaml`, `src/training/rollout.py` et `GNNController.target_vector`.
3. **Phase 10 cassée** : les anciennes anecdotes de checkpoint et de `KeyError` ne décrivent pas un module actuel ; phase à préparer après le curriculum.
4. **Définition du succès** : la déclaration historique d'absence de `docs/decisions/06-evaluation.md` est périmée (voir phase 2). Conserver une définition unique conforme à `SuccessCriterion` et à `success_threshold` avant toute comparaison empirique.
5. **Conventions et audit** : vérifier l'inventaire de `skills/` et le livrable `docs/decisions/00-audit.md` lors d'un audit autorisé, sans créer de documentation ni modifier les règles dans la présente tâche.

---

## Bloc 1 — Fondations (Noyau)

### Phase 1 — Simulateur et formes cibles [x] (validée 2026-10-21)

**État (2026-10-21, réécriture from scratch)** — structure `src/simulator/`, `src/shapes/`, `config/`, `tests/`.
- Simulateur : `Arena`, `SwarmState`, `CommunicationGraph` (sans auto-boucle, recalculé à chaque pas), `FailureModel`, `Simulator` (commande bornée avant mise à jour), `Episode`/`EpisodeResult`, façade `simulate_episode`. Règle des robots retirés figée dans `docs/decisions/02-simulator.md` (supprimés du graphe, immobiles, hors métriques, présents dans l'état).
- Formes : `ShapeSource` abstrait + `GeometryShape`/`GlyphShape`/`NumberShape`, `ShapeSampler` (farthest-point, graine contrôlée), `ShapeIndex`, `ShapeSplit`. Découpage figé dans `docs/decisions/04-shapes.md` et `config/shapes.yaml`. 64 formes, index + 4 planches dans `outputs/shapes/`.
- Validation de tests consignée précédemment, revérifiée par la suite automatique sans entraînement : anti-doublon, vitesse bornée, robot isolé, robot en panne immobile et hors graphe, même graine = même épisode, normalisation dans le cadre commun, présence dans l'index, clés de config manquantes/inconnues refusées.
- Critère d'acceptation consigné comme validé : contrôleur trivial tourne sur 20/40/100 robots, reproductible (`figures/demo_trivial_sizes.png`, `outputs/phase1_demo/`) ; planches inspectées. Pas de nouvelle exécution dans ce tour.
- Différé : les nombres à 3 chiffres restent grossiers à 50 points (prévu, résolution 80–100 en phase 10).


**Objectif** : disposer d'un simulateur fiable et d'un jeu de formes cibles bien échantillonnées, reproductibles et testées.

**Classes (POO)** : `ExperimentConfig`, `ConfigLoader`, `SeedManager`, `Arena`, `SwarmState`, `CommunicationGraph`, `FailureModel`, `Simulator`, `Episode`, `EpisodeResult` ; formes : `ShapeSource` (abstraite) avec `GeometryShape`, `GlyphShape`, `NumberShape`, puis `ShapeSampler`, `ShapeIndex`, `ShapeSplit`.

**Tâches simulateur**
- [ ] Arène carrée ; état par robot (position, vitesse) ; vitesse directement commandée.
- [ ] Commande bornée par la vitesse maximale, appliquée avant la mise à jour de position.
- [ ] Limites d'arène.
- [ ] Graphe de communication par portée, recalculé à chaque pas, sans auto-boucle.
- [ ] Pannes : sélection aléatoire à mi-épisode, selon la règle décidée en phase 0.
- [ ] Interface d'épisode unique (cible, positions initiales, graine, contrôleur) utilisable par tous les contrôleurs.
- [ ] Enregistrement des positions à chaque pas.

**Tâches formes**
- [ ] Rendu 128×128 (géométries tracées, caractères par police) puis binarisation au milieu de la plage de gris.
- [ ] Échantillonnage par point le plus éloigné, graine contrôlée.
- [ ] Normalisation dans un cadre commun, à l'échelle du rayon de cible.
- [ ] Jeu : 7 géométries, 26 lettres, 10 chiffres, nombres de 9 à 999.
- [ ] Deux résolutions (20 et 50 points) ; envisager 80 à 100 pour les nombres.
- [ ] Index des formes + image d'origine + planche de contrôle.
- [ ] Découpage entraînement / validation / test déclaré et figé (formes réservées pour la généralisation).

**Tests obligatoires**
- [ ] Portée de communication respectée.
- [ ] Vitesse toujours bornée.
- [ ] Robot isolé géré sans erreur.
- [ ] Robot en panne immobile et hors des collisions.
- [ ] Même graine = même épisode, à l'identique.
- [ ] **Test anti-doublon** : le nombre de points distincts d'une cible doit être égal au nombre de points demandé, ou l'écart doit être explicitement documenté.
- [ ] Normalisation : toutes les formes tiennent dans le cadre commun.
- [ ] Présence de chaque forme dans l'index.

**Livrables**
- `src/simulator/`, `src/shapes/`, `tests/`.
- `outputs/shapes/index.csv` : une ligne par forme (nom, type, résolution, découpage, chemin de l'image).
- `outputs/shapes/sheet_<type>.png` : planche de contrôle visuelle par type de forme.
- `docs/decisions/02-simulator.md` : règles du simulateur (dont la règle des robots retirés).

**Résultats attendus**
- Planches de contrôle montrant les 7 géométries, 26 lettres, 10 chiffres et un échantillon de nombres, chaque forme reconnaissable à l'œil.
- Un tableau du nombre de formes par découpage (entraînement, validation, test).
- Une figure d'un épisode avec un contrôleur trivial sur 20, 40 et 100 robots.

**Critère d'acceptation** : un contrôleur trivial tourne sur 20, 40 et 100 robots, de façon reproductible (deux exécutions donnent les mêmes positions) ; toutes les planches ont été inspectées ; tous les tests passent.

**Points de vigilance** : une seule source d'aléa par épisode ; la cible ne doit jamais avoir de doublons cachés ; les robots retirés suivent une règle unique, testée.

---

### Phase 2 — Métriques [x] (validée 2026-10-21)

**État (2026-10-21)** — `src/metrics/` (base, chamfer, collisions, regularity, success, connectivity, suite) + `config/metrics.yaml` (seuil 0,15, rayon collision 0,1, référence régularité « auto ») ; définitions déjà figées dans `docs/decisions/06-evaluation.md`.
- Classes : `Metric` abstrait (fonction pure de positions/masque/cible enregistrés), `ChamferDistance` (moyenne non au carré, aller+retour, normalisée par le rayon cible), `SuccessCriterion` (franchissement **stabilisé** : premier k avec distance ≤ seuil pour tout t ≥ k), `CollisionRate` (paires actives / paires-pas), `SpreadRegularity` (écart-type des distances au plus proche voisin, rapporté à l'espacement idéal), `GraphConnectivity` (graphe recalculé à chaque pas depuis les positions, composantes connexes), `MetricSuite`/`MetricReport`.
- Validation de tests consignée précédemment (`tests/test_metrics.py`), revérifiée par la suite automatique sans entraînement : valeurs exactes calculées à la main (ensembles identiques → 0 ; décalage connu → 0,6), paire en collision détectée, cas limites du seuil (sous/égal/au-dessus), cohérence succès↔courbe (franchissement instantané non stable refusé), recalcul depuis `.npy` identique au bit près, invariance d'échelle (comparabilité), exclusions des robots retirés.
- Livrables consignés précédemment : `figures/example_distance_curve.png` (courbes 20/40/100 + seuil tracé), table de comparabilité `outputs/metrics/comparability.csv` + `summary.json` (`python -m src.reports.metrics_demo`) : même graine, distances finales 0,093 / 0,103 / 0,133 — même ordre de grandeur, succès aux pas 11/12/10. Pas de nouvelle exécution dans ce tour.
- Nommage actuel : `src/metrics/`, `tests/test_metrics.py`, `example_distance_curve.png` ; valeurs obtenues par `MetricSuite.compute(...).values`.

**Objectif** : des mesures claires, comparables entre tailles d'essaim, recalculables depuis les seules positions enregistrées.

**Classes (POO)** : `Metric` (abstraite) avec `ChamferDistance`, `CollisionRate`, `SpreadRegularity`, `SuccessCriterion`, `GraphConnectivity` ; `MetricSuite` pour les regrouper.

**Tâches**
- [ ] Distance de Chamfer symétrique (préciser si moyenne de distances ou distances au carré) ; normalisation décidée en phase 0 pour comparer 20, 40 et 100 robots.
- [ ] Succès : distance finale sous le seuil, **ou** franchissement du seuil à un moment donné (un seul choix, écrit dans `docs/decisions/06-evaluation.md`).
- [ ] Pas de convergence, selon cette définition unique.
- [ ] Taux de collisions (normalisé par le nombre de paires et la durée), mesuré sur les robots actifs seulement.
- [ ] Régularité de répartition (valeur absolue).
- [ ] Connectivité du graphe pendant l'épisode (nombre de composantes connexes).
- [ ] Courbe de distance au cours du temps.

**Tests obligatoires**
- [ ] Valeurs exactes sur des cas simples (deux ensembles identiques : distance nulle ; décalage connu : distance connue).
- [ ] Paire de robots en collision détectée.
- [ ] Cas limites autour du seuil (juste en dessous, égal, juste au-dessus).
- [ ] **Test de cohérence du succès** : un épisode dont la distance initiale est au-dessus du seuil ne peut pas être déclaré réussi au pas 1 ; le pas de convergence est cohérent avec la courbe de distance.
- [ ] Les mesures donnent le même résultat quand on recalcule à partir des positions enregistrées.

**Livrables**
- `src/metrics/`, `tests/test_metrics.py`.
- `docs/decisions/06-evaluation.md` : définitions, formules en mots, seuil et justification.
- `figures/example_distance_curve.png` : courbe de distance d'un épisode avec le seuil tracé.

**Résultats attendus**
- Pour un épisode donné : distance finale, succès (oui/non), pas de convergence, taux de collisions, régularité, nombre de composantes connexes à chaque pas.
- Une courbe de distance qui décroît, croise le seuil au pas de convergence annoncé, et s'y stabilise.
- Un tableau de vérification : mêmes épisodes évalués à 20, 40 et 100 robots, distances du même ordre de grandeur (preuve de comparabilité).

**Critère d'acceptation** : toutes les mesures sont recalculables à partir des positions enregistrées ; tests sur cas simples et cas limites passent ; la comparabilité entre tailles est démontrée sur au moins un exemple.

**Points de vigilance** : le seuil doit avoir un sens physique (par exemple une fraction du rayon de la forme) ; ne jamais changer la définition du succès après avoir vu les résultats.

---

## Bloc 2 — Contrôleurs (Noyau)

### Phase 3 — Références non apprises [x] (validée 2026-10-21)

**État (2026-10-21)** — `src/controllers/flocking.py`, `src/controllers/manual_shape.py`, réglage et évaluation par scripts (`python -m src.reports.tune_baselines`, `python -m src.reports.baselines_demo`), tout paramétré dans `config/baselines.yaml`.
- Classes : `FlockingController` (cohésion–répulsion sur voisins du graphe, ignore la consigne de forme ; robot isolé → commande nulle, pas de fuite d'information globale), `ManualShapeController` (sous-classe du trivial : attraction au point de contour le plus proche + répulsion), même interface `Controller.command` que le GNN, cible dans `Mission.target`.
- Réglage consigné précédemment : grille croisée 3×3×3 = **27 essais par règle, budget fixé à l'avance** dans la config, critère = distance finale sur cercle à 40 robots (graine 7) ; journal `outputs/references/tuning/<rule>/log.csv`. Valeurs retenues dans `config/baselines.yaml` (`rules`).
- Scores consignés précédemment (seuil 0,15) : flocking échoue partout par construction (d = 0,41–0,50, nuage sans structure) ; manuelle réussit le cercle à 20/40/100 (d = 0,079/0,066/0,043) et passe de justesse sur A, K, 12, 121 à 40 robots (d = 0,109–0,116) — concavités = point faible documenté. Pas de nouvelle évaluation dans ce tour.
- Livrables consignés précédemment : `outputs/references/scores.csv`, `summary.json`, `failure_notes.md`, 10 dossiers de run standard §0.4 ; figures `figures/baselines_circle.png` et `figures/baselines_hard_shapes.png`.
- Validation de tests consignée précédemment (`tests/test_baselines.py`), revérifiée par la suite automatique sans entraînement : budget respecté, valeurs réglées dans la grille, interface commune (forme, fini, déterminisme, commandes bornées par le simulateur), groupement indépendant de la mission, robot isolé immobile, cercle formé à toutes tailles.

**Objectif** : fixer un plancher de performance honnête, sans apprentissage.

**Classes (POO)** : `Controller` (abstraite, contrat commun) avec `FlockingController` et `ManualShapeController`.

**Tâches**
- [ ] Règle de regroupement (cohésion–répulsion), sans consigne de forme.
- [ ] Règle manuelle avec consigne de forme (attraction vers le contour + répulsion).
- [ ] Même interface que le futur GNN ; mêmes tests d'interface.
- [ ] Réglage des paramètres des règles avec un budget défini (nombre d'essais fixé à l'avance).
- [ ] Évaluation de la règle manuelle sur cercle, puis sur lettres et nombres (baseline pour les formes difficiles).

**Livrables**
- `src/controllers/flocking.py`, `src/controllers/manual_shape.py`.
- `outputs/references/<rule>/` avec le dossier de run standard (section 0.4).
- Tableau des scores des deux règles sur le cercle (20, 40, 100 robots) et sur un échantillon de lettres et de nombres.

**Résultats attendus**
- Une figure de l'état final du regroupement (amas sans forme) et de la règle manuelle sur le cercle (contour plus ou moins formé).
- Un tableau : distance finale, taux de succès, taux de collisions, connectivité, pour chaque règle et chaque taille d'essaim.
- Une note explicite sur les échecs (par exemple : la règle manuelle forme un cercle mais échoue sur les lettres à concavités).

**Critère d'acceptation** : les deux règles forment un cercle ou échouent de façon documentée ; leurs scores sont enregistrés et serviront de plancher dans toutes les comparaisons.

**Points de vigilance** : une baseline mal réglée rend le GNN trop flatteur ; documenter l'effort de réglage consacré.

---

### Phase 4 — Contrôleur GNN [x] (validée 2026-10-21)

**État (2026-10-21)** — `src/models/` (encoder, aggregator, message, decoder, gnn) + `src/controllers/gnn.py`. Hyperparamètres dans `config/model.yaml`. Messages écrits à la main (aucune bibliothèque de GNN).
- Classes : `Encoder` (vitesse + embedding identifiant de forme [+ vecteur contour optionnel]), `MessageLayer` (message par arête `[h_i, h_j, relative_position]` + agrégation + mise à jour résiduelle), `Aggregator` abstrait + `MeanAggregator` (défaut) + `NormalizedSumAggregator`/`AttentionAggregator` (bonus), `Decoder` borné (`v_max·tanh` normalisé), `SwarmGNN`, `GNNController`, `NoCommGNNController` (même modèle, graphe vidé → agrégat nul, taille identique).
- Clé `model.message_rounds` configurable (1/2/3) : nombres consignés précédemment de 31 490 / 56 450 / 81 410 paramètres, **indépendants du nombre de robots** (poids partagés), accessibles par `parameter_count`. Voisinage vide = agrégat nul, robot isolé géré ; modèle exposé par `GNNController.model`.
- Validation de tests consignée précédemment (`tests/test_gnn.py`), revérifiée par la suite automatique sans entraînement : permutation, translation, tailles 20/40/100, robot isolé, sans-communication indépendant des voisins, conditionnement par l'identifiant de forme, commandes bornées.
- Livrables consignés précédemment : `outputs/gnn/architecture.txt`, démo modèle non entraîné sur 20/40/100 (`outputs/gnn/untrained_demo/`). Critère d'acceptation précédemment validé, non revérifié dans ce tour.



**Objectif** : un contrôleur décentralisé, indépendant du nombre de robots, avec sa variante sans communication pour isoler l'effet de la communication.

**Classes (POO)** : `GNNController` et `NoCommGNNController` (héritent de `Controller`), composés de `Encoder`, `MessageLayer`, `Aggregator` (moyenne ; somme normalisée et attention en bonus, sous-classes de `Aggregator`) et `Decoder`.

**Tâches**
- [ ] Encodeur de l'observation (vitesse, identifiant de forme).
- [ ] Messages par arête (états des deux robots + position relative), agrégation par moyenne, voisinage vide = agrégat nul.
- [ ] Mise à jour, nombre de tours configurable (1, 2, 3), décodeur borné.
- [ ] Variante sans communication : mêmes entrées, pas de messages, taille de modèle comparable.
- [ ] Messages écrits à la main (pas de bibliothèque de GNN).
- [ ] Bonus : agrégation par somme normalisée ou attention.

**Tests obligatoires**
- [ ] Invariance par permutation : mélanger l'ordre des robots ne change pas les commandes (à l'ordre près).
- [ ] Invariance par translation : décaler tous les robots ne change pas les commandes.
- [ ] Tailles variables : le même modèle tourne sur 20, 40 et 100 robots.
- [ ] Robot isolé : pas d'erreur, commande valide.
- [ ] Communication désactivée : la variante sans communication ne dépend pas des voisins.
- [ ] Conditionnement par la consigne : deux identifiants de forme différents donnent des commandes différentes.
- [ ] Commandes toujours bornées.

**Livrables**
- `src/controllers/gnn.py`, `tests/test_gnn.py`.
- `outputs/gnn/architecture.txt` : liste des couches et nombre de paramètres pour chaque variante (1, 2, 3 tours, sans communication).

**Résultats attendus**
- Tableau : nombre de paramètres de chaque variante (même valeur quel que soit le nombre de robots).
- Rapport de tests : tous verts.
- Une exécution de démonstration (modèle non entraîné) sur 20, 40 et 100 robots, sans erreur.

**Critère d'acceptation** : le nombre de paramètres est indépendant du nombre de robots ; tous les tests passent.

**Points de vigilance** : la variante sans communication doit avoir une taille comparable, sinon la comparaison n'est pas équitable.

---

## Bloc 3 — Entraînement et évaluation (Noyau)

### Phase 5 — Entraînement sur le cercle [~] (implémentation préparée, acceptation empirique non vérifiée)

**État (2026-10-21)** — `src/training/train.py`, `rollout.py`, `losses.py`, `trainer.py`, `checkpoint.py`, `logger.py` dans `src/training/`, et `src/reports/training_summary.py` ; configuration `config/training.yaml`.
- Préparation du pipeline : configuration effective avec sections `model`, `training`, `simulator`, `metrics`, `shapes`, cibles et graines sauvegardées avant optimisation ; refus d'écraser un run, statut conservé en cas d'échec.
- `validate_splits` impose `training_target` dans `train` et `validation_target` dans `val`, jamais `test`. `profiles.<profile>.validation_seeds` pilote explicitement les épisodes de validation ; la géométrie est recalculée pendant le rollout.
- Clés actuelles : `profiles`, `optimizer`, `losses.weights` (`shape`, `coverage`, `collisions`, `motion`), `variants` et `message_rounds`. Profil `standard` : graines 11 à 15 ; variantes `gnn_1_round`, `gnn_2_rounds`, `gnn_3_rounds`, `no_comm`.
- Tests synthétiques et simulés prévus dans `tests/test_training.py` ; aucun entraînement local, aucun profil `smoke` réellement exécuté et aucun checkpoint ou résultat actuel attesté par cette mise à jour.

**Objectif** : obtenir, pour chaque contrôleur appris, des modèles entraînés de façon propre, avec plusieurs graines.

**Classes (POO)** : `LossTerm` (abstraite) avec `ShapeLoss`, `CollisionLoss`, `MotionLoss`, `CoverageLoss` ; `CompositeLoss`, `DifferentiableSimulator`, `Trainer`, `CheckpointManager`, `TrainingLogger`.

**Tâches**
- [~] Simulateur différentiable (tout en tenseurs).
- [~] Perte à quatre termes : forme, collisions, mouvement, couverture ; poids dans la configuration.
- [~] Boucle d'optimisation (Adam), lots d'épisodes, sauvegarde du meilleur état et du dernier.
- [~] Journal des pertes par itération (total et par terme).
- [~] Profil de vérification rapide (`smoke`) configuré ; exécution d'entraînement local interdite.
- [ ] Entraîner chaque variante (GNN 1, 2, 3 tours et sans communication) sur le cercle avec au moins 5 graines, ultérieurement dans un environnement autorisé.
- [~] **Budget de réglage identique** préparé pour tous les contrôleurs appris : même nombre d'essais d'hyperparamètres ; budget de calcul réel restant à documenter.
- [~] Utiliser le découpage validation pour choisir le meilleur état, jamais le test.

**Tests obligatoires**
- [ ] Pertes proches de zéro sur des cas où la réponse est connue.
- [ ] Pas d'auto-boucle dans le graphe d'entraînement.
- [ ] Cohérence de la cible (nombre de points, absence de doublons).
- [ ] Le profil `smoke` tourne de bout en bout : preuve empirique différée, sans lancement local ; les tests simulés du pipeline ne suffisent pas à la valider.

**Livrables attendus par variante et par graine (non produits dans ce tour)**
- `outputs/training/<variant>/seed_<n>/` contenant :
  - `best.pt` et `last.pt` (poids du modèle) ;
  - `losses.csv` : une ligne par itération (`iteration`, `total`, `shape`, `coverage`, `collisions`, `motion`) ;
  - `validation.csv`, `validation_targets.csv`, `validation_episodes.csv` : métriques globales, par cible et par graine de validation ;
  - `effective_config.yaml`, `seeds.json`, `version.txt`, `targets.json`, `status.json`, `summary.json`.
- `outputs/training/budget.md` et `summary.csv`, `figures/training_losses_<variant>.png`, `figures/training_circle_final.png` : synthèse préparée dans `src/reports/training_summary.py`, fondée sur les configurations sauvegardées et toutes les graines complètes.

**Résultats attendus (ce que l'on doit pouvoir montrer à la fin de l'entraînement)**
1. **Courbes de perte** : pour chaque variante, une figure avec les 5 graines superposées, perte totale et chaque terme séparé. Elles décroissent sur toutes les graines, puis se stabilisent (pas de divergence, pas de remontée forte).
2. **Tableau de synthèse de l'entraînement**, une ligne par variante, avec moyenne et écart-type sur les 5 graines :
   - perte finale totale et par terme ;
   - distance de forme finale sur la validation ;
   - taux de succès sur la validation ;
   - taux de collisions ;
   - nombre d'itérations jusqu'au meilleur état ;
   - temps d'entraînement.
3. **Figure qualitative du cercle formé** : état final d'un épisode de validation pour chaque variante (une graine représentative, avec la cible superposée) ; le cercle est visuellement correct et la couverture est acceptable (pas d'amas, pas de trous importants).
4. **Dispersion entre graines** : une phrase chiffrée disant si l'entraînement est stable (par exemple écart-type de la distance finale entre graines).
5. **Trace du budget de réglage** : même effort affiché pour toutes les variantes.

**Critère d'acceptation** : la perte décroît sur toutes les graines ; le cercle formé est visuellement correct ; la couverture est acceptable ; le budget de réglage est identique et documenté.

**Points de vigilance** : ne pas ne garder que la meilleure graine ; toujours rapporter les 5. Si une graine échoue, la garder et l'expliquer plutôt que l'écarter.

**Résultats du prototype — historique, non recevable pour la réécriture (consigné le 2026-10-21)**
- Les anciennes mesures sur cinq graines 41 à 45, les tableaux du cercle spécialisé et les ablations de couverture/transport provenaient du premier prototype. Ils mêlaient défaut de consigne figée, réglages différents et ancien seuil de succès ; ils ne prouvent ni convergence ni supériorité du modèle actuel.
- Les anciens modules sous `src/swarm_gnn/`, configurations `config/training-circle.yaml`, `config/evaluation-pilot.yaml`, `config/curriculum-complex*.yaml` et commandes du prototype sont périmés. Leurs sorties mentionnées auparavant ne sont ni attestées présentes ni compatibles avec les artefacts actuels.
- Les anciennes fonctions de transport et leur journal ne décrivent pas les quatre pertes actuelles. L'interdiction d'affectation centralisée en entrée demeure ; toute ablation empirique doit être refaite, avec provenance et budget équivalent, dans un environnement autorisé.
- Aucune courbe, aucun score, aucun budget mesuré ni checkpoint de la réécriture ne sont validés ici. Le critère d'acceptation de la phase 5 reste intégralement en attente.

---

### Phase 6 — Harnais d'évaluation [~] (implémentation préparée, acceptation empirique non vérifiée)

**État (2026-10-21)** — `src/evaluation/campaign.py` (`Campaign`), `scenario.py`, `evaluator.py`, `factory.py`, `results.py`, `builders.py` dans `src/evaluation/` ; configuration `config/evaluation.yaml`.
- Snapshot figé, graines, dépendances et empreintes des checkpoints ; chargement du modèle depuis sa configuration effective sauvegardée, sans reprendre les paramètres globaux du modèle. Toutes les graines requises sont exigées ; artefacts partiels et dossiers de sortie déjà occupés refusés.
- Agrégats conditionnels à deux niveaux : épisodes à l'intérieur d'une graine puis variation entre graines d'entraînement. `PairedComparison` indexe les mêmes conditions et graines d'épisode et apparie les modèles appris à graine d'entraînement identique.
- Archives `.npz` en pleine précision, avec positions, masque actif, cible et rayon, pour rejouer les métriques ; tableaux et graphiques incluent les runs appris.
- Tests synthétiques dans `tests/test_evaluation.py` et `tests/test_studies.py`, inclus dans la suite complète sans entraînement ; aucune campagne formelle exécutée. La commande `make campaign` est préparée, non lancée.

**Objectif** : une procédure d'évaluation automatique, équitable et reproductible, utilisée pour tous les contrôleurs.

**Classes (POO)** : `Scenario`, `Campaign`, `Evaluator`, `ResultsTable`, `PairedComparison`, `FigureBuilder`, `TableBuilder`.

**Tâches**
- [~] Scénarios : essaims de 20, 40, 100 robots ; retraits de 5, 10 et 20 % (et condition sans panne) ; durée configurée avec marge, à confirmer par les temps de convergence empiriques.
- [~] Mêmes positions initiales et mêmes pannes réutilisées pour tous les contrôleurs (graines d'évaluation figées).
- [~] 50 répétitions par condition configurées, non exécutées.
- [~] Sortie brute préparée : une ligne par épisode.
- [~] Agrégats : moyenne, écart-type, intervalle de confiance.
- [~] Rapport en deux niveaux : variation entre graines d'entraînement, puis variation entre épisodes à l'intérieur d'une graine.
- [~] Comparaisons appariées (mêmes épisodes et même graine d'entraînement pour les modèles appris).
- [~] Génération automatique des tableaux et figures depuis les fichiers bruts.

**Livrables attendus (non produits dans ce tour)**
- `outputs/evaluation/episodes.csv` et `outputs/evaluation/<controller>/episodes.csv` : colonnes `controller`, `training_seed`, `episode_seed`, `n_robots`, `failure_fraction`, `shape`, `shape_id`, `final_distance`, `success`, `convergence_step`, `collision_rate`, `normalized_regularity`, `final_components`, `step`.
- `outputs/evaluation/<controller>/seed_<n>/episodes.csv` et `positions/<condition>/seed_<episode_seed>.npz` pour les modèles appris ; sans dossier de graine d'entraînement pour les références non apprises.
- `effective_config.yaml`, `seeds.json`, `version.txt`, `summary.json` ; tableaux d'agrégats et figures `distance_curves.png`, `final_distance_box.png`, `success_by_size.png` sous `outputs/evaluation/figures/`.
- Une commande unique préparée : `make campaign` ; génération dans `src/evaluation/builders.py` et orchestration dans `src/evaluation/campaign.py`.

**Résultats attendus**
- Un fichier brut complet : nombre de lignes = conditions × 50 × somme du nombre de versions de chaque contrôleur (une version par baseline, toutes les graines d'entraînement pour chaque variante apprise), vérifié par `expected_row_count`.
- Un tableau d'agrégats par condition (moyenne, écart-type, intervalle de confiance) et par contrôleur.
- Des figures standard : distance en fonction du temps (courbe moyenne avec bande de dispersion), distribution des distances finales (boîtes), taux de succès par taille d'essaim.
- Un test de reproductibilité : relancer la campagne avec les mêmes graines donne les mêmes fichiers.

**Critère d'acceptation** : une seule commande exécute la campagne sur un contrôleur et produit tableaux et figures ; le nombre de lignes brutes est conforme ; la reproductibilité est vérifiée.

**Points de vigilance** : ne jamais régler un contrôleur sur les épisodes d'évaluation ; durée d'épisode trop courte = succès sous-estimé.

---

### Phase 7 — Comparaison à information équivalente [~] (implémentation préparée, acceptation empirique non vérifiée)

**État (2026-10-21)** — `CommunicationComparison` dans `src/reports/communication.py`, configuration `config/comparison.yaml`, commande `make comparison` préparée, non lancée.
- Lecture de campagnes et configurations figées ; contrôles d'équivalence de budget, cibles, information, profils, dynamique et graines pour les variantes apprises.
- `information.csv` distingue explicitement la règle manuelle (`nearest_displacement`) du GNN (`nearest_unit_direction`) : la baseline n'a pas exactement la même information géométrique ; l'inférence à consigne égale concerne d'abord les variantes GNN.
- Gains appariés par condition et à mêmes graines ; intervalles de confiance calculés **entre graines d'entraînement**, après agrégation des épisodes dans chaque graine, sans pseudo-réplication.
- Tests synthétiques dans `tests/test_studies.py` ; aucune comparaison empirique ni conclusion chiffrée actuelle validée.

**Objectif** : répondre à la question centrale : la communication apporte-t-elle quelque chose, à consigne égale ?

**Classes (POO)** : `CommunicationComparison` orchestre les rapports depuis les campagnes figées ; aucune nouvelle classe de contrôleur de base. Ajouter les agrégations bonus comme nouvelles sous-classes de `Aggregator`, sans modifier le reste (principe ouvert/fermé).

**Tâches**
- [ ] Évaluer : regroupement, règle manuelle avec consigne, GNN sans communication, GNN 1, 2 et 3 tours (campagnes réelles encore nécessaires).
- [~] Même budget d'entraînement et de réglage pour les contrôleurs appris : contrôles préparés, budget réel à confirmer.
- [~] Un tableau par condition et un tableau d'information indiquant les entrées de chaque contrôleur (consigne, voisins, vitesse).
- [~] Différences appariées entre GNN avec communication et GNN sans communication, avec intervalle de confiance entre graines d'entraînement.
- [ ] Option : somme normalisée, attention, agrégation tronquée.

**Livrables attendus (non produits dans ce tour)**
- `outputs/comparison/table_<condition>.csv`, `information.csv`, `paired_gains.csv`, `summary.json` et `conclusion.txt` ; version mise en forme pour le mémoire.
- `outputs/comparison/gain_<condition>.png` : gains avec incertitude entre graines.
- `docs/decisions/07-conclusion-communication.md` : réponse écrite à la question de recherche après validation empirique, avec les chiffres.

**Résultats attendus**
- Un tableau principal : lignes = contrôleurs, colonnes = distance finale, succès, collisions, connectivité, avec écarts-types.
- Un tableau d'information : pour chaque contrôleur, ce qu'il reçoit en entrée (preuve d'équité).
- Une figure : gain apporté par un, deux, trois tours de communication par rapport à l'absence de communication (avec dispersion).
- Une réponse explicite : « la communication améliore (ou non) la distance finale de X, intervalle de confiance [a, b] », y compris si le résultat est nul. Un résultat négatif honnêtement rapporté est acceptable.

**Critère d'acceptation** : on peut répondre « la communication apporte-t-elle quelque chose à consigne égale ? » avec des chiffres et leur incertitude.

**Points de vigilance** : éviter de conclure sur une différence plus petite que la dispersion entre graines.

---

### Phase 8 — Généralisation à la taille de l'essaim [~] (implémentation préparée, acceptation empirique non vérifiée)

**État (2026-10-21)** — `SizeTransferStudy` dans `src/evaluation/size_transfer.py`, configuration `config/size-transfer.yaml`, commande `make size-transfer` préparée, non lancée.
- Évaluation de modèles préalablement entraînés, gelés et rangés par taille ; contrôle de la taille d'entraînement dans la configuration effective de chaque checkpoint, sans optimisation implicite.
- Tailles 20, 40, 100 en entraînement et évaluation ; tableaux croisés, dégradation relative et courbes avec intervalles de confiance entre graines, par condition de forme et de panne.
- Clés actuelles : `training_sizes`, `evaluation_sizes`, `checkpoint_roots`, `controller`, `evaluation_config`, `output`. Les racines `outputs/training/size_<n>/` sont des chemins configurés, pas des checkpoints attestés existants.
- Tests synthétiques dans `tests/test_studies.py` ; aucun entraînement par taille ni étude empirique exécutés dans ce tour.

**Objectif** : démontrer que le même modèle fonctionne quand le nombre de robots change.

**Classes (POO)** : réutiliser `Scenario`, `Campaign` et les contrats d'entraînement de `Trainer` ; `SizeTransferStudy` orchestre uniquement l'évaluation des modèles gelés aux tailles 20, 40 et 100, sans appeler l'entraînement.

**Tâches**
- [ ] Entraîner sur 20 robots dans un environnement autorisé ; évaluer sur 40 et 100 sans réentraîner (poids gelés).
- [ ] Entraîner directement sur 40 et sur 100, pour comparer, sans lancement local.
- [~] Mêmes épisodes, mêmes métriques que la phase 6 : orchestration préparée, preuve empirique en attente.

**Livrables attendus (non produits dans ce tour)**
- `outputs/size_transfer/trained_on_<n>/` : une campagne par taille d'entraînement avec conditions d'évaluation indexées.
- `outputs/size_transfer/cross_size.csv` : tableau croisé (taille d'entraînement, taille d'évaluation, fraction de panne, forme).
- `outputs/size_transfer/summary.json`, `limitations.txt`, `transfer_failure<percentage>_<shape>.png` : dégradation relative, limites et courbes de transfert.

**Résultats attendus**
- Une courbe de performance (distance finale) en fonction de la taille d'essaim, avec dispersion, pour le modèle entraîné sur 20 et pour les modèles entraînés directement.
- Un chiffre de dégradation : perte de performance relative du transfert 20 vers 100, comparée au modèle spécialisé.
- Une conclusion : le transfert est bon, partiel ou mauvais, avec la raison probable.

**Critère d'acceptation** : courbe de performance en fonction de la taille, avec dispersion.

**Points de vigilance** : à nombre de robots différent, la densité change (portée de communication identique) ; noter si la dégradation vient de la densité plutôt que du modèle.

**Jalon de sécurité** : à la fin de cette phase, le mémoire est défendable (cercle, baselines, communication contre absence de communication, généralisation à la taille).

---

## Bloc 4 — Formes complexes (Important)

### Phase 9 — Curriculum de formes [~] (implémentation préparée, validation empirique différée)

**État de la réécriture (2026-10-21)** — `src/training/curriculum.py`, `sampling.py`, `src/reports/curriculum.py` et `config/curriculum.yaml`. Niveaux cumulatifs géométries → chiffres → lettres → nombres, cibles test exclues ; échantillonnage cyclique équilibré, transfert du meilleur checkpoint avec contrôle d'architecture et d'empreinte, remise à zéro explicite de l'optimiseur à chaque niveau. Mesures des pertes et distances par cible sur toutes les graines de validation ; oubli apparié au niveau précédent, carte de chaleur et dispersion entre graines. Variantes identifiant seul et géométrie séparées. Tests de contrats et orchestration simulée ; aucun entraînement exécuté par l'agent.

**Objectif** : apprendre progressivement des formes de plus en plus complexes sans oublier les précédentes.

**Classes (POO) prévues** : `Curriculum` (liste ordonnée de niveaux), `CurriculumLevel`, `ForgettingMonitor` (mesure de la perte par cible).

**Tâches**
- [~] Niveaux : géométries, puis chiffres, puis lettres, puis nombres ; les cibles précédentes sont conservées dans chaque niveau.
- [~] Mesure de la perte **par cible** à chaque niveau, pour détecter l'oubli.
- [~] Cohérence des journaux : pertes pondérées vérifiées depuis les configurations sauvegardées, sans confondre meilleur état et dernière itération.
- [~] Orchestration de l'évaluation du modèle spécialisé cercle et du modèle du curriculum sur les mêmes épisodes ; résultats réels attendus.

**Livrables**
- `outputs/curriculum/level_<n>/` avec les dossiers de run standard.
- `outputs/curriculum/losses_by_target.csv` : une ligne par (niveau, cible, perte).
- Matrice « niveau d'entraînement × cible » des distances finales.

**Résultats attendus**
- Une carte de chaleur : lignes = niveaux, colonnes = cibles, valeur = distance finale ; les cibles anciennes restent à un niveau comparable après chaque nouveau niveau.
- Un tableau comparant modèle spécialisé et modèle du curriculum sur le cercle et sur les nouvelles formes.
- Un chiffre d'oubli : variation de la distance sur les anciennes cibles entre la fin d'un niveau et la fin du niveau suivant.

**Critère d'acceptation** : pas de dégradation notable des anciennes cibles ; résultats par forme disponibles.

**Points de vigilance** : le curriculum doit être terminé et empiriquement validé avant l'évaluation de la phase 10 ; l'exception d'implémentation seule de la section 0.2 ne lève pas cette dépendance expérimentale.

**Anecdotes du prototype — historique, non recevable pour la réécriture (consigné le 2026-10-21)**
- Les anciennes références `outputs/curriculum_full` (graines 43 à 47) et `outputs/recipes_sweep/sweep.csv` décrivaient des essais du prototype, avec consigne figée ou recettes et budgets non comparables. Leur présence actuelle n'est pas attestée ; leurs chiffres ne démontrent ni gain ni absence d'oubli pour la réécriture.
- L'ancienne annonce de campagne « en cours » liée à `config/curriculum-complex.yaml` et `outputs/curriculum_complex_geom/seed_59/` est périmée. Aucun module ou checkpoint de curriculum actuel ne peut être déduit de ces chemins historiques.
- Toute comparaison cercle spécialisé/curriculum doit être refaite après implémentation, avec mêmes épisodes appariés, configurations sauvegardées et toutes les graines requises, dans un environnement autorisé.

---

### Phase 10 — Formes complexes à 50 robots [~] (implémentation préparée, validation empirique différée)

**État de la réécriture (2026-10-21)** — `ComplexShapeStudy` dans `src/evaluation/shape_studies.py`, configuration `config/complex-shapes.yaml`. Réutilise les campagnes gelées, vérifie les checkpoints du niveau 4, évalue les formes train/val à 50 robots et rapporte séparément les variantes identifiant seul et géométrie. Comparaison avec la règle manuelle et le modèle cercle spécialisé sur les mêmes épisodes ; tableaux par type et par graine, exemples fixes favorables et défavorables. Coût algorithmique du curriculum consultable sans optimisation ; durée réelle à mesurer par l'utilisateur. Aucun résultat empirique actuel.

**Objectif** : évaluer les lettres et les nombres, face à la règle manuelle.

**Classes (POO)** : `ComplexShapeStudy`, `ShapeStudyReport` et contrats existants ; la variante géométrique configure l'encodeur partagé via `use_target_geometry`, sans copier le GNN.

**Tâches**
- [~] Configuration dédiée : 50 robots, 50 points par cible, 64 composantes, 3 tours, épisodes de 120 pas.
- [~] Variante avec vecteur vers le contour, **résultats séparés** des autres.
- [~] Comparaison préparée avec la règle manuelle de la phase 3 sur les mêmes formes et épisodes.
- [~] Budget algorithmique calculable avant entraînement avec `make curriculum-plan`.
- [ ] Mesurer la durée CPU d'un niveau lors des entraînements lancés par l'utilisateur.

**Livrables**
- `outputs/complex_shapes/` : dossiers de run, épisodes bruts, positions.
- Planche de résultats visuels : état final pour chaque lettre et chiffre testés (cible et essaim superposés).
- Tableau par type de forme (géométries, chiffres, lettres, nombres) : distance finale, succès, collisions, pour GNN et règle manuelle.

**Résultats attendus**
- Une planche lisible des lettres et chiffres avec exemples favorables **et** défavorables, graines fixes.
- Un tableau GNN contre règle manuelle par type de forme, avec dispersion.
- Une liste des formes qui échouent, avec une explication (concavités, traits fins, formes proches).

**Critère d'acceptation** : lettres et nombres évalués visuellement et chiffrés, GNN contre règle manuelle.

**Points de vigilance** : cible avec assez de points distincts (au moins autant que de robots, ou écart documenté).

**Anecdotes du prototype — historique, non recevable pour la réécriture (consigné le 2026-10-21)**
- Les anciennes références `config/evaluation-complex-shapes.yaml`, `src/swarm_gnn/evaluation/complex_shapes.py` et checkpoints annoncés sous `outputs/curriculum_complex_smoke/` ou `outputs/curriculum_complex_geom/` sont périmées ; leur présence n'est pas attestée. Les récits de correction de `main()` et de chargement d'index ne décrivent pas des fonctions actuelles ni une campagne terminée.
- Les anciennes commandes de formes complexes ne sont pas des commandes actuelles garanties. Aucune config « prête », aucun checkpoint de phase 9 et aucun résultat de phase 10 ne sont validés par ces anecdotes.
- Exigences conservées pour la future implémentation : identifiants de forme issus des cibles sauvegardées, distinction formes vues/réservées, aucune duplication de cible, même interface d'épisode pour les méthodes et planches montrant les meilleurs **et** les pires exemples. Les noms des nouveaux artefacts seront anglais et leurs explications françaises.

---

### Phase 11 — Généralisation aux formes et lisibilité [~] (implémentation préparée, réponses humaines attendues)

**État (2026-10-21)** — `HeldOutEvaluator` dans `src/evaluation/shape_studies.py`, configurations `config/shape-transfer.yaml`, `config/readability.yaml`, protocole `HumanReadabilityProtocol` dans `src/reports/readability.py`. Vérifie l'absence des cibles test dans les listes effectives d'entraînement et de validation de tous les checkpoints ; rapporte vues/réservées. Questionnaire aveugle depuis les positions archivées, vérité séparée et réponses CSV explicites ; précision et IC bootstrap entre participants, aucun score sans réponses. L'embedding inédit de la variante ID reste une limitation, pas une preuve zéro-shot. Campagne et collecte humaines différées.

**Objectif** : tester sur des formes jamais vues, et mesurer si l'essaim est lisible.

**Classes (POO)** : `ReadabilityProtocol` (abstraite), `HumanReadabilityProtocol` et `HeldOutEvaluator` ; méthode humaine retenue, sans classifieur supplémentaire.

**Tâches**
- [~] Orchestration de l'évaluation sur les formes réservées (jeu de test), poids gelés.
- [~] Protocole de lisibilité préparé : questionnaire humain aveugle et IC bootstrap entre participants.
- [~] Exemples favorables et défavorables, graines fixes.
- [ ] Exécuter la campagne et recueillir les réponses réelles ; aucun score sans collecte.

**Livrables attendus**
- `outputs/shape_transfer/` : épisodes bruts sur formes réservées.
- `outputs/readability/questionnaire.html`, `questionnaire.csv`, `images/`, `responses_template.csv` ; vérité, configuration figée et consignes privées dans `private/`.
- `outputs/readability/responses.csv` : réponses fournies par l'utilisateur ; synthèse et données par participant dans `outputs/readability_scores/`.

**Résultats attendus**
- Un tableau : distance finale sur formes vues contre formes réservées.
- Un chiffre de lisibilité (par exemple pourcentage de formes correctement reconnues), avec sa méthode et son incertitude.
- Une planche d'exemples favorables et défavorables.

**Critère d'acceptation** : chiffre de lisibilité rapporté avec sa méthode.

**Points de vigilance** : les formes réservées ne doivent jamais avoir servi pour choisir un modèle ou des paramètres.

---

## Bloc 5 — Robustesse (Important)

### Phase 12 — Robustesse [~] (implémentation préparée, validation empirique différée)

**État (2026-10-21)** — perturbations injectables dans `src/simulator/perturbations.py`, orchestration `src/evaluation/robustness.py` et paramètres dans `config/robustness.yaml`. Bruit d'observation, perturbation de communication, réduction de portée et panne sont rapportés selon leurs mécanismes configurés. Les positions physiques et les observations sont séparées, les graines initiales communes sont préservées. Archives, connectivité, reformation et points de rupture depuis les niveaux réellement testés ; aucun réglage sur ces épisodes, aucune campagne formelle exécutée.

**Objectif** : mesurer comment les performances se dégradent quand les conditions deviennent difficiles.

**Classes (POO)** : `Perturbation` (abstraite) avec `PositionNoise`, `CommunicationNoise`, `RangeReduction`, `FaultyRobot` ; chacune est appliquée par le simulateur sans le modifier.

**Tâches**
- [~] Bruit de position observée et suppression symétrique d'arêtes de communication (plusieurs niveaux croissants) ; pas de bruit de vecteur de message.
- [~] Réduction progressive de la portée physique de communication.
- [~] Robot défaillant immobile et exclu, retraits indépendants réunis avec les pannes ordinaires ; aucune transmission aberrante simulée.
- [~] Suivi du graphe réellement utilisé et durée de reformation : formation stable avant panne, suffixe stable strictement après panne.
- [~] Comparaison préparée GNN avec communication, GNN sans communication et règle manuelle.
- [ ] Exécuter la campagne sur les checkpoints gelés et examiner les courbes empiriques.

**Livrables**
- `outputs/robustness/<perturbation>/` : épisodes bruts par niveau de perturbation.
- Figures de dégradation par perturbation.

**Résultats attendus**
- Pour chaque perturbation : une courbe « niveau de perturbation / performance » avec bande de dispersion, un trait par contrôleur.
- Un tableau du point de rupture : niveau à partir duquel le succès tombe sous 50 %.
- Un chiffre de temps de reformation après panne (en pas de simulation).

**Critère d'acceptation** : courbes de dégradation avec dispersion.

**Points de vigilance** : si le temps manque, garder deux perturbations sur quatre (portée et panne).

---

## Bloc 6 — Démonstration (Bonus)

### Phase 13 — Démonstration Web (Bonus) [~] (API et navigateur testés, inspection visuelle différée)

**État (2026-10-21)** — `demo/server.py`, `demo/index.html`, `config/web-demo.yaml` : API en lecture seule, serveur sur 127.0.0.1, entrées bornées, catalogue chargé à la demande, checkpoints gelés et contrôle d'identité. La page ne fait que dessiner les positions reçues ; racine fonctionnelle, curseur et lecture de trajectoire. Tests API et parcours navigateur sur catalogue synthétique : déterminisme, données reçues, erreurs 422/503, GNN absent désactivé. Le catalogue historique incompatible a été conservé ; l'utilisateur le régénère avec `make shapes`. Capture native indisponible : aucune inspection visuelle ni capture pour le mémoire attestées.

**Classes (POO)** : `ApiServer`, `SimulationService` ; aucune règle de contrôle dans la page web.

**Tâches**
- [~] API locale en lecture seule : `/api/health`, `/api/targets`, `/api/simulate`.
- [~] Schéma JSON exposé par `/openapi.json`, rejouable par (forme, contrôleur, taille, graine).
- [~] Page de visualisation (cible + positions) sans règle de contrôle côté navigateur.
- [ ] Inspection visuelle et capture pour le mémoire avec un navigateur visible.

**Livrables**
- Dossier `demo/` avec le serveur et la page ; lancement : `make install`, `make shapes`, `make web`.
- Page `/` fonctionnelle, sans dépendance à une documentation interactive.

**Résultats attendus**
- Deux lancements identiques donnent les mêmes positions.
- Une capture d'écran de la page de visualisation pour le mémoire.

**Critère d'acceptation** : deux lancements identiques donnent les mêmes positions.

---

## Récapitulatif des livrables par phase

| Phase | Ce qu'on doit avoir à la fin | Statut |
|---|---|---|
| A. Audit | Tableau d'audit daté, sortie des tests, liste des défauts | [ ] |
| 1. Simulateur et formes | Simulateur testé, planches de formes, index, règles écrites | [x] validée 2026-10-21 |
| 2. Métriques | Définitions écrites, courbe de distance, comparabilité démontrée | [x] validée 2026-10-21 |
| 3. Références | Scores du regroupement et de la règle manuelle (plancher) | [x] validée 2026-10-21 |
| 4. GNN | Modèle testé, nombre de paramètres par variante | [x] validée 2026-10-21 |
| 5. Entraînement | 5 graines par variante, courbes de perte, tableau de synthèse, cercle formé | [~] implémentation préparée, acceptation empirique non vérifiée |
| 6. Évaluation | Fichiers bruts, agrégats, figures, commande unique | [~] implémentation préparée, acceptation empirique non vérifiée |
| 7. Communication | Tableau d'équité, gain apparié, réponse chiffrée | [~] implémentation préparée, acceptation empirique non vérifiée |
| 8. Taille | Courbe de performance selon la taille, comparaison avec modèles spécialisés | [~] implémentation préparée, acceptation empirique non vérifiée |
| 9. Curriculum | Carte de chaleur niveau par cible, mesure de l'oubli | [~] pipeline et rapports préparés, entraînements attendus |
| 10. Formes complexes | Planche lettres et chiffres, tableau GNN contre règle manuelle | [~] orchestration préparée, résultats attendus |
| 11. Lisibilité | Chiffre de lisibilité et méthode, formes réservées | [~] protocole préparé, campagne et réponses humaines attendues |
| 12. Robustesse | Courbes de dégradation, point de rupture | [~] perturbations et rapports préparés, campagne attendue |
| 13. Démo Web | API, page, reproductibilité | [~] API et parcours navigateur testés, inspection visuelle différée |

---

## Commandes à lancer par l'utilisateur

`make help` liste les commandes. Aucun entraînement n'est une dépendance automatique des tests, rapports ou études. Le profil `standard` comporte cinq graines ; les dossiers de run existants sont refusés, jamais écrasés. Les valeurs qui influencent les expériences sont dans les fichiers YAML versionnés.

Préparation, puis cercle et comparaison de communication :

```bash
make install
make shapes
make test
make training-plan
make train-all
make training-summary
make campaign
make comparison
```

Un seul modèle : `make train VARIANT=gnn_2_rounds SEED=11 PROFILE=standard`. Un entraînement rapide réel, uniquement lancé par l'utilisateur : `make train-smoke`.

Transfert de taille : `make train-sizes`, puis `make size-transfer`. Curriculum et formes complexes :

```bash
make curriculum-plan
make train-curriculum-all
make curriculum-summary
make complex-shapes
make curriculum-comparison
make shape-transfer
make readability
```

`curriculum-comparison` exige aussi les cinq checkpoints spécialisés `gnn_3_rounds` du cercle. Le questionnaire est dans `outputs/readability/` ; ne pas communiquer le sous-dossier `private/` aux participants. Copier les réponses réelles au format `participant_id,sample_id,predicted_shape` dans `outputs/readability/responses.csv`, puis lancer `make readability-score`. Sans réponses, aucun score n'est annoncé.

Après entraînement du cercle : `make robustness`. Démonstration locale : `make web` après `make shapes` ; le GNN n'est disponible qu'avec ses checkpoints compatibles. Sous Windows, les recettes utilisent PowerShell ; `PYTHON` peut désigner l'interpréteur de l'environnement virtuel.

`training-plan` et `curriculum-plan` sont des calculs de budget sans optimisation. Aucune durée CPU n'est promise : les temps réels sont journalisés pendant les entraînements. Les entrées et sorties propres à chaque étude sont configurées dans leurs YAML ; les phases empiriques doivent être examinées dans l'ordre du plan.

## Prochaines actions

1. Régénérer le catalogue avec `make shapes`, puis lancer les entraînements et examiner les courbes, toutes les graines et les budgets réels.
2. Exécuter les campagnes et vérifier leurs critères d'acceptation avant toute conclusion ; les tests automatiques sans optimisation ne les remplacent pas.
3. Collecter les réponses humaines pour la lisibilité et réaliser l'inspection visuelle de la démonstration.
4. Compléter ultérieurement l'audit de la phase A et vérifier les figures historiques avec les configurations et flux aléatoires actuels, avant de les utiliser comme preuves.
