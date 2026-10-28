# 12 — Données d'imitation, initialisation et séparation

Date : 2026-10-28. Statut : validé par l’utilisateur (tâche 005).

## Domaine de collecte explicite

Le tirage historique permet les contacts initiaux. Pour la collecte, utiliser un nouveau tirage séquentiel avec rejet dans le même disque : rayon tiré avec racine carrée pour chaque proposition uniforme en aire ; accepter si la distance à chaque robot déjà placé est >= 0,4 m. Cette distance vaut la distance de sécurité configurée et dépasse strictement le diamètre physique. Le domaine est donc conditionné par l'absence de proximité initiale, sans prétendre que la configuration complète est uniforme parmi tous les arrangements possibles.

La source SeedManager utilise une clé dédiée `imitation_initial_positions`. Un budget global de 20 000 propositions empêche une boucle infinie ; une configuration trop dense provoque une erreur explicite, jamais un placement superposé en repli. Les anciens pilotes sont conservés, et leurs échecs initiaux ne sont pas requalifiés.

## Partition avant collecte et absence de fuite

config/imitation-data.yaml fixe les graines avant simulation : train 100–119, validation 200–204, test 300–304. Ces graines sont distinctes de celles utilisées pour régler le professeur (0–9) et le valider (10–29). Interdire graines répétées, ensembles vides et partitions autres que train/validation/test.

Un épisode entier appartient à un seul ensemble ; aucun mélange des pas ni des robots entre ensembles. Des empreintes des positions initiales et des trajectoires détectent les duplications exactes, même sous des graines distinctes. Elles ne détectent pas toutes les configurations géométriquement similaires : les états proches restent possibles dans un même domaine, sans partage d'une même simulation. La collecte échoue sur une duplication exacte et reste marquée incomplète.

## Format et cibles

Une archive NPZ sans pickle par épisode contient :

- node_features `(T,N,8)` : position x/y, vitesse x/y, centre x/y, rayon, effectif ; états AVANT la commande ;
- adjacency `(T,N,N)` : graphe de communication physique sans auto-boucles ;
- relative_neighbors `(T,N,N,4)` : position et vitesse relatives ; valeurs nulles hors voisinage ;
- targets `(T,N,2)` : accélérations du professeur bornées par la limite physique ; requested_targets conserve les demandes brutes pour diagnostic ;
- positions, velocities et times avec T+1 états, pour évaluation et reproductibilité.

Les composantes nodales correspondent à l'observation propre et à la consigne partagée, pas à une affectation de places. Le stockage global est une représentation d'apprentissage centralisée ; le futur GAT doit impérativement masquer les échanges selon adjacency pour préserver son exécution locale.

Choisir la commande bornée comme cible de MSE : elle est l'accélération effectivement fournie à l'intégrateur avant saturation de vitesse/contact, et évite d'apprendre des demandes inaccessibles. Les grandes demandes brutes restent exportées. La variable temps n'est pas une entrée du professeur circulaire et n'entre donc pas dans les huit caractéristiques. Les gains du professeur sont fixes pour ce jeu et inscrits au manifeste ; les mélanger exigerait un contrat supplémentaire.

## Traçabilité et qualité

Le manifeste enregistre schéma, paramètres physiques, unités, professeur, protocole, initialisation, graines, chemins, empreintes des fichiers, nombre d'exemples et résultats de chaque épisode. Son état reste incomplete jusqu'à la fin. Un dossier de sortie existant est refusé pour éviter toute réécriture accidentelle. Une reprise automatique n'est pas fournie : recommencer dans un nouveau dossier après correction.

Le chargement du manifeste refuse les collectes incomplètes, les incohérences de partition et les duplications. Le chargement d'une archive vérifie empreinte, ensemble attendu et métadonnées ; les chemins extérieurs au dossier sont refusés. Le chargement normal du manifeste refuse un jeu dont la qualité reste à examiner ; `allow_quality_failures=True` permet son inspection explicite, sans valider la qualité ni autoriser un entraînement.

Conserver tous les épisodes, y compris les échecs, sans remplacement par une autre graine ni sélection silencieuse des réussites. Cela conserve une estimation honnête de la qualité du professeur et permet un diagnostic ultérieur.

## Jeu pilote collecté sans entraînement

Commande exécutée : `python -m src.imitation.collect --output outputs/imitation/circle-task005`.
30 épisodes × 100 pas × 12 robots = 36 000 exemples : 24 000 train, 6 000 validation, 6 000 test.

| Ensemble | Réussite du professeur | Erreur finale moyenne | Collisions |
|---|---|---|---|
| train | 19/20 | 0,00589 m | 2 |
| validation | 5/5 | 0,00489 m | 0 |
| test | 5/5 | 0,00316 m | 0 |

La graine 113 contient deux collisions apparues pendant le mouvement malgré une initialisation sans chevauchement. Elle reste un échec avec temps de stabilisation null et est conservée dans train. `quality_status` vaut review_required. Aucun exemple n'est présenté comme prêt pour un entraînement sans examen de ce problème.

Les valeurs validation/test ci-dessus évaluent seulement le professeur : aucun modèle n'a été entraîné ni sélectionné sur ces ensembles. Ne pas utiliser l'ensemble de test pour régler le futur GAT.

Le manifeste est copié dans tasks/evidence/005-circle-manifest.json ; les archives sont dans outputs/imitation/circle-task005/ (ignoré par Git). La tâche 006 peut définir l'architecture et ses tests sans entraînement ; avant tout entraînement en tâche 007, traiter explicitement la qualité du professeur et refaire une collecte versionnée si les données changent.
