# 14 — Revue explicite, intégration et entraînement par l'utilisateur

Date : 2026-10-28. Statut : préparation implémentée ; validation et entraînement utilisateur attendus (tâche 007).

## Traitement du défaut de qualité 113

Ne pas modifier les anciennes collectes ou leurs métriques. Créer une version dérivée excluant explicitement les démonstrations train en échec. La seule exclusion du jeu 005 est la graine 113 (deux collisions pendant le contrôle). Inscrire l'épisode exclu, ses métriques, la politique, le dossier source et l'empreinte du manifeste source dans le nouveau manifeste. Les archives conservées sont copiées sans modification et leurs empreintes restent identiques.

Refuser toute exclusion de validation ou test et refuser un train devenu vide : la sélection de démonstrations ne doit pas rendre l'évaluation artificiellement plus facile. Ce choix introduit une sélection des réussites dans les démonstrations train ; il ne prouve pas que le professeur est sûr. La campagne brute demeure la référence pour rapporter son taux d'échec. L'absence de collision dans une démonstration n'est pas une garantie de sûreté du GAT qui l'imite.

La version revue `outputs/imitation/circle-reviewed-task007` conserve 29 épisodes : 19 train, 5 validation, 5 test, soit 34 800 exemples. Le statut passed concerne les épisodes conservés, avec provenance explicite ; il ne requalifie pas le jeu brut. Une copie du manifeste est versionnée dans tasks/evidence/007-circle-reviewed-manifest.json.

## Exécution locale en simulation

LocalGATController construit pour chaque robot une étoile contenant uniquement son état et ceux de ses voisins directs, reconstruits depuis les observations relatives. Il restitue la sortie du nœud central. Le GAT possède un seul échange : les arêtes entre les autres voisins n'interviennent pas dans cette commande. Les tests comparent ce calcul à la sortie d'un graphe global masqué et vérifient leur équivalence numérique. Le modèle est en mode évaluation, sous inference_mode, sans accès aux robots distants et sans gradient pendant le contrôle.

## Préparation et apprentissage destinés à l'utilisateur

La commande train vérifie le manifeste revu, les empreintes des archives train/validation, le schéma, les limites des commandes et les tailles des graphes. Le test n'est pas chargé. Les lots contiennent des graphes entiers d'un pas ; mélanger les pas train est permis après la séparation par épisodes.

Configuration initiale : Adam, 200 époques maximum, lots de 32 graphes, taux 0,001, patience 20 et graine 42. La validation choisit la meilleure MSE et l'arrêt anticipé ; le test ne sert ni à choisir les poids ni à arrêter l'apprentissage. Les moyennes pondèrent les deux composantes de chaque robot de chaque graphe. Les échelles et l'architecture restent celles de 006. Le jeu actuel a un effectif fixe de 12 robots ; des épisodes de tailles différentes ne sont pas empilés en un même lot dans cette première version.

Le mode --dry-run prépare les données et vérifie les paramètres, sans créer d'optimiseur, sans mise à jour de poids et sans écrire de dossier d'entraînement. Il a été exécuté sur le jeu revu : 1 900 graphes train, 500 validation. Le mode réel n'a PAS été exécuté par l'agent, conformément à la demande utilisateur.

Lors de l'exécution par l'utilisateur, le dossier neuf contient run.json (paramètres, versions, empreinte de données, pertes et meilleure époque), des sauvegardes aux améliorations et best.pt. Une interruption laisse le statut training ; seule la fin renseigne trained. Une sortie existante est refusée. Reprendre automatiquement un apprentissage interrompu n'est pas implémenté ; utiliser un nouveau dossier. Les graines rendent le parcours contrôlable mais l'identité des résultats entre appareils/versions n'est pas garantie.

## Comparaison après entraînement

La commande compare exige un run terminé, contrôle l'empreinte des données et les métadonnées du modèle, puis charge uniquement les épisodes test. Rejouer professeur et GAT depuis exactement les mêmes positions et vitesses initiales, sous le même protocole. Exporter les quatre métriques, les trajectoires appariées et une MSE séparée sur les états du professeur. Cette MSE hors boucle ne remplace pas la réussite du contrôle en boucle fermée.

Les tests de raccordement utilisent des poids initiaux et une fiche de run fictive explicitement marquée comme fixture ; ils ne mesurent pas un modèle appris. La comparaison réelle attend les poids produits par l'utilisateur.

## Vérification et périmètre restant

222 tests ciblés réussis : 203 précédents et 19 nouveaux. Ils couvrent revue/provenance, refus d'exclure validation/test, chargement sans test pour l'apprentissage, lots, mode à blanc sans optimiseur, inférence locale, préservation des poids à la validation et comparaison appariée. La boucle d'optimisation réelle et la qualité du modèle appris restent à vérifier après l'exécution utilisateur. Aucun entraînement neuronal lancé par l'agent.

## Raccourcis Make demandés par l’utilisateur

Ajouter train-imitation pour un environnement Python activé et train-imitation-local pour la commande exacte Linux temporaire déjà fournie. Les chemins et dossiers sont surchargeables ; les anciennes cibles restent inchangées. Vérifier par make -n uniquement, sans exécuter l’entraînement.
