# 29 — Pilote partagé à 12, 24 et 64 robots

Date : 2026-10-29. L’utilisateur autorise l’entraînement multi-effectifs et demande une clarification de son coût : les calculs PyTorch sont locaux, sans appel API OpenAI ; les échanges de l’agent restent soumis à son quota.

## Domaine et partitions

Le pilote utilise un rendez-vous en anneau espacé de 0,6 m. Les départs sont proches de la propre place de rendez-vous, avec faible bruit et faible rotation. Ce domaine contrôlé facilite la faisabilité à 64, mais ne valide pas des départs dispersés. Les profils 12/24/64 emploient des arènes 10/18/32 m et des figures de largeur 6/12/28 m. Les phases de regroupement durent 8/15/30 s, le freinage 2 s, les échanges N² pas et la construction finale 40 s, à dt 0,1 s.

Train : carré, A, 1, départ 8100. Validation : triangle, B, 7, départ 8200. Test : rectangle, C, 3, départ 8300. Ces partitions sont disjointes par groupes géométriques. Les trois formes partagent un seul départ par effectif et partition ; ce ne sont pas trois positions initiales indépendantes.

L’audit complet conserve 56/56 géométries réalisables à 12 et 54/56 à 24 et 64. E et 10 sont refusées pour absence de connexité des places avec la portée de 3 m. Ces refus ne sont pas effacés de l’inventaire.

## Coût et stockage

Le protocole d’allocation conserve tous ses tours, sans verdict global d’arrêt anticipé. Les paquets immuables et fusions idempotentes identiques sont mis en cache localement. Le filtre retourne directement la projection sur la boule lorsque celle-ci satisfait déjà les contraintes ; le résultat du filtre est également réutilisé si toutes les entrées locales sont inchangées, obstacles compris. Un benchmark à 64 robots confirme une identité stricte des positions, vitesses et accélérations avant/après les dernières optimisations, pour un temps de calcul passé de 69,6 à 31,5 s.

Le schéma de stockage 3 utilise des listes de messages locaux CSR (offsets et enregistrements relatifs de quatre composantes) et conserve le contrat de caractéristiques local v2. Les phases 0/3 sont échantillonnées tous les deux pas ; les labels individuellement infaisables restent comptés mais ne sont pas sélectionnés. Les archives physiques restent disponibles. Les tableaux d’apprentissage ne dupliquent pas N voisins pour chaque robot lorsque seuls quelques-uns sont visibles.

## Apprentissage

Les 18 épisodes train/validation du professeur sont réussis. Le train offre 8 640, 19 788 et 67 169 exemples pour 12/24/64 ; la validation 8 640, 19 794 et 67 143. Les lots utilisent 8 640 exemples par effectif dans chaque partition, soit 25 920 exemples train et autant de validation. Cette sélection est reproductible et évite de pondérer automatiquement l’essaim le plus grand davantage.

Le GAT partagé garde huit modules linéaires et une largeur de 64. Ses normalisations sont déclarées : échelle spatiale 28, vitesse 1 et effectif 64. Il est initialisé séparément, sans modifier les poids de démonstration. Adam 0,001, lots 512, maximum 40 époques, patience 8, graine 42. Meilleur checkpoint à l’époque 38, MSE validation 0,0462629131. Aucune archive de test n’est chargée pour choisir les poids.

## Réserve et conclusion

Sur trois scénarios par effectif, le professeur réussit 3/3 à chaque effectif. Le GAT brut réussit 2/3 à 12 avec un contact, puis 0/3 à 24 avec 65 contacts et 0/3 à 64 avec 127 contacts. Le GAT protégé réussit 3/3 à 12 et 0/3 aux deux autres effectifs, sans contact robot/robot. Ses erreurs finales moyennes sont respectivement 0,037846 m, 6,659459 m et 14,155301 m. Un filtre de séparation ne garantit donc pas la convergence de la mission.

Ces résultats ne valident pas la généralisation à 24/64. Ils restent figés et ne servent pas à régler ce candidat après test. Une extension demandera plus de départs et formes, une analyse des trajectoires hors distribution et une nouvelle réserve. Les poids de référence restent inchangés. Le choix expérimental de l’interface est distinct de ce modèle et limité aux formations proches du rendez-vous.

Les options de pilote préparées dans l’interface sont explicitement expérimentales, jamais sélectionnées à la place du modèle de référence par défaut. Les dimensions de rendu sont liées à la physique enregistrée. Les exports complets peuvent être restaurés au redémarrage du serveur afin de préserver leurs liens. La suite complète réussit 590 tests.

## Commandes et preuves

```powershell
python -m src.imitation.multisize_collect --output outputs/imitation/multisize-pilot-task012
python -m src.imitation.multisize_train --dataset outputs/imitation/multisize-pilot-task012 --output outputs/imitation/training-multisize-pilot-task012
python -m src.imitation.multisize_evaluate --dataset outputs/imitation/multisize-pilot-task012 --run outputs/imitation/training-multisize-pilot-task012 --output outputs/imitation/multisize-test-task012
```

Les dossiers existants sont refusés. Preuves : `tasks/evidence/012-multisize-benchmark.json`, `012-multisize-manifest.json`, `012-multisize-run.json`, `012-multisize-evaluation.json`.
