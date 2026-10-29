# 22 — Contrat local et jeu pilote d'imitation multi-formes

Date : 2026-10-29. Suite autorisée par « continue », après validation 008b.

## Schéma distinct et frontière locale

Le jeu porte `schema_version: 2`, type `multiform_local_target_imitation`. L'ancien chargeur cercle le refuse : huit caractéristiques de même dimension ne signifient pas un même contrat.

Chaque état robot/pas contient : position propre (x,y), vitesse propre (x,y), cible propre (x,y), phase et effectif. Le voisinage contient uniquement position et vitesse relatives (quatre composantes), masquées hors portée. Les cibles privées des voisins ne sont jamais incluses dans ces messages.

`local_example` est la frontière destinée au futur modèle : état propre, voisins physiques visibles, label propre et masque d'apprentissage. Modifier les cibles d'autres robots ne change pas cet exemple local. Les tableaux complets de l'archive servent à la collecte hors ligne et au rejeu ; les fournir aveuglément à l'encodeur voisin d'un GAT violerait le contrat.

Le simulateur enregistre la cible avant chaque commande : place de rendez-vous propre pendant la phase 0, position propre pendant le freinage/diffusion (phases 1/2), place attribuée pendant le mouvement final (phase 3). Aucun état après action n'est utilisé comme entrée de cette action.

## Phases et labels

La cible supervisée est l'accélération effectivement appliquée par le professeur, bornée en norme à 2 m/s². Les commandes demandées restent également enregistrées. Le masque apprend les phases de mouvement 0 et 3. Le freinage et la diffusion d'affectation restent classiques dans ce premier contrat ; les trajectoires, communications, collisions et délais de toutes les phases restent dans les archives et métriques de mission.

Le professeur emploie les paramètres figés de 008b : rendez-vous 8 s, freinage 1 s, diffusion N² tours de 0,1 s, détour conditionné et filtre local. Domaine : 12 robots, portée 3 m, arène 10 m, disque initial de rayon 4 m sans chevauchement, séparation minimale 0,45 m, horizon total 40 s, cadre des formes 6 m (rectangle 6×4 m). Les échecs et infaisabilités du filtre ne sont pas effacés.

## Partition des épisodes et formes

`config/multiform-data.yaml` fige la liste complète de 56 identifiants et trois graines disjointes : 7100, 7200, 7300 respectivement pour train, validation, test. Chaque graine donne un départ partagé par les formes de son ensemble. Ce pilote utilise donc trois départs distincts, pas 56 tirages initiaux indépendants.

La partition par défaut regroupe les mêmes tracés vectoriels via une empreinte de leurs segments non orientés, quantifiés au nanomètre. Ce mécanisme reconnaît les alias connus du catalogue, notamment O/cercle et V/V géométrique. Il n'est pas un solveur général d'équivalence de courbes discrétisées différemment. Les groupes sont stratifiés par catégorie, classés de façon déterministe avec la graine 20261029, puis séparés approximativement à 70/15/15 avant expansion des alias. La configuration explicite est la référence figée.

Une forme réservée n'est pas disponible à l'apprentissage du futur GAT. Les échecs du professeur sont néanmoins mesurés lors de la collecte de référence, avant tout choix de poids. Aucun entraînement ou sélection de modèle neuronal n'a été effectué sur ce jeu.

Les alias collectés dans le même ensemble sont conservés pour couvrir les identifiants du catalogue, mais une duplication exacte d'entrées/labels est marquée `duplicate_within_split_of`. L'entraînement futur devra exclure ces épisodes doublons de ses poids de sélection et d'optimisation. Une duplication entre ensembles est refusée.

## Contrôles et résultats réels

La collecte conserve les métadonnées, consignes géométriques, empreintes des archives et simulations, qualités par épisode et récapitulatifs. Un jeu incomplet ou dont la qualité exige une revue est refusé par défaut. L'inspection des échecs reste possible via une option explicite. Aucun dossier existant n'est écrasé.

Le chargeur vérifie schéma, appartenance à l'ensemble, intégrité, métadonnées, dimensions, alignement des états/cibles/phases, voisinage masqué, masque d'apprentissage et bornes des labels. L'audit vérifie également que les états initiaux ne sont pas partagés entre ensembles. Il ne charge pas test par défaut ; son inclusion exige une demande explicite.

| Ensemble | Épisodes | Épisodes uniques | Réussites professeur | Exemples robot/pas avec masque appris, alias inclus |
|---|---:|---:|---:|---:|
| Train | 37 | 36 | 37 | 109 224 |
| Validation | 10 | 9 | 10 | 29 520 |
| Test | 9 | 9 | 9 | 26 568 |

La collecte `outputs/imitation/multiform-pilot-task008c` est complète et porte le statut de qualité `passed` pour les 56 épisodes observés. L'audit des trois ensembles est réussi. Après déduplication des alias, il reste 54 simulations uniques et 159 408 exemples robot/pas apprenables. Les autres phases restent archivées, sans devenir des exemples d'optimisation.

Ce résultat valide le contrat et la collecte pilote ; il n'établit ni la suffisance statistique du corpus ni les performances d'un modèle appris. Le filtre conserve ses infaisabilités et sa portée empirique décrites en 008b.

## Vérification et suite

326 tests ciblés réussis sur physique, catalogue, affectation, professeur, reconnexion et données. Les tests ajoutés couvrent les alias entre partitions, l'alignement avant commande, la confidentialité des cibles voisines, la corruption, l'absence d'écrasement, la revue des échecs et l'audit sans lecture de test par défaut.

Preuves : `tasks/evidence/008c-multiform-manifest.json` et `008c-multiform-audit.json`. Les archives NPZ sont dans le dossier de collecte ignoré par Git.

La suite 008d doit créer un modèle à cible locale et son propre checkpoint. L'encodeur de messages ne doit pas reprendre les cibles privées des voisins, même si toutes les lignes propres existent dans le lot hors ligne. Les modules classiques communs et le rôle du filtre devront être explicités dans l'évaluation.

## Commandes

```powershell
python -m src.imitation.multiform_data --config config/multiform-data.yaml --output outputs/imitation/multiform-pilot-task008c
python -m src.imitation.multiform_data --verify outputs/imitation/multiform-pilot-task008c
python -m src.imitation.multiform_data --verify outputs/imitation/multiform-pilot-task008c --include-test
```

Le dernier audit inclut explicitement le test ; aucun de ces outils ne crée d'optimiseur. Utiliser une nouvelle sortie pour reproduire la collecte. Les chemins Python et dépendances Windows de 007b restent utilisables.
