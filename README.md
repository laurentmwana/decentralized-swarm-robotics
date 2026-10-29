# swarm-gnn

Contrôle décentralisé d'un essaim de robots par réseaux de neurones sur graphes.

## Démonstration actuelle — PLAN.md

L’interface du parcours à accélérations propose A–Z, 1–10 et 20 géométries, les transitions, la navigation et les obstacles. Elle compare le contrôleur classique au **GAT protégé par un filtre classique** ; le GAT brut reste insuffisamment sûr. Aucun entraînement n’est lancé depuis l’interface.

Un second modèle est maintenant disponible : **GAT structuré brut · 12/24/64**. Il réussit les 18 essais réservés de formation dans le domaine contrôlé de rendez-vous, sans filtre pendant les mouvements appris ; freinage et attribution restent classiques. Les départs dispersés, les autres figures, la navigation et les obstacles avec ce candidat restent à évaluer. Voir la [décision 30](docs/decisions/30-gat-structure-validation.md). Les résultats d’échec de l’ancien GAT brut restent conservés.

Sur cette installation Windows :

```powershell
./demo/start-reference.ps1
```

Avec un environnement Python contenant les dépendances du projet :

```powershell
python -m demo.reference_server
```

Ouvrir [la démonstration locale](http://127.0.0.1:8765). Choisir une expérience puis « Lancer l’expérience ». Le calcul est suivi d’une lecture animée avec pause, vitesse, remise au départ et curseur temporel. Les communications, trajectoires et places peuvent être affichées. « Exporter les résultats » fournit les mesures et paramètres JSON, les trajectoires CSV et les archives NPZ, dans un ZIP. Les fichiers sont aussi conservés sous `outputs/demo-reference/`.

Pour le modèle de référence, les effectifs 8, 16 et 24 sont exploratoires et l’apprentissage multi-formes initial utilise 12 robots. Le modèle structuré a son propre apprentissage à 12/24/64, des arènes 10/18/32 m, des largeurs de forme 6/12/28 m et un départ proche du rendez-vous. Choisir « Former l’essaim », puis le modèle structuré, pour l’utiliser. Les formes incompatibles sont refusées ; les autres formes non évaluées restent expérimentales. Les routes de navigation du modèle de référence sont fournies explicitement.

Le volet « Campagnes conservées » affiche les taux de réussite avec leurs dénominateurs et garde les échecs du GAT brut et des parcours difficiles. Pour réexporter ce tableau à partir des preuves, sans nouvelle simulation :

```powershell
python -m demo.comparison_report --output outputs/comparison-final
python -m pytest tests/test_reference_demo.py tests/test_navigation.py tests/test_transitions.py -q -p no:cacheprovider
```

Le dossier de sortie d’un rapport doit être nouveau. Le modèle validé est attendu dans `outputs/imitation/training-multiform-expanded-v1/` ; sans poids disponibles, le contrôleur classique reste utilisable. Les décisions [25](docs/decisions/25-navigation-obstacles.md) et [26](docs/decisions/26-interface-demonstration.md) décrivent les contrats et limites.

## Organisation conservée

Le [guide des expériences et commandes](docs/GUIDE_EXPERIENCES.md) explique les sept expériences, chaque contrôleur, les effectifs, le cercle provisoire, les métriques et les commandes de lancement. Il est aussi accessible dans l'interface. Les obstacles de la scène multiple ont été rapprochés des places finales du carré à 12 robots ; voir la [décision 35](docs/decisions/35-protocole-et-guide.md).

Les dimensions des formes sont vérifiées avant le lancement et adaptées si nécessaire pour conserver l'espacement de sécurité, les communications et les limites de l'arène. L'interface affiche la largeur retenue et désactive les combinaisons irréalisables. Une forme agrandie dispose d'un temps de construction adapté. Voir la [décision 34](docs/decisions/34-compatibilite-geometrique.md).

Pour cinq à dix obstacles, choisissez **Naviguer avec plusieurs obstacles**, puis **Nombre d’obstacles**. Le parcours utilise une préparation classique puis le contrôleur choisi pour le transport, avec planification sur carte connue et protection des murs. Les six comptes ont été évalués avec le carré à 12 robots ; les autres combinaisons restent expérimentales. Voir la [décision 33](docs/decisions/33-obstacles-multiples.md).

Le **GAT de référence protégé** a été entraîné avec 12 robots. Le **GAT structuré brut · 12/24/64** utilise d’autres poids, entraînés avec les trois effectifs et validés en formation contrôlée seulement. Choisir 24 robots ne change pas automatiquement de modèle.

En navigation, l’interface calcule désormais un détour classique sur la carte statique de la scène pour l’ensemble des places de la formation. La destination reste identique ; une place finale occupée entraîne un refus explicite avant simulation. Le GAT conserve ses poids et observations locales. Les résultats antérieurs sont présentés comme archives. Voir la [décision 31](docs/decisions/31-navigation-planifiee.md).

- `src/simulator/` : inertie, bornes physiques, observations locales et épisodes.
- `src/reference/` : catalogue, allocation locale, professeurs, sécurité, transitions et navigation.
- `src/imitation/` : données, GAT, entraînements et évaluations du cercle et des formes générales.
- `demo/` : interface actuelle, lanceur et rapport comparatif.
- `tests/` : tests du parcours à accélérations et des exports.
- `tasks/evidence/` et `docs/decisions/` : résultats et décisions scientifiques conservés.

Les anciens contrôleurs à vitesse, leur simulateur, leurs entraînements, leur interface et leurs configurations ont été supprimés après vérification des dépendances. Les décisions historiques restent consultables pour la traçabilité. Les données, modèles, documents de mémoire et de soutenance ne sont pas supprimés.

## Installation et vérification

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q -p no:cacheprovider
```

`make web` lance la démonstration actuelle ; `make test` exécute la suite conservée. Les commandes d’apprentissage et de campagnes sont documentées dans les décisions 23 à 27. Le candidat correctif 011 réduit les contacts de 102 à 65 sur les 18 nouveaux scénarios, mais échoue encore sans filtre ; il ne remplace pas le modèle de l’interface. La généralisation à 24 et 64 robots reste la tâche 012.
