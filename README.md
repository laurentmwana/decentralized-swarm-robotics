# swarm-gnn

Contrôle décentralisé d'un essaim de robots par réseaux de neurones sur graphes.

Commencer par `AGENT.md`, puis `PLAN.md`, puis le dossier `docs/` (un fichier par étape).

## Commandes rapides

Le `Makefile` centralise les commandes du projet. Après installation de GNU Make, afficher la liste avec :

```powershell
make help
```

Les cibles les plus utiles sont `make test`, `make circle`, `make curriculum`, `make complex-smoke`, `make complex` et `make evaluation`. La cible `make complex` est plus longue : elle entraîne 50 robots et peut prendre plusieurs dizaines de minutes sur CPU.

## Démonstration du simulateur

Après installation de Python et des dépendances listées dans `AGENT.md`, lancer :

```powershell
python -m src.sim.demo
python -m pytest tests -q -p no:cacheprovider
```

L'animation produite est enregistrée dans `results/simulator-gathering.gif`.

## Génération des formes cibles

```powershell
python -m src.shapes.build
```

Les images, nuages de points, index et planche-contact sont créés dans `data/shapes/`; l'archive est `data/shapes.zip`.

## Vérification du contrôleur GNN

```powershell
python -m src.models.demo
```

Cette commande exécute le contrôleur non entraîné dans le simulateur et vérifie que ses commandes respectent la vitesse maximale configurée.

## Premier entraînement sur le cercle

```powershell
python -m src.train.run
```

Le modèle, la configuration, les pertes CSV et leur courbe sont enregistrés dans `results/training-circle/`. Pour exporter l'animation finale :

```powershell
python -m src.train.visualize
```

## Apprentissage progressif

```powershell
python -m src.train.curriculum
python -m src.train.visualize_curriculum
```

Le curriculum réduit préserve les formes des niveaux précédents tandis qu'il ajoute chiffres, lettres et nombres. Les résultats sont des contrôles intermédiaires ; consulter la planche produite avant toute conclusion sur la lisibilité.

## Apprentissage des formes complexes

La configuration `configs/curriculum-complex.yaml` entraîne un modèle distinct avec 50 robots, 50 points de contour, 60 pas de simulation et une géométrie locale de cible. Générer d'abord le jeu de données correspondant :

```powershell
python -m src.shapes.build --config configs/shapes-complex.yaml --output data/shapes-complex
python -m src.train.curriculum --config configs/curriculum-complex.yaml --initial-model none --output results/curriculum-complex
python -m src.train.visualize_curriculum --config configs/curriculum-complex.yaml --model results/curriculum-complex/numbers/model.pt --output results/curriculum-complex/numbers/contact-sheet.png
```

Cette campagne est volontairement plus coûteuse que le curriculum réduit et peut prendre plusieurs dizaines de minutes sur CPU. Le profil `configs/curriculum-complex-smoke.yaml` sert uniquement à vérifier le pipeline, pas à conclure sur la qualité des formes.

## Évaluation pilote

```powershell
python -m src.eval.run
```

Cette commande produit les épisodes bruts, agrégats et figures du protocole pilote (pannes et passage à l'échelle) dans `results/evaluation-pilot/`.

## API locale pour une démonstration Web

Pour afficher une trajectoire réellement calculée par le GNN dans une page Web, démarrer l'API locale :

```powershell
python -m src.web.server
```

Elle n'emploie ni Internet ni WebSocket. La page Web peut lire les routes suivantes :

- `GET /api/health` : vérifie que le modèle est disponible ;
- `GET /api/targets` : liste les formes que connaît le modèle final du curriculum ;
- `GET /api/simulate?target=alphabet_A&seed=142` : produit, en JSON, la cible, toutes les positions des robots à chaque pas et les mesures du protocole du mémoire. Paramètres optionnels : `method=gnn|gathering` (contrôleur appris ou règle de regroupement), `failure=0|0.05|0.10|0.20` (retrait à mi-épisode), `robots=20|40|100` (passage à l'échelle, cible dupliquée).

Les identifiants de cibles disponibles comprennent notamment `alphabet_A`, `digit_2` et `number_123`. L'animation Web doit dessiner les valeurs de `frames` retournées par cette dernière route ; elles sont calculées par `results/curriculum/numbers/model.pt`, pas par une règle codée dans le navigateur.

Le contrat détaillé des réponses JSON est dans [`docs/10-api-visualisation.md`](docs/10-api-visualisation.md).
