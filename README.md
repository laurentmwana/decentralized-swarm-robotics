# swarm-gnn

Contrôle décentralisé d'un essaim de robots par réseaux de neurones sur graphes.

Commencer par `AGENT.md`, puis `PLAN.md`, puis le dossier `docs/` (un fichier par étape).

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
