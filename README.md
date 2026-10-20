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

## API locale pour une démonstration Web (Flask + Swagger)

L'API est maintenant construite avec **Flask** et documentée via **Swagger UI (OpenAPI 3.0)**.

### Démarrage

```powershell
python -m src.web.server
# Options: --host 127.0.0.1 --port 8000 --debug
```

### Interfaces disponibles

| Interface | URL | Description |
|-----------|-----|-------------|
| **Swagger UI** | http://127.0.0.1:8000/docs | Documentation interactive, test des endpoints |
| **Health check** | http://127.0.0.1:8000/api/health | Vérification modèle chargé |
| **Targets** | http://127.0.0.1:8000/api/targets | Liste des 11 formes du curriculum |
| **Simulate** | http://127.0.0.1:8000/api/simulate | Lance une simulation complète |
| **OpenAPI Spec** | http://127.0.0.1:8000/static/swagger.json | Spécification JSON brute |

### Endpoint `/api/simulate` - Paramètres

| Paramètre | Type | Requis | Défaut | Valeurs | Description |
|-----------|------|--------|--------|---------|-------------|
| `target` | string | Oui | - | `geometry_circle`, `alphabet_A`, `digit_0`, `number_123`, etc. | Identifiant de la forme cible |
| `seed` | integer | Non | 142 | ≥ 0 | Graine aléatoire pour reproductibilité |
| `method` | string | Non | `gnn` | `gnn`, `gathering`, `manual`, `gnn_no_comm` | Méthode de contrôle |
| `failure` | number | Non | 0.0 | 0.0, 0.05, 0.10, 0.20 | Fraction de robots en panne à mi-épisode |
| `robots` | integer | Non | 20 | 20, 40, 100 | Nombre de robots (passage à l'échelle) |

### Exemples d'utilisation

```powershell
# Simulation GNN standard (20 robots, cercle)
curl "http://127.0.0.1:8000/api/simulate?target=geometry_circle"

# GNN avec 10% de pannes à mi-épisode
curl "http://127.0.0.1:8000/api/simulate?target=alphabet_A&failure=0.10"

# Baseline gathering (sans GNN)
curl "http://127.0.0.1:8000/api/simulate?target=digit_2&method=gathering"

# Passage à l'échelle : 100 robots
curl "http://127.0.0.1:8000/api/simulate?target=number_123&robots=100"

# Contrôle manuel avec consigne de forme
curl "http://127.0.0.1:8000/api/simulate?target=geometry_square&method=manual"
```

### Réponse `/api/simulate` (extrait)

```json
{
  "model": {"path": "...", "stage": "numbers", "type": "GNNController"},
  "target": {"id": "alphabet_A", "source": "...", "points": [[x,y], ...]},
  "simulation": {
    "seed": 142,
    "method": "gnn",
    "robot_count": 20,
    "failure_fraction": 0.1,
    "failure_step": 10,
    "retired": [3, 7, 14],
    "episode_steps": 20,
    "arena_size": [10.0, 10.0],
    ...
  },
  "metrics": {
    "shape_distance": 0.098,
    "regularity_std": 0.12,
    "collisions": 45,
    "collision_rate": 0.023,
    "convergence_step": 15,
    "success": 1
  },
  "frames": [
    [[x1,y1], [x2,y2], ...],  // t=0
    [[x1,y1], [x2,y2], ...],  // t=1
    ...
  ]
}
```

Le contrat détaillé des réponses JSON est dans [`docs/10-api-visualisation.md`](docs/10-api-visualisation.md).
