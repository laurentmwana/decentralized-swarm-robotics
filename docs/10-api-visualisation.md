# API locale de visualisation du GNN

## But

Cette API locale sert une application Web de démonstration. Elle calcule une trajectoire avec le modèle GNN entraîné et renvoie des données JSON ; le navigateur est responsable de les dessiner et de les animer.

Le serveur ne requiert ni accès Internet, ni WebSocket, ni base de données. Il écoute par défaut uniquement sur `127.0.0.1`, donc il n'est accessible que depuis la machine qui exécute le projet.

## Démarrage

Depuis la racine du projet :

```powershell
python -m src.web.server
```

Le serveur est alors disponible à l'adresse `http://127.0.0.1:8000`.

Options possibles :

```powershell
python -m src.web.server --host 127.0.0.1 --port 8765
```

## Conventions communes

- Toutes les réponses utilisent `application/json; charset=utf-8`.
- Les coordonnées sont en 2D, sous la forme `[x, y]`.
- Une requête réussie reçoit le code HTTP `200`.
- Une requête invalide reçoit le code HTTP `400` avec un champ `error`.
- Une route inconnue reçoit le code HTTP `404` avec un champ `error`.
- Le serveur autorise les appels d'une page Web locale avec l'en-tête `Access-Control-Allow-Origin: *`.

## `GET /api/health`

### But

Vérifie que le serveur répond et indique le chemin du modèle prévu pour la démonstration.

### Réponse réussie — `200`

```json
{
  "status": "ok",
  "model": "results/curriculum/numbers/model.pt"
}
```

## `GET /api/targets`

### But

Retourne les formes connues par le dernier niveau du curriculum. L'interface doit utiliser la valeur `id` comme paramètre `target` de `/api/simulate`.

### Réponse réussie — `200`

```json
{
  "targets": [
    {"id": "geometry_circle", "source": "data/shapes/geometry/geometry_circle.npy"},
    {"id": "geometry_square", "source": "data/shapes/geometry/geometry_square.npy"},
    {"id": "geometry_triangle", "source": "data/shapes/geometry/geometry_triangle.npy"},
    {"id": "digit_0", "source": "data/shapes/digits/digit_0.npy"},
    {"id": "digit_1", "source": "data/shapes/digits/digit_1.npy"},
    {"id": "digit_2", "source": "data/shapes/digits/digit_2.npy"},
    {"id": "alphabet_A", "source": "data/shapes/alphabet/alphabet_A.npy"},
    {"id": "alphabet_B", "source": "data/shapes/alphabet/alphabet_B.npy"},
    {"id": "alphabet_C", "source": "data/shapes/alphabet/alphabet_C.npy"},
    {"id": "number_010", "source": "data/shapes/numbers/number_010.npy"},
    {"id": "number_123", "source": "data/shapes/numbers/number_123.npy"}
  ]
}
```

`source` est informatif : la page Web ne lit pas directement le fichier `.npy`.

## `GET /api/simulate`

### But

Exécute une simulation complète avec le GNN et renvoie la cible et les positions successives des robots. C'est la route utilisée pour l'animation.

### Paramètres de requête

| Paramètre | Obligatoire | Type | Description |
|---|---|---|---|
| `target` | Oui | texte | Identifiant retourné par `/api/targets`, par exemple `alphabet_A`. |
| `seed` | Non | entier | Graine du tirage des positions initiales. Valeur par défaut : `142`. Même graine = même trajectoire. |
| `method` | Non | texte | `gnn` (défaut) ou `gathering` : contrôleur appris ou règle de regroupement du protocole pilote. |
| `failure` | Non | décimal | Fraction de robots retirés à mi-épisode : `0` (défaut), `0.05`, `0.10` ou `0.20`. |
| `robots` | Non | entier | Taille d'essaim : `20` (défaut), `40` ou `100` — la cible est dupliquée comme dans le protocole pilote. |

Exemples :

```text
GET /api/simulate?target=alphabet_A&seed=142
GET /api/simulate?target=geometry_circle&method=gathering&failure=0.2&robots=40
```

### Réponse réussie — `200`

```json
{
  "model": {
    "path": "results/curriculum/numbers/model.pt",
    "stage": "numbers",
    "type": "GNNController"
  },
  "target": {
    "id": "alphabet_A",
    "source": "data/shapes/alphabet/alphabet_A.npy",
    "points": [[4.12, 2.75], [4.31, 3.10]]
  },
  "simulation": {
    "seed": 142,
    "method": "gnn",
    "robot_count": 20,
    "failure_fraction": 0.0,
    "failure_step": null,
    "retired": [],
    "episode_steps": 20,
    "arena_size": [10.0, 10.0],
    "communication_radius": 2.5,
    "time_step": 0.3,
    "max_speed": 1.0,
    "safety_distance": 0.3,
    "success_threshold": 0.5
  },
  "metrics": {
    "shape_distance": 0.59,
    "regularity_std": 0.24,
    "collisions": 8,
    "collision_rate": 0.021,
    "convergence_step": 20,
    "success": 0
  },
  "frames": [
    [[1.20, 7.81], [8.41, 2.34]],
    [[1.31, 7.65], [8.26, 2.44]]
  ]
}
```

Les tableaux de l'exemple sont volontairement raccourcis. Dans une réponse réelle du curriculum actuel :

- `target.points` contient 20 points de contour ;
- chaque élément de `frames` contient 20 positions de robots ;
- `frames` contient 21 éléments : l'état initial, puis un état après chacun des 20 pas.

### Signification des champs

| Champ | Signification |
|---|---|
| `model` | Identification du contrôleur ayant produit la trajectoire. |
| `target.points` | Nuage de points de la forme à afficher en arrière-plan (déjà dupliqué si `robots > 20`). |
| `simulation` | Paramètres du protocole : échelle du canvas, méthode, panne, seuil de succès. |
| `simulation.retired` | Identifiants des robots retirés à mi-épisode (figés ensuite, à afficher en gris). |
| `metrics` | Mesures du chapitre 3 calculées à la fin de l'épisode sur les robots actifs. |
| `frames[0]` | Positions initiales aléatoires des robots. |
| `frames[i]` | Positions après `i` pas de simulation. |
| `frames[20]` | État final à montrer pour le résultat de la formation. |

### Erreurs — `400`

Sans cible :

```json
{"error": "Le paramètre ?target= est obligatoire."}
```

Avec une forme qui n'appartient pas au curriculum :

```json
{
  "error": "Forme inconnue : alphabet_Z. Choix possibles : alphabet_A, alphabet_B, alphabet_C, digit_0, digit_1, digit_2, geometry_circle, geometry_square, geometry_triangle, number_010, number_123"
}
```Avec une graine non entière :

```json
{"error": "invalid literal for int() with base 10: 'bonjour'"}
```

Avec une méthode, une fraction de panne ou une taille hors protocole :

```json
{"error": "Méthode inconnue : oracle. Choix possibles : gnn, gathering."}
{"error": "Fraction de panne invalide : 0.15. Choix possibles : 0.0, 0.05, 0.1, 0.2."}
{"error": "Nombre de robots invalide : 30. Choix possibles : 20, 40, 100 (protocole pilote)."}
```

## Protocole reproduit

Les trajectoires et les mesures ne sont pas codées dans le serveur : elles sont calculées par les mêmes fonctions que celles qui ont produit les tableaux et les figures du chapitre 4 du mémoire (`src.eval.run` et `src.eval.metrics`). En conséquence :

- le graphe de communication est recalculé à chaque pas, un robot retiré est figé et exclu du graphe, des collisions et de la mesure finale ;
- la distance de forme est la distance de Chamfer symétrique **non carrée** sur les positions finales des robots actifs ;
- le taux de collisions est normalisé par le nombre de paires possibles et par les `episode_steps` pas ;
- le succès vaut 1 si le seuil `success_threshold = 0,5` est franchi au plus tard au pas `T-1` (convention du tableau 4.3) ;
- en passage à l'échelle, les points de la cible sont dupliqués avec `np.resize`, exactement comme dans le protocole pilote.

## Exemple minimal côté navigateur

```js
const result = await fetch(
  "http://127.0.0.1:8000/api/simulate?target=alphabet_A&seed=142"
).then((response) => response.json());

const { arena_size } = result.simulation;
const frame = result.frames[0]; // Pour animer, avancer ensuite d'une image à la fois.

// Conversion du repère de la simulation vers un canvas de largeur `canvas.width`.
const toCanvas = ([x, y]) => [
  x / arena_size[0] * canvas.width,
  canvas.height - y / arena_size[1] * canvas.height
];

// result.target.points : dessiner les points de la forme en arrière-plan.
// frame : dessiner les robots.
```

Avant de dessiner, vérifier `response.ok` et afficher `result.error` si la requête échoue.

## Limite importante pour la démonstration

Cette API utilise le modèle final du **curriculum réduit**. Il connaît les 11 formes listées par `/api/targets` ; demander une autre lettre ou un autre chiffre ne crée pas une nouvelle capacité. Les formes complexes restent une piste d'entraînement à poursuivre, pas un résultat à revendiquer.
