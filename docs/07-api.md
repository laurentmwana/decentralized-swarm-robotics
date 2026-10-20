# Phase 7 — API Web (FastAPI + Swagger/ReDoc)

## Objectif

API REST pour démonstration et visualisation du modèle GNN entraîné.

## Stack

- **FastAPI** 0.110+ — framework web moderne, asynchrone
- **Uvicorn** — serveur ASGI performant
- **Swagger UI** (`/docs`) — documentation interactive
- **ReDoc** (`/redoc`) — documentation alternative
- **OpenAPI 3.1** — spécification auto-générée

## Démarrage

```bash
pip install fastapi uvicorn
python -m src.web.server --host 127.0.0.1 --port 8000
```

Options : `--host 0.0.0.0` (réseau), `--reload` (dev).

## Interfaces

| Interface | URL | Usage |
|-----------|-----|-------|
| Swagger UI | `/docs` | Tests interactifs |
| ReDoc | `/redoc` | Documentation |
| OpenAPI JSON | `/openapi.json` | Spéc machine |
| Health | `/api/health` | Vérification |

## Endpoints

### `GET /api/health`
```json
{ "status": "ok", "model": "results/curriculum-v2/numbers/model.pt" }
```

### `GET /api/targets`
11 formes : `geometry_circle`, `alphabet_A`, `digit_0`, `number_123`...

### `GET /api/simulate`
Paramètres :
| Param | Type | Requis | Défaut | Valeurs |
|-------|------|--------|--------|---------|
| `target` | string | ✓ | — | `geometry_circle`, `alphabet_A`... |
| `seed` | int | ✗ | 142 | ≥ 0 |
| `method` | string | ✗ | `gnn` | `gnn`, `gathering`, `manual`, `gnn_no_comm` |
| `failure` | float | ✗ | 0.0 | 0.0, 0.05, 0.10, 0.20 |
| `robots` | int | ✗ | 20 | 20, 40, 100 |

Exemples:
```bash
curl "http://127.0.0.1:8000/api/simulate?target=geometry_circle"
curl "http://127.0.0.1:8000/api/simulate?target=alphabet_A&failure=0.10"
curl "http://127.0.0.1:8000/api/simulate?target=digit_2&method=gathering"
curl "http://127.0.0.1:8000/api/simulate?target=number_123&robots=100"
curl "http://127.0.0.1:8000/api/simulate?target=geometry_square&method=manual"
```

### Réponse
```json
{
  "model": {"path": "...", "stage": "numbers", "type": "GNNController"},
  "target": {"id": "alphabet_A", "source": "...", "points": [[x,y],...]},
  "simulation": {"seed": 142, "method": "gnn", "robot_count": 20, "failure_fraction": 0.1, ...},
  "metrics": {"shape_distance": 0.098, "regularity_std": 0.12, "collisions": 45, "success": 1, ...},
  "frames": [[[x1,y1], [x2,y2],...], ...]  // t=0, t=1, ...
}
```

## Validation
- Enum auto-validé par FastAPI
- Erreurs 422 claires
- CORS activé

```bash
curl ".../simulate?target=geometry_circle&method=oracle"
# {"detail": "Méthode inconnue : oracle. Choix possibles : gnn, gathering, manual, gnn_no_comm."}
```

## Tests
```bash
python -m pytest tests/test_web_api.py -v
```
10 tests : comparaison protocole, frames, robots retirés, échelle, paramètres invalides.