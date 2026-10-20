# Phase 6 — Évaluation Pilote (Pilot Evaluation)

## Objectif

Protocole expérimental reproductible : baselines, pannes, passage à l'échelle.

## Métriques

| Métrique | Définition |
|----------|------------|
| **Distance de Chamfer** | Symétrique non-carrée entre robots actifs et cible |
| **Régularité (std NN)** | Écart-type des distances au plus proche voisin |
| **Collisions** | Paires actives `< safety_distance` |
| **Taux collisions** | Collisions / (pas × paires possibles) |
| **Convergence** | Premier pas `t` où Chamfer ≤ 0.5 |
| **Succès** | `t_conv ≤ T-1` |
| **Connectivité** | Fraction robots dans plus grande composante connexe |

## Protocole

```python
from src.swarm_gnn.evaluation import simulate_episode, EvaluationConfig
from src.swarm_gnn.models import GNNController

config = EvaluationConfig(
    seed=101,
    episodes=50,
    episode_steps=20,
    robot_count=20,
    arena_size=(10.0, 10.0),
    communication_radius=2.5,
    max_speed=1.0,
    time_step=0.3,
    safety_distance=0.3,
    target_radius=2.0,
    failure_fractions=(0.0, 0.05, 0.10, 0.20),
    scale_factors=(1, 2, 5),  # 20, 40, 100 robots
    success_threshold=0.5,
    model_path="results/curriculum-v2/numbers/model.pt",
    target_path="data/shapes/geometry/geometry_circle.npy",
)

model = GNNController(hidden_dim=32, message_passing_steps=2, target_count=11, max_speed=1.0)

# Simulation unique
result = simulate_episode(
    model, "gnn", target, 20, 0.1, 142, config, target_id=0
)
# result: dict avec métriques + frames + simulation params
```

## Méthodes Comparées

| Méthode | Description |
|---------|-------------|
| `gnn` | GNN entraîné avec communication |
| `gathering` | Règle locale : moyenne positions voisins |
| `manual` | Attraction cible + répulsion (règle manuelle) |
| `gnn_no_comm` | GNN sans messages (adjacence nulle) |

## Paramètres

| Paramètre | Valeurs |
|-----------|---------|
| `failure_fraction` | 0.0, 0.05, 0.10, 0.20 |
| `robots` (échelle) | 20, 40, 100 |
| `seed` | 142 + episode |
| `target` | Cercle (geometry_circle.npy) |

## Sorties

```python
result = {
    "method": "gnn",
    "robots": 20,
    "failure_fraction": 0.1,
    "seed": 142,
    "shape_distance": 0.098,
    "regularity_std": 0.12,
    "collisions": 45,
    "collision_rate": 0.023,
    "convergence_step": 15,
    "success": 1,
    "final_connectivity": 0.95,
    "retired": [3, 7, 12],  # si panne
}
```

## Tests

```bash
python -m pytest tests/test_evaluation.py -v
```

9 tests :
- Métriques (collision, Chamfer, régularité, connectivité)
- Simulation basique (gnn)
- Pannes (retired, failure_step)
- 4 méthodes (gnn, gathering, manual, gnn_no_comm)