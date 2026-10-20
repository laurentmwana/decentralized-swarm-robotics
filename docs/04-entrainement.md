# Phase 4 — Entraînement (Training)

## Objectif

Entraînement différentiable du GNN via rollout de simulateur.

## Pipeline

```
config → model → rollout(episode_steps) → losses → backward → optimizer.step()
```

## Fonctions de perte

| Perte | Formule | Rôle |
|-------|---------|------|
| **Chamfer** | `min_dist(pred→target)² + min_dist(target→pred)²` | Formation géométrique |
| **Collision** | `relu(safety_dist - dist)²` sur paires actives | Évitement collisions |
| **Mouvement** | `||commandes||²` | Régularisation commandes |

## Configuration (TrainingConfig)

```python
config = TrainingConfig(
    seed=42,
    batch_size=8,
    iterations=100,
    episode_steps=20,
    learning_rate=0.001,
    # ... params simulateur (robot_count, arena_size, etc.)
    target_radius=2.0,
    shape_loss_weight=1.0,
    collision_loss_weight=1.0,
    motion_loss_weight=0.01,
    hidden_dim=32,
    message_passing_steps=2,
)
```

## Entrées

- **Cibles** : liste de `np.ndarray [robot_count, 2]` normalisées `[-1,1]`
- **Scaling interne** : `target * target_radius + arena_size/2`

## Sorties

```
output/
├── model.pt           # dernier état
├── best.pt            # meilleur loss (dict)
├── circle-best.pt     # si train_circle
├── config.yaml        # config sérialisée
├── loss.csv           # loss par itération
├── loss.png           # courbe
├── loss_target_*.csv  # perte par cible (si multi)
└── loss_per_target.png
```

## API

```python
from src.swarm_gnn.training import train_circle, train_targets, TrainingConfig

# Cercle unique
model, losses = train_circle(config, circle_target, "results/circle")

# Multi-cibles (curriculum)
model, losses, metrics, per_target = train_targets(
    config, [target1, target2, ...], "results/multi"
)

# Perte finale par cible
metrics = {0: 0.05, 1: 0.08, ...}
```

## Rollout différentiable

```python
from src.swarm_gnn.training import rollout

final_pos, commands = rollout(
    model,           # GNNController
    positions,       # [batch, robots, 2]
    target_ids,      # [batch]
    config,          # TrainingConfig
    target_points,   # [batch, robots, 2] (centrés, scalés)
)
# final_pos: [batch, robots, 2]
# commands:  [batch, steps, robots, 2]
```

## Tests

```bash
python -m pytest tests/test_training.py -v
```

9 tests :
- Chamfer (symétrique, zéro si égal, non-négatif)
- Collision (pénalise proche, zéro si loin, exclut self)
- Rollout (formes, bornes arena)
- train_circle (loss diminue, fichiers créés)
- train_targets multi (métriques par cible)