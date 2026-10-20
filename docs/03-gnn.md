# Phase 3 — Contrôleur GNN (GNN Controller)

## Objectif

Politique partagée, invariante à la permutation, à passage de messages pour nombre variable de robots.

## Architecture

```
vitesse(2) + but_relatif(2) + target_embedding(hidden)
    │
    ▼
Encoder: Linear(hidden+4→hidden) → ReLU → Linear(hidden→hidden) → ReLU
    │
    ▼
Message Passing × K steps:
    ├─ Message: MLP(récepteur + émetteur + edge_features)  edge = (dx, dy, dist)
    ├─ Agrégation: moyenne pondérée par adjacence
    └─ Update: MLP(état_courant + messages_agrégés)
    │
    ▼
Decoder: Linear(hidden→hidden) → ReLU → Linear(hidden→2) → tanh + bornage max_speed
```

## Classes

### `GNNController`

Contrôleur principal avec communication.

```python
ctrl = GNNController(
    hidden_dim=64,
    message_passing_steps=2,
    target_count=11,
    max_speed=1.0,
)

commands = ctrl(
    positions,      # [batch, robots, 2]
    velocities,     # [batch, robots, 2]
    adjacency,      # [batch, robots, robots] bool
    target_ids,     # [batch] int
    goal_relative_positions=None,    # optional [batch, robots, 2]
    target_geometry_features=None,   # optional [batch, robots, 3]
)
# → commands [batch, robots, 2] bornés à max_speed
```

**Propriétés :**
- **Invariant à la permutation** : même entrée permutée → sortie permutée
- **Nombre variable de robots** : même modèle pour 2, 5, 10, 20, 50 robots
- **Conditionné par cible** : embedding appris par target_id
- **Géométrie cible optionnelle** : features locales (vecteur + distance au point cible le plus proche)

### `NoCommunicationController`

Baseline sans communication (MLP par robot).

```python
ctrl = NoCommunicationController(
    hidden_dim=64,
    target_count=4,
    max_speed=1.0,
)

commands = ctrl(
    velocities,     # [batch, robots, 2]
    target_ids,     # [batch]
    goal_relative_positions=None,
    target_geometry_features=None,
)
```

## Bornage de vitesse

```python
def _limit_speed(commands, max_speed):
    bounded = torch.tanh(commands)
    norms = ||bounded||.clamp_min(1e-12)
    return bounded * clamp(max_speed / norms, max=1.0)
```
- Borne la **norme euclidienne**, pas chaque composante
- `tanh` pour saturation douce avant bornage

## Tests

```bash
python -m pytest tests/test_gnn.py -v
```

15 tests :
- Initialisation (params valides/invalides)
- Forward shape, speed bound
- Entrées optionnelles (goal, geometry)
- Validation formes (erreurs claires)
- **Équivariance par permutation** (clé !)
- Nombre variable de robots (2→20)
- Déterminisme
- Baseline NoCommunication