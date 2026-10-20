# Phase 5 — Curriculum Progressif (Progressive Training)

## Objectif

Apprentissage par étapes : formes simples → complexes, avec transfert de poids.

## Principe

```
Stage 1 (geometry) → Stage 2 (digits) → Stage 3 (symbols) → Stage 4 (numbers)
    3 cibles              6 cibles              9 cibles             11 cibles
    50 iters             100 iters            150 iters            200 iters
```

Transfert de poids : `model.state_dict()` → stage suivant (embedding cible étendu)

## Configuration

```python
from src.swarm_gnn.training.curriculum import CurriculumConfig, run_curriculum_stage

config = CurriculumConfig.from_yaml("configs/curriculum.yaml")
state = CurriculumState()

for stage in config.stages:
    # Charger targets .npy
    targets = [np.load(p) for p in stage.targets]
    
    state = run_curriculum_stage(
        config.base,
        stage.name,
        stage.targets,
        output_dir / stage.name,
        state
    )
```

## YAML Curriculum

```yaml
base:
  seed: 43
  batch_size: 8
  episode_steps: 20
  learning_rate: 0.001
  robot_count: 20
  arena_size: [10.0, 10.0]
  communication_radius: 2.5
  max_speed: 1.0
  time_step: 0.3
  safety_distance: 0.3
  target_radius: 2.0
  shape_loss_weight: 1.0
  collision_loss_weight: 1.0
  motion_loss_weight: 0.01
  coverage_loss_weight: 0.0
  hidden_dim: 32
  message_passing_steps: 2

stages:
  - name: geometry
    iterations: 100
    targets: [circle.npy, square.npy, triangle.npy]
  - name: digits
    iterations: 120
    targets: [...]
  - name: symbols
    iterations: 150
    targets: [...]
  - name: numbers
    iterations: 200
    targets: [...]
```

## Transfert d'État

```python
state = CurriculumState()

# Stage 1
state = run_curriculum_stage(base_config, "geometry", targets, out_dir, state)

# Stage 2 - récupère les poids du stage 1
state = run_curriculum_stage(base_config, "digits", targets2, out_dir, state)
```

**Transfert intelligent :**
- Poids compatibles (même forme) → rechargés
- `target_embedding` : premiers `n` embeddings copiés
- Nouveau embedding pour nouvelles cibles → initialisé aléatoirement

## Sorties par Stage

```
results/
├── geometry/
│   ├── model.pt
│   ├── best.pt
│   ├── config.yaml
│   ├── loss.csv
│   ├── loss.png
│   ├── loss_target_0.csv
│   └── targets.yaml
├── digits/
│   └── ...
└── symbols/
    └── ...
```

## Tests

```bash
python -m pytest tests/test_curriculum.py -v
```

4 tests :
- Chargement YAML curriculum
- Transfert d'état (CurriculumState)
- Exécution stage unique
- Curriculum 2 étapes