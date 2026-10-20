# Phase 0 — Configuration System

## Objectif

Système de configuration reproductible, typé et validé pour toutes les phases du projet.

## Design

- **Dataclasses gelées** (`frozen=True`) → immuables, thread-safe
- **Validation dans `__post_init__`** → fail-fast au chargement
- **Chargement YAML** → `from_yaml()` sur chaque classe
- **Pas d'héritage dataclass** → évite les problèmes de champs par défaut avec `frozen=True`
- **Conversion explicite** → `to_simulator_config()` pour extraire la config simulateur

## Classes

| Classe | Responsabilité | Fichier YAML |
|--------|----------------|--------------|
| `SimulatorConfig` | Dynamique, voisins, pannes, bruit | `configs/simulator.yaml` |
| `TrainingConfig` | Simulateur + hyperparamètres entraînement | `configs/training-circle.yaml` |
| `EvaluationConfig` | Protocole pilote (pannes, échelle, baselines) | `configs/evaluation-pilot.yaml` |

## Utilisation

```python
from src.swarm_gnn.config import TrainingConfig, load_config

# Via classe
config = TrainingConfig.from_yaml("configs/training-circle.yaml")

# Via utilitaire
config = load_config("configs/training-circle.yaml", "training")

# Extraire config simulateur pour la simulation
sim_config = config.to_simulator_config()
```

## Validation

Toutes les validations dans `__post_init__` :
- `robot_count > 0`
- `arena_size` : 2 dimensions positives
- `communication_radius`, `max_speed`, `time_step`, `safety_distance` > 0
- `episode_steps > 0`, `measurement_noise_std >= 0`
- Champs d'entraînement : `batch_size`, `iterations`, `learning_rate`, `hidden_dim`, `message_passing_steps` > 0

## Tests

```bash
python -m pytest tests/test_config.py -v
```

11 tests covering :
- Valeurs par défaut
- Validation (robot_count, arena_size, paramètres positifs)
- Chargement YAML
- Immuabilité (frozen)
- Utilitaire `load_config`