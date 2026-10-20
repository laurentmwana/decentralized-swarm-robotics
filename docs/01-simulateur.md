# Phase 1 — Simulateur 2D

## Objectif

Simulateur temps discret pour essaim 2D avec contrôle par vitesse directe.

## Fonctionnalités

| Fonction | Description |
|----------|-------------|
| **Dynamique** | `pos_{t+1} = clip(pos_t + v_t * dt, arena)` |
| **Contrôle vitesse** | Clipping euclidien à `max_speed` |
| **Voisinage** | Graphe radio `communication_radius`, symétrique, sans auto-boucle |
| **Pannes** | `fail(ids)` → robots inactifs (figés, exclus du graphe) |
| **Collisions** | Détection paires actives `< safety_distance` |
| **Bruit mesure** | Gaussien optionnel `N(0, measurement_noise_std)` |
| **Reproductibilité** | `np.random.Generator` seedé par config |

## API

```python
from src.swarm_gnn.config import SimulatorConfig
from src.swarm_gnn.simulator import SwarmSimulator

config = SimulatorConfig(
    seed=42,
    robot_count=20,
    arena_size=(10.0, 10.0),
    communication_radius=2.5,
    max_speed=1.0,
    time_step=0.3,
    episode_steps=20,
    safety_distance=0.3,
)

sim = SwarmSimulator(config)

# Boucle de simulation
for _ in range(config.episode_steps):
    commands = controller(sim.positions, sim.velocities, sim.neighbor_matrix())
    sim.step(commands)

# Observables
positions = sim.positions
velocities = sim.velocities
neighbors = sim.neighbor_matrix()
observed = sim.observed_positions()  # avec bruit si config.measurement_noise_std > 0
collisions = sim.collision_pairs()

# Pannes
sim.fail([3, 7, 12])  # robots figés à mi-épisode
```

## Tests

```bash
python -m pytest tests/test_simulator.py -v
```

23 tests couvrant :
- Initialisation (positions, vitesses, actifs, step_count)
- Reset (même seed = même trajectoire)
- Step (mise à jour positions, clipping arena, clipping vitesse)
- Robots inactifs (ne bougent pas, vitesse nulle)
- Matrice voisinage (forme, pas d'auto-boucle, symétrique, rayon, inactifs exclus)
- Pannes (fail, IDs invalides)
- Collisions (détection, inactifs exclus)
- Bruit de mesure (sans bruit, avec bruit gaussien)
- Déterminisme (même seed = même trajectoire)
- Validation forme commandes
- Incrémentation step_count