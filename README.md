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
