# Notes d'echecs des baselines (phase 3)

Seuil de succes : 0.15 (Chamfer normalisee, docs/decisions/06-evaluation.md). 7/10 runs reussis.

- **Regroupement (flocking)** : sans consigne de forme, les robots restent un nuage sans structure (equilibre cohesion-repulsion) ; l'echec sur le cercle est donc **volontaire** et documente (plancher « aucune information de forme »).
- **Regle manuelle** : attraction au point de contour le plus proche + repulsion. Sur l'echantillon difficile (lettres a diagonales, nombres denses a 40 robots), la regle passe mais de justesse (distances 0,10-0,12 pour un seuil a 0,15) : les concavites restent son point faible — plusieurs robots s'agglutinent sur une branche, laissant une autre branche sous-peuplee, ce que le GNN devra depasser.

| regle | cible | n_robots | distance_finale | reussite |
|---|---|---|---|---|
| flocking | geometries_circle@50 | 20 | 0.440 | non |
| flocking | geometries_circle@50 | 40 | 0.383 | non |
| flocking | geometries_circle@50 | 100 | 0.465 | non |
