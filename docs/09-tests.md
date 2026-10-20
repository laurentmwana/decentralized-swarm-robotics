# 09 — Tests

On vérifie des **propriétés**, pas des valeurs obtenues après coup.

| Composant | Propriétés |
|---|---|
| Simulation | respect de la portée et de l'état actif ; vitesse bornée ; robot en panne immobile et non compté en collision ; reproductibilité d'une graine ; règle locale valide pour un robot isolé |
| Contrôleur GNN | tailles d'essaim et voisinages variables ; invariance par permutation ; invariance par translation globale ; conditionnement par la consigne ; robots isolés ; communication désactivée |
| Métriques | valeurs exactes sur cas simples ; détection d'une paire en collision |
| Entraînement | pertes proches de zéro sur cas attendus ; pas d'auto-boucle dans l'adjacence ; cohérence de la cible ; plusieurs cibles dans un lot |
| Formes | normalisation et reproductibilité des points ; présence de l'index et de l'archive |

## À ajouter
- Cohérence du traitement des robots retirés (une fois la décision prise).
- Définition du succès : cas limites autour de τ (valeur exactement égale, franchissement avant la fin).
- Indépendance de la métrique vis-à-vis du nombre de points (ou documentation de la dépendance).
- Chaque nouveau contrôleur de baseline passe les mêmes tests d'interface.
