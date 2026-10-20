# 07 — Baselines et ablations

Règle : un écart n'est attribuable au GNN que si les contrôleurs reçoivent **la même information de consigne**.

## Matrice de comparaison
| Contrôleur | Messages des voisins | Consigne de forme | Appris | Statut |
|---|---|---|---|---|
| Regroupement (cohésion–répulsion) | oui | aucune | non | à refaire |
| GNN 2 tours (modèle du curriculum) | oui | identifiant | oui | à refaire |
| Règle manuelle avec consigne | oui | vecteur vers le contour | non | à faire |
| Réseau appris sans communication | non | identifiant | oui | à faire (prioritaire) |
| GNN à 1 tour et 3 tours | oui | identifiant | oui | à faire |
| Agrégation : somme normalisée, attention, tronquée | oui | identifiant | oui | à faire (optionnel) |

## Règles d'expérience
- Mêmes conditions initiales entre contrôleurs ; budget d'entraînement identique ; taille de modèle comparable.
- Évaluer **modèle spécialisé cercle** et **modèle du curriculum** sur les mêmes épisodes (sinon l'effet du curriculum se mélange à celui de l'architecture).
- Rapporter ce que chaque contrôleur reçoit, dans le tableau lui-même.

## Questions auxquelles cela répond
1. La communication apporte-t-elle quelque chose à consigne égale ?
2. La profondeur (portée de l'information) compte-t-elle ?
3. Quelle part du résultat vient de la consigne plutôt que du passage de messages ?
