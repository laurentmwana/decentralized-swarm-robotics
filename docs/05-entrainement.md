# 05 — Entraînement

## Principe
Apprentissage **direct à travers un simulateur différentiable** : une seule perte par épisode, calculée sur tout l'essaim, rétropropagée jusqu'aux paramètres partagés. Pas de récompense individuelle, pas d'imitation d'expert. Choix motivé par le coût, la variance du gradient et la reproductibilité ; ce n'est **pas** une preuve de supériorité sur l'imitation ou le renforcement (non comparés).

## Perte : quatre termes (poids dans les configurations)
| Terme | Rôle | Ce qu'il empêche |
|---|---|---|
| Forme (distance de Chamfer symétrique) | rapprocher essaim et cible | l'essaim n'a aucune raison de bouger |
| Collisions | pénaliser les paires sous la distance de sécurité | compression sur le contour |
| Mouvement | limiter les commandes saturées | oscillations |
| Couverture | rapprocher la régularité des espacements de celle de la cible | une partie du contour reste vide |

## Boucle (résumé)
Initialiser paramètres et Adam ; pour chaque itération : tirer un lot de cibles et de départs, dérouler les épisodes, calculer la perte du lot, mettre à jour ; **sauvegarder le meilleur état** (pas seulement le dernier : les dernières itérations d'un niveau ne sont pas forcément les meilleures).

## Curriculum réduit
Géométries (3 cibles) → chiffres (6) → lettres (9) → nombres (11). Chaque niveau **conserve les cibles précédentes** dans le lot. Cela réduit l'oubli sans l'éliminer : mesurer à chaque niveau la perte sur toutes les formes déjà vues.
Cibles symboliques actuelles : lettres A, B, C ; nombres 010 et 123 seulement.

## Campagnes
| Paramètre | Cercle | Curriculum réduit | Formes complexes |
|---|---|---|---|
| Graine | 41 | 43 | 59 |
| Robots / points de cible | 20 / 20 | 20 / 20 | 50 / 50 |
| Pas de simulation | 20 | 20 | 60 |
| Pas de temps | 0,3 | 0,3 | 0,2 |
| Itérations | 80 | 4 × 30 | 120 à 200 |
| Taille de lot | 8 | 8 | 4 |
| Taux d'apprentissage | 0,003 | 0,002 | 0,001 |
| Dimension cachée | 32 | 32 | 64 |
| Tours de messages | 2 | 2 | 3 |
| Portée | 2,5 | 2,5 | 2,5 |
| Vitesse max | 1,0 | 1,0 | 1,0 |
| Distance de sécurité | 0,3 | 0,3 | 0,22 |
| Rayon de cible | 2,0 | 2,0 | 2,8 |
| Poids forme / collisions / mouvement / couverture | 1 / 1 / 0,01 / 0,20 | idem | 1 / 0,5 / 0,005 / 0,15 |

Ces valeurs ne sont pas optimales : elles rendent chaque campagne exécutable sur un poste standard. Ne pas comparer les campagnes entre elles sans rappeler leurs différences (taille, durée, information de consigne).

## Résultats d'entraînement du premier prototype (référence historique)
Cercle, graine 41 : perte de 10,08 à 0,29 en 80 itérations (diagnostic d'optimisation, pas une mesure de généralisation). Curriculum : pertes finales 0,362 / 0,251 / 0,240 / 0,117 (à réconcilier avec le tableau par cible, voir `12`).

## À faire
Plusieurs graines d'entraînement ; comparer plusieurs dimensions cachées et profondeurs à budget identique.
