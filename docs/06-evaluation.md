# 06 — Évaluation

## Mesures (définitions en mots)
- **Distance de forme** : distance de Chamfer symétrique en fin d'épisode. Moyenne des écarts « robot → point cible le plus proche » plus « point cible → robot le plus proche ». Sans affectation fixe. Préciser dans le texte si les distances sont au carré.
- **Taux de collisions** : proportion de paires plus proches que la distance de sécurité, moyennée sur la durée et normalisée par le nombre de paires. Pas une collision mécanique.
- **Régularité de répartition** : écart entre la dispersion des distances au plus proche voisin dans l'essaim et dans la cible (utiliser une valeur absolue pour éviter les signes ambigus). Détecte les amas, pas la couverture complète.
- **Succès** : distance de forme finale sous le seuil τ. Pilote : τ = 0,5. Toujours afficher τ avec le pourcentage.
- **Comparabilité entre tailles** : distance de Chamfer symétrique normalisée par la racine carrée du nombre de points cible, afin que le seuil τ ait le même sens indépendamment du nombre de robots et de la résolution de la cible.
- Lettres et nombres : évaluation visuelle ; lisibilité non validée.

Les mesures sont rapportées **ensemble**, jamais la plus flatteuse seule.

## Protocole pilote
- 240 épisodes, cible circulaire, 10 graines par condition, 20 pas.
- Tailles : 20, 40, 100 robots (modèle entraîné à 20). Retraits : 5, 10, 20 % à mi-épisode (20 robots).
- Contrôleurs : GNN (modèle du curriculum) et règle de regroupement (cohésion–répulsion, **sans** consigne de forme).
- Un fichier brut avec une ligne par épisode ; agrégats moyenne et écart-type.

## Résultats du premier prototype (référence historique, à refaire — implémentation reprise à zéro)
| Contrôleur | Retrait | Chamfer moyen | Collisions | Succès |
|---|---|---|---|---|
| GNN | 0 % | 0,592 | 0,021 | 0,10 |
| GNN | 20 % | 0,638 | 0,013 | 0,00 |
| Regroupement | 0 % | 3,956 | 0,187 | 0,00 |
| Regroupement | 20 % | 4,162 | 0,129 | 0,00 |

Échelle (sans panne) : GNN 0,592 / 0,522 / 0,428 pour 20 / 40 / 100 robots ; regroupement 3,956 / 3,799 / 3,507 ; succès GNN 0,10 / 0,30 / 1,00.
Pannes (écart-type) : GNN 0,592 (0,086), 0,612 (0,085), 0,626 (0,095), 0,638 (0,079) pour 0 / 5 / 10 / 20 % ; regroupement 3,956 (1,027), 3,966 (1,043), 3,971 (1,014), 4,162 (1,321).

## Lecture prudente
- Asymétrie d'information : l'écart GNN vs regroupement montre surtout l'effet de la consigne.
- Dix épisodes : validation de chaîne de mesure, pas d'intervalle de confiance.
- Le taux de collisions baisse mécaniquement quand des robots sont retirés (moins de paires) : pas un signe de sécurité accrue.
- Le succès est très sensible au seuil (10 % → 0 % pour un faible décalage).
- Densité des points et nombre de robots sont liés : la baisse de distance à 100 robots peut venir de l'échantillonnage du cercle.

## Campagne finale (à faire)
50 répétitions ou plus ; mêmes positions initiales entre contrôleurs ; conditions fixées avant de consulter les résultats ; un tableau + une figure par facteur ; exemples favorables **et** défavorables à graines fixes.
