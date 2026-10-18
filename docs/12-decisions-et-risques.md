# 12 — Décisions ouvertes, incohérences, risques

## Décisions ouvertes
1. **Intitulé** : « réseaux de neurones sur graphes » (décision prise) - L'idée cellulaire = unités identiques, voisinage limité (inspiration biologique) ; l'implémentation est un GNN à passage de messages sur graphe dynamique.
2. **Robot retiré** : supprimé du graphe (décision prise) - les robots retirés sont supprimés de la simulation et ne participent plus au voisinage ni à la communication.
3. **Définition du succès** : final sous τ (décision prise) - succès = distance de forme finale sous le seuil τ.
4. **Comparabilité entre tailles** : normaliser la distance (décision prise) - distance de Chamfer symétrique normalisée par la racine carrée du nombre de points cible.
5. **Modèle évalué** dans le pilote : spécialisé cercle ou curriculum (les figures d'entraînement viennent du cercle).

## Incohérences relevées dans le mémoire
- Pannes (voir décision 2).
- Pertes finales du curriculum (0,362 ; 0,251 ; 0,240 ; 0,117) ≠ pertes par cible (0,316 ; 0,126 ; 0,141 ; 0,117) : préciser que ce sont deux mesures différentes, ou corriger.
- Équation de régularité : valeur absolue manquante ; distance de Chamfer : carré ou non, à préciser.
- Coquilles : « convergegences », « se accumule », « baseline appraise », « robotsiques », « L'arête est finie » (arène ?), « oeuvre » → « œuvre ».

## Menaces à la validité
| Menace | Parade |
|---|---|
| Peu de répétitions (10) | 50 répétitions + intervalles de confiance |
| Une seule graine d'entraînement | plusieurs graines d'entraînement |
| Baseline sans consigne | étape 6 du plan |
| Épisodes trop courts (seuil rarement atteint) | épisodes plus longs, courbes temporelles |
| Simulateur idéalisé (repère commun, mesures exactes, pas de délai) | bruit, délais, repère relatif |
| Pannes « propres » | robot défaillant immobile ou aberrant |
| Lisibilité des symboles non mesurée | protocole humain ou classifieur |
| Généralisation non testée | formes réservées avant entraînement |
| Effet de bord de l'arène | arène plus grande, comparaison |
| Couplage curriculum / architecture | évaluer les deux modèles |

## Risques de projet
Coût de calcul des campagnes complexes (processeur) ; dérive des configurations ; temps de rédaction ; répétition des limites dans le texte (à regrouper).
