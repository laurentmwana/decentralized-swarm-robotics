# 02 — Simulateur

## Modèle
- Arène carrée 2D, dimension 10 × 10 dans la configuration de référence.
- Chaque robot : position et vitesse 2D. Dynamique à **vitesse directement commandée** : la nouvelle position est l'ancienne plus pas de temps × commande ; la norme de la commande est bornée par la vitesse max.
- Valeurs de référence (campagnes simples) : 20 robots, pas de temps 0,3, vitesse max 1, portée de communication 2,5.
- Positions limitées aux frontières de l'arène (effet de bord possible : un robot en bord ne se comporte pas comme en espace ouvert).

## Graphe de communication
Recalculé à **chaque pas** : une arête existe si la distance est inférieure ou égale à la portée. La topologie évolue avec les déplacements. Le graphe n'a pas d'auto-boucle.

## Hypothèses (simplifications assumées)
Repère commun, positions et messages exacts, instantanés et sans perte ; pas d'obstacle ; pas de masse, contact ni rebond (la « collision » est un seuil de sécurité, pas un choc). Ces limites sont à lever progressivement (voir PLAN, étape 10).

## Pannes
Un robot déclaré en panne est supprimé de la simulation : il cesse de se déplacer, n'émet plus de messages et n'est plus pris en compte dans le voisinage des autres robots. Retrait à mi-épisode, fraction tirée aléatoirement parmi les robots actifs ; la mesure finale ne porte que sur les robots actifs.

## Reproductibilité
La graine commande les positions initiales, l'ordre des cibles et les scénarios de retrait.
