# 09 — Consigne et professeur circulaires locaux

Date : 2026-10-28. Statut : validé par l’utilisateur (tâche 003).

## Consigne et informations accessibles

Le centre, le rayon et l'effectif sont une consigne connue de chaque robot, dans un repère commun en mètres. La localisation propre dans ce repère est supposée disponible. Cela ne donne pas accès aux positions distantes et n'implique pas de calcul centralisé des commandes. La même consigne devra être fournie au GAT.

La corde cible est `2 R sin(pi/N)` ; pour un robot isolé aucune répartition n'est nécessaire. Pas d'affectation de places, pas d'identifiant, pas d'angle cible par robot. Le cercle est invariant en rotation. Les différences de position des voisins sont celles de l'observation locale de la tâche 002.

## Loi de contrôle et raisons

Le professeur combine quatre termes :

1. Rappel radial proportionnel à l'écart entre rayon demandé et rayon propre.
2. Amortissement proportionnel à la vitesse propre, afin de stabiliser le double intégrateur.
3. Répulsion tangentielle des voisins plus proches que la corde cible, pour répartir les robots sans déplacer directement leur rayon.
4. Répulsion dans la direction de séparation sous la distance de sécurité, pour s'écarter des robots trop proches même dans une configuration radiale.

Les coefficients sont explicites dans config/circle-teacher.yaml. La sortie est une accélération demandée ; les limites physiques restent appliquées par le simulateur. La répulsion est une mesure d'évitement heuristique, sans garantie formelle de sécurité.

Le chargement vérifie que le cercle tient dans l'arène avec les disques physiques, que la distance de sécurité dépasse le diamètre, que l'espacement est réalisable et que la portée couvre cet espacement et la sécurité. Un changement manuel de configuration exige de refaire cette validation.

## Symétries et limites

Au centre exact, la direction radiale est fixée à +x pour éviter une division par zéro. Deux robots exactement superposés avec même vitesse et même observation reçoivent la même commande : aucun bruit ni identifiant caché ne brise cette symétrie. Cette situation doit être rapportée comme une limite ; elle ne doit pas être corrigée par une affectation globale.

Les gains ne garantissent pas l'absence de collisions depuis n'importe quel état, notamment à vitesse relative élevée ou si les robots se découvrent trop tard. Les tests montrent une convergence pour un robot isolé et un cercle perturbé, une séparation d'une paire proche au repos et un équilibre circulaire régulier. Ils ne constituent pas une campagne de réussite sur positions dispersées. Celle-ci nécessite les quatre métriques de la tâche 004 avant de valider le jalon de phase 1.

## Utilisation sans entraînement

`python -m src.reference.demo --seed 7 --output outputs/circle-reference`

La commande produit episode.npz et teacher.json (consigne et gains). Le scénario par défaut part de positions dispersées et peut échouer ; aucune réussite n'est déclarée avant l'évaluation. Aucun ancien modèle ni entraînement n'est utilisé. L'interface historique n'est pas encore migrée.
