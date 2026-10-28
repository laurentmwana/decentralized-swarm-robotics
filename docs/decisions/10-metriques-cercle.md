# 10 — Quatre métriques et protocole circulaire

Date : 2026-10-28. Statut : validé par l’utilisateur (tâche 004).

## Erreur de formation en mètres

Pour chaque état, soustraire le centre demandé aux positions. Calculer :

- le maximum des écarts absolus au rayon demandé ;
- trier les angles autour du centre, inclure l'intervalle bouclant sur 2π, puis calculer `R * max(abs(intervalle - 2π/N))`.

L'erreur de formation est le maximum de ces deux termes, en mètres. Elle contrôle le rayon ET la répartition : des robots tous au même point du cercle ne doivent pas avoir une erreur nulle. Elle est invariante à l'ordre des robots et à une rotation collective, sans affectation des robots à des places. Pour N=1, seul l'écart radial intervient. Au centre exact, l'angle conventionnel de numpy est utilisé ; l'écart radial signale toujours ce défaut. Le maximum empêche qu'un robot défaillant soit masqué par une moyenne. Cette définition concerne le cercle ; ligne et V nécessiteront une définition adaptée.

Le rapport d'épisode affiche l'erreur au dernier état disponible avant ou au délai maximal. L'évaluateur est global et hors contrôle : ses calculs ne sont jamais transmis au professeur.

## Stabilisation en secondes et réussite

Protocole fixé dans config/circle-metrics.yaml avant le pilote : erreur <= 0,1 m, vitesse maximale des robots <= 0,05 m/s, maintien >= 1 s, délai 10 s.

Chercher le premier début d'une fenêtre continue respectant erreur et vitesse sur tous ses états. Une durée de maintien correspond à des intervalles de temps, pas au nombre d'images : arrondi supérieur au nombre de pas si nécessaire. La fenêtre complète doit finir au plus tard au délai. Un épisode trop court est refusé ; si le délai tombe entre deux états, seul le dernier état antérieur est pris en compte (convention conservatrice).

La réussite exige cette stabilisation ET zéro collision sur tout le scénario évalué. Tout échec reçoit `stabilization_time_s: null`, jamais le délai maximal ou zéro. Un état initial stable et sans contact peut réussir avec un temps égal à zéro. Le seuil de vitesse précise la stabilisation physique et n'ajoute pas une cinquième métrique.

Le taux de réussite vaut 100 × réussites / épisodes. La moyenne des temps porte uniquement sur les réussites, avec leur effectif explicite ; elle est null si aucune réussite. La moyenne et l'écart-type descriptif (population) des erreurs et le total des collisions sont également exportés.

## Nombre de collisions

Un événement est un contact d'une paire non ordonnée de disques : distance <= deux rayons physiques (tolérance numérique de 1e-12 sur la distance au carré). Le contact initial compte. Un contact prolongé compte une fois ; une séparation suivie d'un nouveau contact compte à nouveau.

Pour chaque intervalle, minimiser analytiquement la distance relative sur le segment interpolé entre les positions enregistrées : une traversée entre deux états est donc détectée. Un contact nouvellement atteint compte si la paire n'était pas en contact au début de l'intervalle. L'interpolation correspond au déplacement linéaire entre états ; elle ne reconstitue pas des sous-pas physiques cachés. Les contacts au dernier état sont inclus. Les collisions entre robots ne sont jamais corrigées automatiquement.

Les contacts initiaux sont signalés séparément dans `initial_collision`, tout en restant comptabilisés et éliminatoires. Cela distingue une difficulté d'initialisation d'une collision apparue pendant le contrôle, sans masquer les échecs. Les contacts avec les murs ne font pas partie de ce nombre de collisions entre robots ; les bords sont contraints par le moteur.

## Campagne et résultat du pilote

Commande : `python -m src.reference.evaluate --output outputs/plan-v2/circle-evaluation-task004`.
Graines 0 à 9, 12 robots, rayon cible 2 m, portée 2 m, initialisation dispersée dans un disque de rayon 4 m. Aucune optimisation, recherche de gains ni entraînement. Chaque épisode produit une archive NPZ ; report.json contient la configuration physique, la consigne, les gains, le protocole, les résultats individuels et l'agrégat. Une copie légère du rapport est versionnée dans tasks/evidence/004-circle-pilot.json ; les archives volumineuses restent dans outputs/ (ignoré par Git).

Résultat : 0/10 réussite, erreur finale moyenne 0,9116 m (écart-type 0,4914 m), un contact initial pour la graine 9, aucun temps de stabilisation attribué. Ce pilote à effectif et portée fixes ne mesure pas encore le passage à l'échelle ni la robustesse de communication.

Conséquence : le jalon de phase 1 n'est PAS validé. Ajouter une tâche 004b de diagnostic et correction du professeur avant la collecte d'imitation. Les tests favorables proches du cercle ne suffisent pas à attester la convergence depuis des positions dispersées. Ne pas assouplir rétroactivement les seuils pour transformer le pilote en succès.
