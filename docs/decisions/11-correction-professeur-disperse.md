# 11 — Corriger la lenteur de répartition du professeur

Date : 2026-10-28. Statut : validé par l’utilisateur (tâche 004b).

## Diagnostic à partir des trajectoires

Dans le pilote 004, les rayons finaux sont déjà voisins de 2 m. Exemple graine 0 : rayon de 1,99989 à 2,00052 m, mais un intervalle angulaire atteint environ 0,77 rad au lieu de 0,524 rad. Pour la graine 9, un intervalle atteint environ 1,51 rad. Le défaut principal est la relaxation angulaire trop lente sous les gains initiaux, et non l'accès au cercle. La première correction doit donc porter sur le réglage de cette loi, sans ajouter une affectation globale ni changer le protocole.

## Réglage classique, sans entraînement neuronal

Un diagnostic manuel a comparé six jeux (radial, amortissement, espacement), avec répulsion de sécurité 60, sur les graines de développement 0–9 :

| Gains | Réussites / 10 |
|---|---|
| 8, 5, 20 | 1 |
| 12, 6, 60 | 9 |
| 20, 8, 100 | 9 |
| 10, 5, 100 | 9 |
| 20, 8, 200 | 0 |
| 30, 10, 300 | 0 |

Retenir le premier réglage performant : radial 12, amortissement 6, espacement 60, répulsion 60. Un espacement trop fort produit un mouvement résiduel incompatible avec le seuil de stabilisation malgré une faible erreur géométrique. La saturation d'accélération et de vitesse reste celle du moteur. Aucune modification de la loi de commande ni des métriques n'a été nécessaire. Aucun apprentissage, rétropropagation ou entraînement GAT n'est exécuté.

Le protocole 004 (0,1 m ; 0,05 m/s ; maintien 1 s ; délai 10 s), les graines, les positions initiales, le rayon et l'effectif restent identiques. Les graines de développement ont servi au réglage et ne sont donc pas un test indépendant.

## Vérification indépendante après fixation des gains

| Campagne | Graines | Réussite | Erreur finale moyenne ± écart-type | Collisions | Temps moyen sur réussites |
|---|---|---|---|---|---|
| Pilote initial, anciens gains | 0–9 | 0/10 | 0,9116 ± 0,4914 m | 1 initiale | null |
| Réévaluation du pilote | 0–9 | 9/10 | 0,00504 ± 0,00311 m | 1 initiale | 5,344 s |
| Validation sans nouveau réglage | 10–29 | 17/20 | 0,00941 ± 0,00698 m | 3 initiales | 6,265 s |

Tous les cas initialement sans chevauchement de ces deux campagnes réussissent (26/26). Les quatre échecs stricts sont les graines 9, 12, 20 et 22 : un contact initial chacune, conservé dans les résultats. Aucun événement supplémentaire n'a été détecté après l'initialisation. Cela ne garantit pas la sécurité pour toutes les configurations, les autres effectifs ou les autres portées.

Rapports complets versionnés : tasks/evidence/004b-circle-pilot.json et tasks/evidence/004b-circle-validation.json. Les trajectoires restent dans outputs/plan-v2/circle-pilot-task004b/ et outputs/plan-v2/circle-validation-task004b/.

## Conséquences pour la suite

Le professeur est utilisable pour préparer la prochaine tâche sur ce périmètre circulaire testé ; la réussite absolue depuis toute position dispersée n'est pas établie. Avant la collecte 005, définir explicitement le domaine d'initialisation (par exemple un tirage sans chevauchement) et ses tests, tout en conservant ces campagnes avec leurs échecs. Ne pas requalifier rétroactivement les épisodes initialement en collision ni en supprimer les lignes. Les conditions de portée, densité et rayon doivent rester validées.

Le cas d'une superposition parfaite et les arrivées rapides potentiellement dangereuses restent des limites connues. Les nouveaux tests de régression portent sur les neuf graines du pilote sans collision initiale, la reproduction de la lenteur avec les anciens gains et le maintien de l'échec de la graine 9.
