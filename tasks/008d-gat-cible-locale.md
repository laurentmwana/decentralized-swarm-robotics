# 008d — GAT à cible locale et imitation multi-formes

Statut : modèle, deux entraînements et comparaison finale réalisés ; 348 tests ciblés réussis. Prototype GAT protégé validé sur les neuf formes test ; sécurité du brut insuffisante, conservée comme limite. Suite fonctionnelle : transitions 008e.

## Résultat final

- Corpus élargi : 214/214 épisodes professeur réussis ; test inchangé depuis le pilote. Cinq départs train, deux validation et un test.
- Second essai : 23 époques, meilleure époque 11, MSE validation 0,1125160754.
- Validation unique (18 épisodes) : professeur et protégé 18/18 ; brut 0/18, 70 contacts.
- Test final unique (9 formes réservées, un départ commun) : professeur 9/9 sans contact, brut 0/9 avec 54 contacts, protégé 9/9 sans contact.
- Erreur finale moyenne du protégé : 0,012690 m ; stabilisation moyenne 30,433 s, contre 29,833 s pour le professeur. Aucun temps imputé au brut en échec.
- Le filtre classique intervient sur 5 323 commandes de mouvement du protégé et rencontre 207 infaisabilités robot/pas sur l'ensemble du scénario test. La sécurité observée n'est pas attribuée au seul apprentissage.
- Poids figés avant test et conservés : `outputs/imitation/training-multiform-expanded-v1/best.pt`. Aucun réglage après examen du test.

Preuves finales : `tasks/evidence/008d-local-target-expanded-{run,validation,test}.json`. La version protégée est le prototype fonctionnel retenu, sans effacer le résultat brut. Une amélioration de celui-ci demandera une nouvelle réserve finale.

Les éléments « en cours » ci-dessous conservent l'historique expérimental et sont remplacés par ce résultat.

## Réalisation en cours

- `LocalTargetGAT` : encodeur propre/cible et messages physiques relatifs, attention à un échange direct, sortie bornée et checkpoint distinct du cercle.
- Apprentissage sur train/validation uniquement, phases 0/3 et alias dédupliqués ; aucun accès aux cibles privées voisines.
- Rejeu apparié professeur / GAT brut / GAT protégé, avec temps classiques et compteurs d'interventions séparés.
- Premier essai pilote : meilleur modèle à l'époque 1, arrêt à 16, MSE validation 0,205807. Sur les neuf validations uniques : brut 0/9 avec 45 contacts, protégé 9/9 avec zéro contact ; l'effet du filtre n'est pas attribué à l'apprentissage.
- Extension conservant test : cinq départs train et deux validation, 214/214 épisodes professeur réussis, 180 train et 18 validation uniques ; 531 360 et 53 136 exemples appris respectivement.
- Deuxième essai : initialisation fraîche, maximum 80 époques, sélection sur validation. Évaluation test attendue après gel des poids.

Décision : `docs/decisions/23-gat-cible-locale.md`. Rapports et preuves : `tasks/evidence/008d-*`.

## Entrées autorisées

Utiliser le contrat v2 de 008c : observation et cible propres, phase, effectif, positions/vitesses relatives des voisins visibles. Le futur modèle ne doit pas recevoir les cibles privées voisines via un encodeur global des lignes de caractéristiques.

Prévoir un encodeur de l'état propre et de sa cible, un encodeur de messages relatifs à quatre composantes (éventuellement distance dérivée), une attention partagée à un échange direct et une sortie d'accélération bornée. Le modèle doit avoir un type et un format de sauvegarde distincts des checkpoints cercle.

## Apprentissage et évaluation

1. Charger seulement train/validation, appliquer le masque des phases 0/3 et exclure de l'optimisation/sélection les épisodes `duplicate_within_split_of`.
2. Vérifier localité, permutation des voisins, robots isolés, bornes, dérivées et sauvegardes avant un entraînement réel.
3. Garder le test réservé pour l'évaluation finale du modèle figé ; ne pas sélectionner les poids ou hyperparamètres sur les formes test.
4. Partager le même rendez-vous, freinage et affectation classiques entre professeur et GAT, avec leurs délais dans les métriques.
5. Distinguer contrôle appris brut et variante avec filtre classique éventuel ; mesurer les interventions plutôt que les attribuer à l'apprentissage.
6. Évaluer les quatre métriques en boucle fermée sur les mêmes départs et conserver échecs/contacts. Le jeu 008c ne comporte qu'un départ par ensemble ; étendre le domaine si la couverture ne suffit pas.

L'autorisation d'entraînement donnée dans cette conversation reste pertinente ; elle ne remplace pas les vérifications de contrat, de partition et de qualité préalables.
