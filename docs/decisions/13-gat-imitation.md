# 13 — GAT local à un échange et objectif d'imitation

Date : 2026-10-28. Statut : validé par l’utilisateur (tâche 006).

## Architecture et respect du plan

Le nouveau modèle AccelerationGAT est séparé du SwarmGNN historique commandant des vitesses. Il reçoit le schéma des huit caractéristiques de la collecte 005 et le masque booléen de communication. Tous les robots partagent les mêmes poids, indépendamment de leur nombre.

Prétraitement local : soustraire le centre de consigne à la position propre, puis normaliser position et rayon par spatial_scale, vitesse par speed_scale, effectif par count_scale. L'entrée encodée possède donc six composantes : position relative au centre (2), vitesse (2), rayon (1), effectif (1). Le modèle est invariant à une translation simultanée des positions et de la consigne. Aucune normalisation calculée sur validation/test, ni moyenne globale de l'essaim.

Blocs : deux couches denses avec ReLU ; une unique couche d'attention additive à une tête avec projection linéaire et LeakyReLU ; concaténation du propre encodage, du message agrégé et du degré local normalisé ; une couche dense avec ReLU et une sortie à deux composantes.

Le degré est une information locale calculable depuis les voisins : l'attention normalise la somme des poids à un, tandis que le professeur additionne des interactions. Fournir le degré évite de perdre cette information de cardinalité, sans ajouter un second échange. Le propre encodage est conservé dans le décodeur : un robot sans voisin doit pouvoir rejoindre le cercle et freiner.

Le graphe physique est reconstruit par le simulateur. Le modèle refuse les auto-boucles du contrat. L'attention masque les arêtes absentes avant le softmax. Pour un robot isolé, les scores sont temporairement remplacés par zéro avant softmax, puis les poids absents sont annulés : agrégat nul et dérivées finies. Les messages utilisent uniquement les encodages initiaux des voisins, jamais leurs agrégats : un robot à deux liens ne peut pas influencer la commande à ce pas.

Implémentation dense O(N²), simple pour la première version ; aucune généralisation à de très grands essaims n'est affirmée. Une implémentation creuse pourra être étudiée si les mesures le justifient.

## Sortie et MSE

La sortie brute est projetée dans le disque unité par `raw / max(1, ||raw||)`, puis multipliée par a_max. La norme euclidienne de commande respecte donc a_max, plutôt que de borner séparément x et y. La saturation indépendante du simulateur reste une protection du contrat physique. Le modèle prédit des accélérations en m/s² et ne peut pas remplacer directement un ancien modèle en vitesse.

La fonction acceleration_mse calcule la moyenne des carrés des écarts entre les deux composantes prédites et les accélérations bornées du professeur, sur tous les exemples fournis. Unité : m²/s⁴. Elle refuse formes, types, appareils ou valeurs non compatibles. Elle ne contient ni optimiseur ni boucle d'entraînement. Le futur découpage en lots devra conserver les graphes complets de chaque pas et la partition par épisodes.

## Configuration, initialisation et sauvegarde

config/imitation-model.yaml fixe dimension 64, échelles 4 m / 1 m/s / 12 robots, pente 0,2 et graine 42. a_max vient de la configuration physique. L'initialisation utilise un contexte RNG isolé pour reproduire les poids sans perturber l'aléa de l'appelant.

Les sauvegardes contiennent type/version, configuration, limite d'accélération, schéma et state_dict. Chargement strict, weights_only=True, appareil explicite ; refus des anciennes sauvegardes et d'une réécriture d'un fichier existant. Une sauvegarde de poids n'atteste aucun entraînement. Les fichiers créés dans les tests sont des poids initiaux temporaires ; aucun modèle entraîné n'est livré.

## Vérification sans entraînement

36 nouveaux tests : voisinage direct, masque, robot isolé, tailles variables, permutation, translation, lots, saturation euclidienne, initialisation, sauvegarde/relecture, MSE et compatibilité des archives de collecte. Les calculs de dérivées vérifient leur finitude et l'absence d'influence distante, sans optimiseur, sans mise à jour des poids et sans apprentissage.

203 tests ciblés réussis : 155 de la migration antérieure, 36 du GAT et 12 du simulateur historique. PyTorch CPU a été installé dans un dossier temporaire pour ces vérifications. L'ensemble des autres tests historiques n'a pas été exécuté. Preuve légère : tasks/evidence/006-gat-verification.json.

## Précondition de la suite

La tâche 007 reste responsable du lancement par l'utilisateur de tout entraînement, de la sélection sur validation et de l'évaluation indépendante. Avant de lui proposer une commande d'entraînement, résoudre ou définir explicitement le traitement de qualité du jeu 005 (deux collisions pendant l'épisode 113). Cette tâche ne contourne pas le statut review_required et ne mesure pas encore les performances d'un modèle appris.
