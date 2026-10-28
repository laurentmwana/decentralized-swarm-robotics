# 006 — GAT à une couche et MSE supervisée

Statut : validée par l’utilisateur ; commit autorisé.
Décisions : docs/decisions/13-gat-imitation.md.

## Livraison

- Encodeur à deux couches denses, attention à un échange direct, décodeur dense et sortie d'accélération bornée.
- Poids partagés, tailles variables, traitement des robots isolés et masque strict des voisins.
- Entrées compatibles avec le jeu 005, normalisation locale explicite et degré local.
- Objectif MSE sur commandes du professeur ; aucune boucle d'entraînement.
- Sauvegarde/relecture versionnée et configuration YAML.

## Vérification

203 tests ciblés réussis : 155 précédents, 36 nouveaux du GAT et 12 historiques du simulateur.
Les tests de dérivées n'effectuent aucune mise à jour de poids. Aucun optimiseur ni entraînement exécuté.
Les autres tests historiques n'ont pas été exécutés ; les 12 tests du simulateur précédemment bloqués par l'absence de PyTorch passent désormais.
Le modèle n'est pas entraîné et aucune qualité de contrôle n'est encore affirmée.

## Suite après validation et commit

Préparer l'intégration du GAT dans les épisodes, les commandes d'entraînement à exécuter par l'utilisateur et le protocole de comparaison. Avant tout entraînement, traiter le statut review_required du jeu 005 (épisode 113). Les poids initiaux testés ne remplacent pas un modèle appris.
