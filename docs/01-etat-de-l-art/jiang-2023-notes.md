# Note de lecture partielle — Jiang, Huang et Guo (2023)

## Référence

Jiang, C., Huang, X. et Guo, Y. (2023). *End-to-end decentralized formation control using a graph neural network-based learning method*. Frontiers in Robotics and AI, 10, 1285412. DOI : [10.3389/frobt.2023.1285412](https://doi.org/10.3389/frobt.2023.1285412).

## Ce qui est confirmé dans l'article consulté

- Chaque robot utilise une observation locale LiDAR, transformée en carte d'occupation, et les échanges inter-robots sont représentés par un GNN.
- La politique est entraînée à imiter un contrôleur expert pour former une structure triangulaire ; il s'agit donc d'apprentissage par démonstration, et non d'apprentissage par renforcement.
- L'entraînement utilise cinq robots ; les expériences simulées évaluent des équipes de trois à neuf robots.
- L'article présente la méthode comme décentralisée à l'exécution et extensible à des tailles d'équipe différentes.

## Utilité pour ce mémoire

Cette étude est le précédent le plus proche de l'architecture cible : politique GNN partagée, perception locale et passage à une autre taille d'équipe. Notre projet se distinguera par une dynamique 2D plus simple au départ, des formes cibles plus nombreuses et un protocole explicite de retrait de robots en panne.

## Limites par rapport au projet

La tâche porte sur une formation triangulaire avec distances prescrites, pas sur la représentation de lettres, chiffres ou nombres. Les expériences de robustesse aux pannes ne constituent pas le protocole principal présenté.

## Statut

Lecture partielle documentée à partir du texte HTML de l'article (résumé, formulation, protocole et conclusion). Une lecture intégrale avec vérification des figures et résultats chiffrés reste nécessaire avant d'utiliser cette note dans le mémoire.
