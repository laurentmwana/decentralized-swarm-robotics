# Tableau comparatif provisoire

Ce tableau organise la lecture. Les cellules marquées « à vérifier » proviennent uniquement des notices et résumés consultés ; elles doivent être validées pendant la lecture intégrale avant toute utilisation dans le mémoire.

| Travail | Information à l'exécution | Approche | Tâche | Robustesse / échelle | Lien avec le mémoire | État |
|---|---|---|---|---|---|---|
| Reynolds (1987) | Voisins proches | Règles comportementales écrites à la main | Flocking | À vérifier dans le texte complet | Baseline cohésion, séparation et alignement | À lire |
| Olfati-Saber (2006) | Interactions de voisinage | Contrôle distribué théorique | Flocking | À vérifier dans le texte complet | Baseline de contrôle distribué et critères de cohésion | À lire |
| Tolstaya et al. (2019) | Information locale et échanges sur graphe mobile | GNN, imitation d'un contrôleur centralisé pendant l'entraînement | Flocking | Étudie le rayon de communication et des graphes mobiles | Référence la plus directe pour un contrôleur GNN local partagé | À lire en priorité |
| Gama, Tolstaya et Ribeiro (2020) | Information locale, communications retardées | GNN pour contrôleur décentralisé | Flocking | Équivariance et transférabilité revendiquées ; détails à vérifier | Justifie les propriétés de permutation et de généralisation recherchées | À lire en priorité |
| Gama et al. (2021) | Information locale et communication entre voisins | GNN et apprentissage par imitation | Flocking, planification multi-agent | Passage à des équipes plus grandes revendiqué ; protocole à vérifier | Point de comparaison pour choisir apprentissage direct ou imitation | À lire en priorité |
| Hsieh, Kumar et Chaimowicz (2008) | Positions relatives des voisins dans une portée connue | Contrôleur décentralisé fondé sur la géométrie | Courbes planes fermées | Propriétés de stabilité et convergence pour une classe de courbes ; conditions à étudier | Baseline de formation de formes directement liée à la cible 2D | À lire en priorité |
| Li et al. (2018) | Décisions locales via un graphe de communication | Contrôle de formation progressive | Forme cible | À vérifier pendant la lecture complète | Référence de formation progressive sans contrôleur appris | À lire en priorité |
| Jiang, Huang et Guo (2023) | LiDAR local et voisinage de proximité | CNN + GNN, apprentissage par démonstration | Formation triangulaire | Entraîné avec 5 robots, évalué de 3 à 9 robots en simulation | Référence GNN la plus directe pour la formation ; contraste avec notre simulation à états plus simple et nos formes variées | Lecture partielle documentée |
| Gilmer et al. (2017) | Graphe de voisinage | Message passing neural network | Chimie moléculaire | Non applicable directement | Formalisme d'agrégation de messages pour l'architecture | À lire |
| Battaglia et al. (2018) | Structures relationnelles sous forme de graphes | Graph networks | Cadre général | Non applicable directement | Fondement conceptuel du biais inductif relationnel | À lire |

## Positionnement à démontrer après lecture

Le projet vise un contrôleur GNN partagé, exécuté de manière décentralisée, pour former des représentations géométriques et symboliques. Son protocole d'évaluation doit étudier explicitement les pannes et l'augmentation de la taille de l'essaim. Cette formulation est un positionnement de travail, non une affirmation de nouveauté : elle sera révisée après la lecture complète de la bibliographie et l'ajout d'approches de formation de formes directement comparables.
