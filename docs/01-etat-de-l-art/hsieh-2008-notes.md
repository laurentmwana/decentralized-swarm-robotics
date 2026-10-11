# Note de lecture partielle — Hsieh, Kumar et Chaimowicz (2008)

## Référence

Hsieh, M. A., Kumar, V. et Chaimowicz, L. (2008). *Decentralized controllers for shape generation with robotic swarms*. Robotica, 26(5), 691–701. DOI : [10.1017/S0263574708004323](https://doi.org/10.1017/S0263574708004323).

## Problème traité

Les auteurs cherchent à faire générer à un essaim une forme géométrique plane décrite par une courbe fermée simple. Le contrôleur doit rester décentralisé : aucun robot ne connaît l'état des autres robots ni ne communique leur état complet.

## Méthode confirmée

Chaque robot mesure les positions relatives de voisins présents dans une portée connue. Les interactions locales servent à éviter les collisions ou à maintenir des contraintes de distance relative. L'article annonce des propriétés de stabilité et de convergence pour une classe de courbes fermées, puis des simulations et des extensions à des courbes planes plus générales.

## Utilité pour ce mémoire

Cette publication fournit une référence non apprise, directement reliée à la formation de formes 2D avec perception locale. Elle peut inspirer une baseline de type attraction vers la forme, répulsion à courte portée et maintien de connectivité, à comparer au GNN.

## Limites par rapport au projet

Le cadre concerne des courbes géométriques, alors que le jeu de données prévu comprend aussi lettres, chiffres et nombres. Le résumé consulté ne fournit pas un protocole de comparaison avec un modèle appris, ni une évaluation des pannes de robots.

## Statut

Lecture partielle documentée à partir du résumé détaillé et des métadonnées de l'éditeur. Lire l'article complet avant de reprendre les équations ou les résultats comme références scientifiques dans le mémoire.
