# Discours de soutenance

## 1. Salutations et introduction

Monsieur le Président du jury,

Honorables membres du jury,

Distingués invités,

Mesdames et Messieurs,

Bonjour.

Avant toute chose, je tiens à vous remercier pour votre présence et pour l'occasion qui m'est donnée de présenter ce travail.

Le mémoire que j'ai l'honneur de vous présenter aujourd'hui s'intitule :

**« Contrôle décentralisé d'un essaim de robots par réseaux de neurones sur graphe ».**

Ce travail s'inscrit dans le domaine de l'intelligence artificielle et de la robotique collective.

Au cours de cette présentation, je vais aborder le contexte, la problématique, les hypothèses, les objectifs, la méthodologie, les résultats obtenus et enfin la conclusion.

## 2. Contexte de l'étude

Aujourd'hui, les robots sont de plus en plus utilisés dans plusieurs domaines, notamment la surveillance, l'exploration et les opérations de recherche.

Certaines missions peuvent nécessiter l'utilisation de plusieurs robots qui travaillent ensemble. On parle alors d'un essaim de robots.

Cependant, coordonner plusieurs robots représente un défi important.

Dans un système centralisé, un seul contrôleur prend les décisions pour l'ensemble des robots.

Cette organisation peut présenter des limites, notamment lorsque le nombre de robots augmente ou lorsque le contrôleur central tombe en panne.

C'est dans ce contexte que s'inscrit notre étude, qui s'intéresse au contrôle décentralisé des essaims de robots.

## 3. Problématique

Le principal problème étudié dans ce travail concerne la coordination des robots sans dépendre d'un contrôleur central.

En effet, chaque robot doit pouvoir prendre ses propres décisions tout en tenant compte des robots qui l'entourent.

Cela nous conduit à la question principale suivante :

**Comment permettre à un essaim de robots de se coordonner de manière décentralisée en utilisant des réseaux de neurones sur graphe ?**

De cette question découlent d'autres interrogations :

- Comment représenter les interactions entre les robots ?
- Comment permettre à chaque robot de calculer son déplacement à partir des informations de ses voisins ?
- Comment évaluer le comportement collectif obtenu ?

## 4. Hypothèse de recherche

Pour répondre à cette problématique, l'hypothèse principale de ce travail est la suivante :

**L'utilisation d'un réseau de neurones sur graphe pourrait permettre à chaque robot de prendre des décisions locales à partir des informations de ses voisins, tout en favorisant une coordination collective de l'essaim.**

Cette hypothèse repose sur le fait qu'un GNN est capable de traiter des informations organisées sous forme de graphe.

Dans notre cas, chaque robot représente un nœud, tandis que les relations entre robots voisins représentent les arêtes.

## 5. Objectifs de l'étude

L'objectif général de ce travail est de développer et d'évaluer, en simulation, une approche de contrôle décentralisé d'un essaim de robots à l'aide d'un réseau de neurones sur graphe.

Plus précisément, ce travail poursuit plusieurs objectifs.

Premièrement, développer un environnement de simulation permettant de représenter le déplacement et les interactions des robots.

Deuxièmement, représenter les relations entre les robots sous forme de graphe.

Troisièmement, mettre en place un contrôleur GNN capable de calculer les commandes de déplacement des robots.

Quatrièmement, entraîner ce modèle à partir de données générées dans l'environnement de simulation.

Enfin, évaluer les performances du système à l'aide de métriques adaptées.

## 6. Méthodologie

Pour atteindre ces objectifs, une approche expérimentale fondée sur la simulation a été adoptée.

Dans un premier temps, un environnement bidimensionnel a été utilisé pour représenter les robots, leurs positions et leurs vitesses.

Ensuite, les interactions entre les robots ont été modélisées sous forme de graphe.

Chaque robot constitue un nœud et les relations de voisinage constituent les arêtes.

À partir de cette représentation, un réseau de neurones sur graphe a été utilisé pour traiter les informations locales et produire des commandes de déplacement.

L'apprentissage du modèle repose notamment sur des exemples de comportements générés dans le simulateur.

Enfin, des expérimentations permettent d'observer le comportement collectif et d'évaluer les performances du contrôleur.

## 7. Résultats obtenus

Les expérimentations permettent d'étudier la capacité du contrôleur GNN à reproduire des comportements de déplacement collectif.

L'évaluation porte notamment sur la qualité des déplacements, la coordination entre les robots et les distances de séparation.

Une attention particulière est accordée aux collisions, car une bonne coordination ne doit pas seulement permettre aux robots de se déplacer ensemble, mais aussi de maintenir des distances suffisantes.

Les résultats quantitatifs seront présentés à partir des mesures et des graphiques issus des expérimentations.

## 8. Limites et perspectives

Comme tout travail expérimental, cette étude présente certaines limites.

La première concerne l'utilisation d'un environnement simulé plutôt que de robots physiques.

La deuxième concerne la prise en compte simplifiée de certaines contraintes réelles, comme les erreurs des capteurs, les retards de communication et les limites des moteurs.

À l'avenir, cette approche pourrait être étendue à des robots physiques équipés de capteurs et de systèmes de communication.

Il serait également intéressant d'améliorer la gestion des collisions et d'étudier le comportement du système dans des environnements plus complexes.

## 9. Conclusion

Au terme de ce travail, nous avons étudié une approche de contrôle décentralisé d'un essaim de robots à l'aide des réseaux de neurones sur graphe.

L'idée principale consiste à permettre à chaque robot d'utiliser les informations de ses voisins pour prendre une décision locale.

Cette approche offre un cadre intéressant pour étudier la coordination collective sans recourir à un contrôleur central pour chaque déplacement.

Les expérimentations en simulation permettent d'évaluer les possibilités de cette méthode, mais aussi d'identifier les aspects qui nécessitent encore des améliorations.

Monsieur le Président du jury,

Honorables membres du jury,

Je vous remercie pour votre aimable attention.

**Je reste à votre disposition pour vos questions et observations.**