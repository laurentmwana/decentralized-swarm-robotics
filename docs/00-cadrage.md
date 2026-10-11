# Étape 0 — Cadrage

## Objectif
Poser clairement ce que le mémoire étudie, pourquoi, et ce qu'il ne traite pas.

## Contexte
Un essaim de robots est un groupe de petits robots simples qui coopèrent sans chef. Chaque robot ne voit que ses voisins proches.
Les comportements collectifs (se rassembler, avancer en formation, former une figure) sont habituellement obtenus avec des règles écrites à la main.
Ces règles sont difficiles à concevoir pour des tâches complexes. L'idée du mémoire est de **faire apprendre ces règles à un réseau de neurones sur graphes**.

## Problématique (à finaliser)
> Un réseau de neurones sur graphes, qui n'utilise que des informations locales, peut-il apprendre à faire former des formes à un essaim de robots,
> et ce comportement est-il robuste aux pannes et au changement de taille du groupe ?

## Sous-questions
1. Quelles formes le réseau apprend-il à produire (géométriques, lettres, chiffres, nombres) et avec quelle qualité ?
2. Que se passe-t-il quand des robots tombent en panne ?
3. Un réseau entraîné avec peu de robots fonctionne-t-il avec beaucoup plus de robots ?
4. Que gagne-t-on par rapport à des règles écrites à la main ?

## Périmètre
**Dans le mémoire** : simulation en 2D, robots simples, communication locale, formes dans un plan, évaluation en simulation.
**Hors périmètre (à mentionner comme perspectives)** : robots physiques, obstacles complexes, 3D, communication avec délais ou pertes réalistes.

## Hypothèses de travail (à valider)
- Les robots se déplacent dans un plan sans obstacles fixes.
- Chaque robot connaît sa position par rapport à ses voisins (distance et direction).
- La consigne de forme est connue du groupe d'une manière précisée à l'étape 3.
- Le temps est découpé en pas réguliers.

## Décisions validées pour la première version

- L'intitulé déposé contenant « réseaux de neurones cellulaires » est conservé provisoirement. Un éventuel changement vers « réseaux de neurones sur graphes » requiert l'accord explicite du professeur.
- Les robots utilisent un repère commun 2D et accèdent aux positions relatives de leurs voisins.
- Chaque robot reçoit l'identifiant partagé de la forme cible (option A de l'étape 3).
- Le périmètre reste une simulation 2D sans obstacles fixes ni robots physiques ; les décisions de mouvement reposent exclusivement sur les observations locales et la consigne commune.

## À faire
- [ ] Rédiger la problématique en une phrase et les sous-questions
- [ ] Présenter l'intitulé au professeur (version « cellulaire » ou « sur graphes ») et noter sa réponse dans `decisions.md`
- [ ] Lister ce qui est exclu
- [ ] Choisir les 2 ou 3 résultats qui feront le cœur du mémoire

## Critères de réussite
Le professeur valide la problématique ; tu peux expliquer ton sujet en deux minutes à quelqu'un qui n'est pas du domaine.

## Pièges
- Sujet trop large (essayer de faire tous les comportements). Priorité : la formation de formes.
- Promettre du matériel réel alors que tout reste en simulation.
