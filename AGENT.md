# AGENT.md — Guide pour l'assistant de code

Ce fichier explique à un agent (ou à un humain qui reprend le projet) **quoi construire, comment, et selon quelles règles**.
Lis-le en entier avant toute modification, puis lis `PLAN.md` et le fichier de `docs/` de l'étape en cours.

## 1. Projet en une phrase

Un mémoire de master (IA et data science) sur le **contrôle décentralisé d'un essaim de robots par réseaux de neurones sur graphes (GNN)**.
Chaque robot ne perçoit que ses voisins proches ; un même réseau de neurones, copié dans chaque robot, décide de son mouvement ;
le groupe doit former une **forme cible** (cercle, lettres, chiffres, nombres) et rester robuste aux pannes et au changement de taille.

> Intitulé déposé : « Contrôle décentralisé d'un essaim de robots par réseaux de neurones cellulaires (CeNN-Inspired Swarm Robotics) ».
> Intitulé envisagé : « ... par réseaux de neurones sur graphes ». Le changement est **à valider avec le professeur** : ne pas renommer
> le mémoire de façon définitive sans cet accord. L'inspiration « cellulaire » (chaque robot agit comme une cellule qui ne voit que ses voisins) reste un principe à expliquer dans le texte.

## 2. Question de recherche

Un réseau de neurones peut-il apprendre, à partir d'informations **uniquement locales**, un comportement collectif efficace
(former des formes), et ce comportement résiste-t-il (a) aux pannes de robots, (b) à un nombre de robots différent de celui de l'entraînement ?

## 3. Principes non négociables

1. **Décentralisé** : aucune décision ne doit dépendre d'un chef ou d'une information globale non justifiée. Tout ce qu'un robot reçoit doit venir de ses voisins dans sa portée de communication, ou de la consigne de forme (documentée dans `docs/03`).
2. **Un seul réseau partagé** : les paramètres sont identiques pour tous les robots. Le nombre de voisins peut varier.
3. **Reproductibilité** : toute expérience est pilotée par un fichier de configuration et une graine aléatoire. Pas de résultat sans config sauvegardée.
4. **Séparation propre** : simulateur, génération de données, modèle, entraînement, évaluation sont des modules distincts.
5. **Honnêteté scientifique** : toujours comparer à des références simples (voir `docs/06`). Les échecs et limites sont à documenter, pas à cacher.

## 4. Stack technique recommandée

- Python 3.10 ou plus
- PyTorch pour le modèle ; PyTorch Geometric (optionnel) ou une implémentation maison de l'échange de messages entre voisins
- NumPy, SciPy (voisinage), Pillow (images des formes), matplotlib (courbes et animations)
- PyYAML pour les configurations ; pytest pour les tests
- LaTeX pour le mémoire (voir `docs/07`)

## 5. Structure de dépôt cible

```
swarm-gnn/
  AGENT.md  PLAN.md  README.md
  configs/            # un fichier YAML par expérience
  data/shapes/        # images générées + nuages de points cibles
  docs/               # documentation détaillée par étape
  src/
    sim/              # simulateur (dynamique, voisinage, pannes)
    shapes/           # génération et chargement des formes cibles
    models/           # GNN, références (baselines)
    train/            # boucle d'entraînement, pertes
    eval/             # métriques, scénarios de test
    viz/              # courbes, animations
  tests/
  results/            # sorties (ignorées par git, sauf résumés)
  thesis/             # sources LaTeX du mémoire
```

## 6. Conventions de code

- Fonctions courtes, noms explicites en anglais pour le code, commentaires et docs en français.
- Typage (`typing`) et docstrings courtes sur toute fonction publique.
- Pas de nombre magique : tout paramètre (rayon de communication, vitesse max, pas de temps...) vient de la config.
- Un test au minimum par module de `src/` (voir `docs/06` pour les tests de comportement).
- Commits petits, messages au format : `etape-N: description courte`.

## 7. Règles de travail pour l'agent

- Travaille **une étape à la fois**, dans l'ordre de `PLAN.md`. Ne saute pas d'étape sans le dire.
- Avant de coder, relis la section « Critères de réussite » du fichier de l'étape.
- Après chaque étape : lance les tests, mets à jour la case correspondante dans `PLAN.md`, note les décisions dans `docs/decisions.md`.
- Si une consigne est ambigüe (ex. la forme à atteindre est-elle connue de tous les robots ?), **pose la question** au lieu de choisir en silence.
- Ne modifie pas les données de test après avoir vu les résultats d'un modèle dessus.
- Ne lance pas d'entraînements très longs sans prévenir : propose d'abord une version réduite (peu de robots, peu d'itérations).

## 8. Définition de « terminé » pour une étape

- Code propre, testé, configuré.
- Résultat visuel ou chiffré sauvegardé dans `results/`.
- Courte note dans `docs/decisions.md` (ce qui a été choisi et pourquoi).
- Case cochée dans `PLAN.md`.
