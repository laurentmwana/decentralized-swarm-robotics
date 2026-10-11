# Étape 7 — Rédaction du mémoire (LaTeX)

## Objectif
Écrire le mémoire au fur et à mesure du projet, pas à la fin.

## Mise en forme demandée
- Texte principal en **taille 12 pt**.
- **Interligne adapté à un document scientifique** (environ 1,5).
- Marges standard, pages numérotées, table des matières, liste des figures et des tableaux.
- Équations peu nombreuses : privilégier les explications en mots, avec formules seulement quand elles sont indispensables.
- Figures et tableaux numérotés, avec légende explicite et appelés dans le texte.
- Bibliographie avec BibTeX ou BibLaTeX, style cohérent.

## Exemple d'en-tête LaTeX
```latex
\documentclass[12pt,a4paper]{report}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[french]{babel}
\usepackage{setspace}
\onehalfspacing
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage[backend=biber,style=numeric]{biblatex}
\addbibresource{references.bib}
```

## Plan type du mémoire
1. **Introduction** : contexte, problème, question de recherche, contributions, plan.
2. **État de l'art** : essaims, réseaux cellulaires, GNN, apprentissage multi-agents, formation de formes.
3. **Méthode** : simulateur, formes cibles, modèle, objectif d'apprentissage.
4. **Expériences** : protocole, résultats, comparaisons, robustesse.
5. **Discussion** : ce que montrent les résultats, limites, échecs.
6. **Conclusion et perspectives** : réponse à la question, pistes (robots réels, formes plus complexes, obstacles).
7. **Annexes** : paramètres, configurations, résultats complémentaires.

## Rythme conseillé
- Rédiger le chapitre de méthode dès que le simulateur et le modèle sont stables.
- Écrire l'état de l'art en parallèle de la lecture.
- Rédiger les résultats juste après chaque expérience, tant que les détails sont frais.

## Règles de style
- Phrases courtes, termes définis à la première apparition.
- Chaque figure répond à une question posée dans le texte.
- Chaque chapitre commence par un court paragraphe d'annonce et finit par un résumé.
- Distinguer clairement ce qui vient de la littérature et ce que tu apportes.

## À faire
- [ ] Squelette LaTeX compilable
- [ ] Fichier de bibliographie
- [ ] Chapitres rédigés en parallèle du projet
- [ ] Relecture complète (orthographe, cohérence des notations, références croisées)
- [ ] Relecture par le professeur ou un camarade

## Critères de réussite
Un lecteur du domaine peut reproduire ton travail à partir du texte ; la mise en forme respecte les consignes.

## Pièges
- Tout écrire dans les dernières semaines.
- Notations qui changent d'un chapitre à l'autre.
- Figures illisibles en noir et blanc.
