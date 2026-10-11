# PLAN.md — Feuille de route du projet

Cocher chaque case au fil de l'avancement. Le détail de chaque étape est dans `docs/`.
Les durées sont indicatives pour un travail à temps partiel ; à adapter à la date de dépôt réelle.

| # | Étape | Document | Durée indicative | Livrable principal |
|---|-------|----------|------------------|--------------------|
| 0 | Cadrage et problématique | docs/00 | 1 semaine | Question de recherche validée par le professeur |
| 1 | État de l'art | docs/01 | 2 semaines | Revue de littérature (fiches + synthèse) |
| 2 | Simulateur d'essaim | docs/02 | 2 semaines | Simulation fonctionnelle et testée |
| 3 | Jeu de formes cibles | docs/03 | 1 semaine | Images + nuages de points + archive zip |
| 4 | Modèle GNN | docs/04 | 2 semaines | Réseau qui produit des mouvements |
| 5 | Entraînement | docs/05 | 2 à 3 semaines | Modèle entraîné + courbes |
| 6 | Évaluation et expériences | docs/06 | 2 à 3 semaines | Tableaux de résultats + figures |
| 7 | Rédaction du mémoire | docs/07 | 4 semaines (en parallèle dès l'étape 2) | Mémoire complet en LaTeX |
| 8 | Soutenance | docs/08 | 1 à 2 semaines | Diapositives + démonstration |

## Cases à cocher

### Étape 0 — Cadrage
- [x] Question de recherche rédigée en une phrase
- [ ] Intitulé confirmé avec le professeur (version « cellulaire » ou « sur graphes »)
- [x] Hypothèses et périmètre écrits (ce qui est dans le mémoire, ce qui n'y est pas)

### Étape 1 — État de l'art
- [ ] 15 à 25 références lues et résumées (essaims, réseaux cellulaires, GNN, apprentissage multi-agents)
- [ ] Tableau comparatif des approches existantes
- [ ] Positionnement : ce que ton travail apporte de différent

### Étape 2 — Simulateur
- [x] Dynamique des robots (position, vitesse, limites)
- [x] Voisinage par portée de communication
- [x] Gestion des pannes (retrait de robots)
- [x] Animation de contrôle (visualisation)
- [x] Tests unitaires

### Étape 3 — Formes cibles
- [x] Formes géométriques (cercle, carré, triangle, etc.)
- [x] Alphabet A à Z
- [x] Chiffres 0 à 9
- [x] Nombres de 10 à 999 (assemblage de chiffres)
- [x] Conversion en nuages de points cibles
- [x] Archive zip des images et des points

### Étape 4 — Modèle
- [x] Échange de messages entre voisins
- [x] Sortie : mouvement de chaque robot
- [x] Références simples pour comparer (règles écrites à la main, version sans échange)
- [x] Tests : le modèle accepte un nombre variable de robots et de voisins

### Étape 5 — Entraînement
- [x] Fonction d'objectif (qualité de la forme, éviter les collisions, rester groupé)
- [x] Boucle d'entraînement avec positions de départ aléatoires
- [x] Apprentissage progressif (formes simples puis complexes)
- [x] Sauvegarde des modèles et des courbes
- [ ] Entrainement avec des formes complexes

### Étape 6 — Évaluation
- [x] Métriques calculées sur un jeu de test fixe
- [x] Test de pannes (5, 10, 20 % de robots retirés)
- [x] Test de passage à l'échelle (plus de robots qu'à l'entraînement)
- [x] Comparaison avec les références
- [x] Analyse des échecs

### Étape 7 — Rédaction
- [ ] Modèle LaTeX configuré (12 pt, interligne scientifique)
- [ ] Introduction et problématique
- [ ] État de l'art
- [ ] Méthode
- [ ] Expériences et résultats
- [ ] Discussion et conclusion
- [ ] Bibliographie, annexes, relecture

### Étape 8 — Soutenance
- [ ] Diapositives
- [ ] Démonstration vidéo ou animation
- [ ] Répétition et questions probables

## Jalons de contrôle

1. **Fin étape 2** : le simulateur fait bouger des robots avec une règle simple (rassemblement) et on voit le résultat.
2. **Fin étape 4** : un réseau non entraîné produit des mouvements sans erreur, quel que soit le nombre de robots.
3. **Fin étape 5** : au moins une forme simple (cercle) est atteinte de façon visible.
4. **Fin étape 6** : tous les tableaux et figures du mémoire sont prêts.
