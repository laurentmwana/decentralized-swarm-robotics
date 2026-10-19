# Explications détaillées du projet Swarm-GNN

Ce document fournit une explication complète des concepts, métriques et mécanismes utilisés dans le projet Swarm-GNN (Swarm Graph Neural Network).

## Qu'est-ce qu'une métrique dans ce contexte ?

Dans ce projet, une **métrique** est une mesure quantitative permettant d'évaluer la performance du système d'essaim de robots. Contrairement à une perte (loss) qui guide l'entraînement, les métriques sont utilisées pour l'évaluation et l'analyse des résultats après l'entraînement.

Les métriques principales utilisées sont :

### 1. Distance de forme (Shape Distance)
- **Définition** : Distance de Chamfer symétrique au carré entre les positions des robots et les points de la cible en fin d'épisode.
- **Calcul** : 
  - Pour chaque robot, trouver la distance au carré au point cible le plus proche et moyenne sur tous les robots
  - Pour chaque point cible, trouver la distance au carré au robot le plus proche et moyenne sur tous les points cible
  - Additionner les deux moyennes
- **Formule mathématique** : 
  ```
  Chamfer²(A, B) = (1/|A|) * Σ_{a∈A} [min_{b∈B} ||a-b||²] + (1/|B|) * Σ_{b∈B} [min_{a∈A} ||a-b||²]
  ```
- **Objectif** : Mesurer dans quelle mesure l'essaim parvient à reproduire la forme cible
- **Valeur idéale** : 0 (parfaite correspondance)
- **Note sur l'implémentation** : Le code utilise la distance au carré pour des raisons de stabilité numérique lors de l'entraînement
- **Seuil de succès (τ)** : Généralement fixé à 0,5 dans le protocole pilote (c'est une distance non au carré, donc la valeur apprise sera différente)

### 2. Taux de collisions (Collision Rate)
- **Définition** : Proportion de paires de robots se trouvant à une distance inférieure à la distance de sécurité, moyennée sur la durée de l'épisode.
- **Calcul** : 
  - À chaque pas de simulation, compter le nombre de paires de robots dont la distance est < distance de sécurité
  - Diviser par le nombre total de paires possibles (n*(n-1)/2 pour n robots)
  - Moyenne sur tous les pas de l'épisode
- **Objectif** : Éviter que les robots ne se rapprochent trop (risque de collision physique)
- **Valeur idéale** : 0 (aucune collision)
- **Note importante** : Ce n'est pas une collision mécanique dans la simulation, mais un seuil de sécurité défini arbitrairement

### 3. Régularité de répartition (Spread Regularity)
- **Définition** : Mesure de l'uniformité de la répartition des robots dans l'essaim comparée à la répartition idéale dans la cible.
- **Calcul** : 
  - Calculer la distance de chaque robot à son plus proche voisin
  - Calculer l'écart-type de ces distances pour l'essaim
  - Calculer l'écart-type des distances entre points cibles pour la forme idéale
  - Prendre la différence absolue entre ces deux écarts-types
- **Objectif** : Détecter les agrégations (amas) ou les vides dans la répartition des robots
- **Valeur idéale** : 0 (répartition parfaitement uniforme comme dans la cible)
- **Interprétation** : Une valeur élevée indique soit des amas (certains robots trop proches), soit des vides (certaines zones de la cible non couvertes)

### 4. Succès (Success Rate)
- **Définition** : Pourcentage d'épisodes où la distance de forme finale est inférieure au seuil τ.
- **Calcul** : 
  - Exécuter N épisodes avec différentes graines aléatoires
  - Compter le nombre d'épisodes où `distance_de_forme < τ`
  - Diviser par N et multiplier par 100
- **Objectif** : Mesurer la fiabilité du contrôleur à atteindre la forme cible
- **Valeur idéale** : 100% (tous les épisodes réussis)
- **Dépend de τ** : Plus τ est élevé, plus le taux de succès sera élevé (mais moins exigeant)

## Fonction de perte utilisée lors de l'entraînement

Contrairement aux métriques d'évaluation, la fonction de perte guide l'optimisation pendant l'entraînement. Elle combine quatre termes :

### 1. Terme de forme (Shape Loss)
- **Objectif** : Attirer l'essaim vers la cible
- **Basé sur** : Distance de Chamfer symétrique
- **Poids typique** : 1.0
- **Effet si négligé** : L'essaim n'a aucune raison de bouger vers la cible

### 2. Terme de collisions (Collision Loss)
- **Objectif** : Éviter les proximités dangereuses entre robots
- **Basé sur** : Pénalisation quadratique pour les distances < distance de sécurité
- **Calcul détaillé** :
  - Calculer toutes les distances inter-robot
  - Masquer les distances d'un robot à lui-même (diagonale)
  - Pour chaque paire, calculer `relu(safety_distance - distance)^2` 
    (carré de la différence si la distance < safety_distance, 0 sinon)
  - Moyenne sur toutes les paires de robots distincts
- **Poids typique** : 1.0
- **Effet si négligé** : Compression des robots sur le contour de la cible

### 3. Terme de mouvement (Motion Loss)
- **Objectif** : Limiter les commandes excessives et les oscillations
- **Basé sur** : Carré de la norme des commandes (vitesse demandée)
- **Poids typique** : 0.01
- **Effet si négligé** : Mouvements brusques, oscillations, consommation d'énergie excessive

### 4. Terme de couverture (Coverage Loss)
- **Objectif** : Assurer une répartition uniforme des robots correspondant à la cible idéale
- **Basé sur** : Comparaison des statistiques de voisinage (moyenne et écart-type des distances au plus proche voisin)
- **Calcul détaillé** :
  - Pour chaque robot, trouver la distance à son plus proche voisin (excluant lui-même)
  - Calculer la moyenne et l'écart-type de ces distances pour l'essaim actuel
  - Faire de même pour l'ensemble cible idéal
  - Calculer l'erreur quadratique sur la moyenne + l'erreur quadratique sur l'écart-type
  - Moyenne sur tous les robots
- **Poids typique** : 0.20
- **Effet si négligé** : Certaines parties de la cible restent vides tandis que d'autres sont surchargées (manque de régularité dans la répartition)

## Architecture du système

### Flux général
```
Forme cible → Nuage de points → Contrôleur local partagé → Simulateur 2D → Métriques
```

Pendant l'entraînement :
- La perte globale est calculée sur tout l'épisode
- Cette perte est rétropropagée à travers tous les pas de simulation
- Les paramètres du contrôleur GNN sont mis à jour

À l'exécution :
- Chaque robot applique la même politique basée sur son observation locale
- Il considère ses voisins (via le graphe de communication) et la consigne de cible
- Aucun état global n'est partagé entre les robots

### Composants principaux

1. **Simulation** : Gère la dynamique 2D, le voisinage, les limites de l'arène, et les pannes de robots
2. **Formes cibles** : Génère les nuages de points à partir d'images ou de définitions géométriques
3. **Contrôleurs** : Implémente le GNN et des contrôleurs de référence pour comparaison
4. **Entraînement** : Boucle d'optimisation avec curriculum learning
5. **Évaluation** : Exécute des scénarios standardisés et calcule les métriques
6. **Démonstration Web** : API locale pour visualiser les trajectoires en temps réel

## Graine et reproductibilité

La **graine** (seed) contrôle tous les éléments aléatoires du système :
- Positions initiales des robots
- Ordre de présentation des cibles pendant l'entraînement
- Scénarios de retrait de robots (pannes)

En utilisant la même graine, on obtient exactement le même épisode, ce qui permet :
- La reproductibilité des expériences
- La comparaison juste entre différents contrôleurs
- Le débogage facilité

## Curriculum Learning

Le curriculum consiste à entraîner progressivement du simple au complexe :
1. Commencer avec des formes géométriques simples (carré, triangle, cercle)
2. Ajouter les chiffres (0-9)
3. Ajouter les lettres (A-Z)
4. Terminer avec les nombres à plusieurs chiffres

Chaque nouveau niveau **conserve les cibles précédentes** dans le lot d'entraînement pour réduire l'oubli catastrophique.

## Seuil de succès (τ)

Le seuil τ définit ce qu'on considère comme un "succès" :
- Un épisode est réussi si `distance_de_forme < τ`
- τ doit être choisi avec soin : trop faible = aucun succès jamais, trop élevé = tout est considéré comme succès
- Dans le protocole pilote : τ = 0,5
- **Toujours rapporter τ avec le pourcentage de succès** car le succès est très sensible à cette valeur

## Limitations et hypothèses

Le modèle fait plusieurs simplifications assumées :
- Référentiel commun partagé par tous les robots
- Communications instantanées et sans perte entre voisins
- Pas d'obstacles dans l'environnement
- Pas de physique de contact (les "collisions" sont juste des seuils de sécurité)
- Masse négligeable et pas d'inertie

Ces limitations sont connues et doivent être progressivement levées dans les futures versions.

## Comparabilité entre tailles

Pour permettre la comparaison entre essaims de tailles différentes :
- La distance de Chamfer est normalisée par la racine carrée du nombre de points cible
- Cela permet d'avoir un seuil τ ayant le même sens indépendamment du nombre de robots
- Sans cette normalisation, plus il y a de robots, plus la distance tend naturellement vers zéro

## Évaluation vs entraînement

Il est crucial de distinguer :
- **Pendant l'entraînement** : On optimise une fonction de perte qui combine plusieurs termes
- **Pendant l'évaluation** : On mesure des métriques séparément pour comprendre le comportement

Une bonne performance à l'entraînement (faible perte) ne garantit pas forcément une bonne performance selon toutes les métriques d'évaluation, car la perte est une combinaison pondérée tandis que les métriques sont examinées individuellement.
