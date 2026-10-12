# Fiche technique — essaim de robots contrôlé par GNN

## Réponse courte pour la soutenance

**Oui, le projet utilise un réseau de neurones sur graphes (GNN) entraîné avec PyTorch.**
Chaque robot exécute la même politique apprise. Il échange seulement avec les robots situés dans son rayon de communication, puis le réseau produit son vecteur de mouvement en 2D. Il n'y a pas de robot chef qui calcule les déplacements de tout l'essaim.

Le projet est à ce stade une **simulation 2D** : il ne pilote pas encore des robots physiques ni une application Web temps réel.

## Problème traité

Faire déplacer un essaim de robots afin qu'il se répartisse sur une forme cible (cercle, formes géométriques, chiffres, lettres et nombres), tout en évitant les collisions. La question étudiée est la suivante :

> Un même réseau, fondé sur des informations locales et partagé entre tous les robots, peut-il apprendre un comportement collectif de formation et rester utilisable lorsqu'il y a des pannes ou un nombre différent de robots ?

## Principe décentralisé

À un instant donné, on construit un graphe :

- un nœud = un robot ;
- une arête entre deux nœuds = les deux robots sont à une distance inférieure ou égale au rayon de communication ;
- le rayon employé dans les expériences est de **2,5 unités** ;
- le même GNN est appliqué à tous les nœuds.

Un robot n'a donc pas besoin de connaître la liste complète des autres robots. Le nombre et l'ordre de ses voisins peuvent changer sans modifier l'architecture.

## Informations utilisées par le GNN

### Informations propres à chaque robot

Le code transmet au réseau :

1. sa vitesse actuelle `(vx, vy)` ;
2. un vecteur de consigne de forme en 2D ;
3. l'identifiant de la forme, converti en vecteur appris (*embedding*) ;
4. les messages agrégés de ses voisins.

La nature du vecteur de consigne dépend de la campagne :

- **entraînement cercle et curriculum réduit** : vecteur entre le robot et le centre de l'arène ; l'identifiant indique quelle forme est demandée ;
- **campagne complexe** : vecteur du robot vers le point le plus proche du contour cible, normalisé par le rayon de la cible. Cette consigne géométrique est consultée localement à la position du robot.

### Informations provenant d'un voisin

Pour chaque voisin, le message prend en compte :

- l'état latent du robot qui reçoit le message ;
- l'état latent du voisin qui l'envoie ;
- la position relative `(dx, dy)` ;
- la distance entre les deux robots.

Les messages de tous les voisins sont ensuite **moyennés**. Cette moyenne rend le calcul insensible à l'ordre des voisins et permet d'accepter un nombre de voisins variable.

## Architecture du réseau

Le contrôleur est défini dans `src/models/gnn_controller.py`.

```text
vitesse + vecteur de consigne + embedding de la cible
                    │
                    ▼
             encodeur MLP
                    │
                    ▼
  2 ou 3 tours : messages locaux → moyenne → mise à jour de l'état
                    │
                    ▼
             décodeur MLP
                    │
                    ▼
        commande (vx, vy), vitesse bornée
```

Un **MLP** est une succession de couches entièrement connectées ; `ReLU` est la fonction d'activation. Le décodeur applique `tanh`, puis la norme de la commande est limitée à la vitesse maximale : un robot ne peut donc pas dépasser la vitesse configurée.

### Nombre de couches et de paramètres

| Campagne | Dimension cachée | Tours d'échange GNN | Paramètres | Modèle sauvegardé |
|---|---:|---:|---:|---|
| Cercle | 32 | 2 | 16 130 | `results/training-circle/circle-best.pt` |
| Curriculum réduit final (11 formes) | 32 | 2 | 16 450 | `results/curriculum/numbers/model.pt` |
| Campagne complexe, niveau géométrie | 64 | 3 | 88 130 | `results/curriculum-complex/geometry/model.pt` |

Dans chaque tour d'échange, il y a deux petits MLP :

- MLP de message : deux couches linéaires, avec ReLU entre elles ;
- MLP de mise à jour : deux couches linéaires, avec ReLU après chacune.

L'encodeur et le décodeur possèdent eux aussi deux couches linéaires. Quand on dit « 2 couches GNN », on parle ici de **deux tours de communication**, pas seulement de deux couches linéaires.

## Entraînement effectué

L'entraînement est direct et différentiable : le GNN commande les robots dans un simulateur PyTorch pendant plusieurs pas, puis la perte obtenue sert à mettre à jour les paramètres avec l'optimiseur **Adam**.

À chaque itération :

1. des positions initiales aléatoires sont générées ;
2. le simulateur déroule l'épisode ;
3. le graphe de voisinage est reconstruit à chaque pas ;
4. une perte est calculée ;
5. la rétropropagation ajuste les poids du GNN.

### Fonctions de perte

La perte totale combine :

- **distance de Chamfer symétrique** : les robots doivent être proches de la forme et chaque zone de la forme doit être couverte ;
- **pénalité de collision** : pénalise les paires plus proches que la distance de sécurité ;
- **pénalité de mouvement** : évite des commandes inutilement fortes ;
- **perte de couverture** : compare la répartition locale des robots avec celle des points de la cible.

Les coefficients sont enregistrés dans chaque fichier de configuration, ce qui rend l'expérience reproductible.

### Campagnes disponibles

| Commande | But | État |
|---|---|---|
| `python -m src.train.run` | Premier apprentissage sur le cercle, 20 robots, 80 itérations | Exécuté |
| `python -m src.train.curriculum` | Apprentissage progressif : géométries, chiffres 0–2, lettres A–C, nombres 010 et 123 | Exécuté à petite échelle |
| `python -m src.train.curriculum --config configs/curriculum-complex.yaml --initial-model none --output results/curriculum-complex` | Version plus détaillée : 50 robots, 50 points de contour, 3 tours d'échange | Seul le niveau géométrie est actuellement présent dans `results/curriculum-complex/` |

Le curriculum réduit est une validation fonctionnelle : 30 itérations par niveau avec 20 robots. Il ne faut pas prétendre que les lettres et les nombres sont déjà parfaitement lisibles ; une campagne plus longue reste nécessaire.

## Évaluation disponible

L'évaluation pilote compare le GNN entraîné à une règle locale de regroupement, sur un cercle, pour :

- 20, 40 et 100 robots ;
- 0 %, 5 %, 10 % et 20 % de robots retirés ;
- 10 épisodes par condition.

Le GNN a une distance moyenne à la forme beaucoup plus faible que la règle de regroupement. Par exemple, avec 100 robots et 20 % de pannes :

| Méthode | Distance à la forme | Taux de réussite |
|---|---:|---:|
| GNN | 0,497 | 90 % |
| Règle de regroupement | 3,557 | 0 % |

Ces chiffres sont dans `results/evaluation-pilot/summary.csv`. Cette étude doit être présentée comme **pilote**, car elle emploie 10 épisodes par condition au lieu des 50 prévus par le protocole final. De plus, la règle de regroupement n'a pas accès à une consigne explicite de cercle : elle est un contrôle comportemental, pas une comparaison strictement équitable.

## Ce qu'il faut montrer pendant la présentation

1. Le GIF du simulateur : `results/simulator-gathering.gif` (illustration d'une règle simple, pas le GNN entraîné).
2. La courbe de perte : `results/training-circle/loss.png` ou les courbes du curriculum.
3. La planche du curriculum : `results/curriculum/numbers/contact-sheet.png`.
4. Les figures de robustesse : `results/evaluation-pilot/failures.png` et `results/evaluation-pilot/scale.png`.
5. Cette idée clé : le modèle est partagé, la communication est locale et l'agrégation moyenne permet de traiter un nombre variable de voisins.

## Limites à dire honnêtement

- Les expériences sont réalisées en simulation 2D, sans capteurs, bruit de mesure, obstacles ni robots physiques.
- Le curriculum réduit est court et les formes complexes demandent un entraînement plus long.
- L'évaluation est pilote : plus de graines et d'épisodes sont nécessaires pour conclure scientifiquement.
- La baseline sans communication, les tests de bruit et les essais avec portée réduite restent à exécuter.
- L'interface HTML de démonstration reçue séparément n'est pas reliée à ce modèle entraîné ; elle ne doit pas être présentée comme une exécution du GNN.

## Réponse prête à dire si on demande « pourquoi un GNN ? »

> Un GNN est adapté parce qu'un essaim est naturellement un graphe dynamique : les robots sont les nœuds et leurs communications sont les arêtes. Le même contrôleur est partagé par tous les robots, et il agrège les informations des voisins, quel que soit leur nombre. Cela correspond à un contrôle réellement distribué et permet de tester le passage à l'échelle.
