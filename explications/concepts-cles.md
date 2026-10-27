# Concepts clés du projet, expliqués simplement

Ce document répond aux questions qu'un lecteur (ou un jury) se pose en découvrant le projet :
qu'est-ce qu'une graine, une ablation, à quoi servent les points de référence d'une figure,
pourquoi trois contrôleurs, et surtout : **qu'est-ce qui prouve que le modèle donne le bon résultat ?**

---

## 1. Les graines (« seeds »)

### Ce que c'est
Un ordinateur ne tire pas de vrai hasard : il utilise un **générateur pseudo-aléatoire**, une formule qui
produit une suite de nombres d'apparence aléatoire à partir d'un nombre de départ. Ce nombre de départ
est la **graine**. Même graine → exactement la même suite de « hasard ».

### Où le hasard intervient dans le projet
| Aléa | Ce qu'il décide | Graine |
|---|---|---|
| Positions initiales des robots | où chaque robot apparaît dans le disque de départ | graine d'épisode |
| Pannes | quels robots tombent en panne à mi-épisode | graine d'épisode |
| Initialisation du réseau | les poids de départ du GNN avant entraînement | graine d'entraînement |
| Tirage des épisodes d'entraînement | les situations vues pendant l'apprentissage | graine d'entraînement |
| Échantillonnage des formes | quels pixels deviennent les points cibles | `sampler_seed` (config/shapes.yaml) |

### Pourquoi c'est indispensable
1. **Reproductibilité** (règle 5 d'AGENT.md) : même configuration + même graine = même épisode, au bit près.
   N'importe qui peut refaire une expérience et retrouver les mêmes chiffres.
2. **Comparaison équitable** : avec la même graine d'épisode, la règle manuelle, le GNN et le GNN sans
   communication partent **exactement des mêmes positions** et subissent **les mêmes pannes**. La différence
   de résultat vient donc du contrôleur, pas de la chance (« comparaison appariée »).
3. **Mesurer la dispersion** : un seul entraînement peut être chanceux ou malchanceux. On entraîne donc
   chaque variante avec **5 graines d'entraînement** (11 à 15) et on évalue chaque modèle sur **50 graines
   d'épisode**. On rapporte la moyenne **et** l'écart entre graines.

### Deux niveaux à ne pas confondre
- **Graines d'entraînement** (5) : 5 modèles différents de la même variante. Leur écart dit si
  l'apprentissage est stable.
- **Graines d'épisode** (50 par condition) : 50 situations de départ pour un même modèle. Leur écart dit si
  le contrôleur est fiable d'une situation à l'autre.

Les intervalles de confiance entre modèles se calculent sur les 5 graines d'entraînement (loi de Student,
facteur 2,776), pas sur les 250 épisodes : sinon on ferait croire à une précision qu'on n'a pas
(« pseudo-réplication »).

Dans la démo, le champ **Graine** choisit la graine d'épisode : changer la graine change les positions de
départ ; la remettre redonne exactement le même film.

---

## 2. L'ablation

### Ce que c'est
Le mot vient de la médecine (retirer un tissu pour voir à quoi il servait). En apprentissage automatique,
une **ablation** consiste à **retirer un seul composant** du modèle, à tout garder identique par ailleurs
(données, budget d'entraînement, graines), et à mesurer ce qui change.

> Si on enlève X et que le résultat ne bouge pas, alors X n'apportait rien.

### Les ablations du projet
| Ablation | Composant retiré | Question à laquelle elle répond |
|---|---|---|
| `no_comm` contre `gnn_2_rounds` | les messages entre robots (graphe vidé) | la **communication** sert-elle à quelque chose ? (question centrale du mémoire) |
| 1, 2, 3 tours de messages | des tours de messages | faut-il propager l'information plus loin dans l'essaim ? |
| `use_target_geometry: false` | le vecteur vers le contour | le réseau sait-il former la forme avec le seul identifiant ? |
| `geometry_prior: 0` (à ajouter) | le « coup de pouce » du décodeur vers le contour | les couches **apprennent-elles** vraiment, ou le prior fait-il tout le travail ? |

### Pourquoi `geometry_prior` est l'ablation la plus urgente
Le décodeur ajoute `geometry_prior × (vecteur vers le contour)` avant de produire la commande. Avec
`geometry_prior: 1`, un réseau **non entraîné** pousse déjà chaque robot vers le point de contour le plus
proche : il se comporte presque comme la règle manuelle. Les pertes des runs actuels le confirment : la perte
de forme reste à 0,042 de l'itération 1 à 1200, et le meilleur checkpoint est souvent celui de l'itération 50.
Sans l'ablation `geometry_prior: 0`, on ne peut pas affirmer que le GNN a appris quoi que ce soit.

---

## 3. On demande aux robots de dessiner une figure, pas un nombre de points

### Ce que voit l'utilisateur
Dans la démo, on choisit **une figure** (Cercle, Lettre A, Chiffre 8, Nombre 314…), le contrôleur, le
nombre de robots et la graine. Les robots dessinent la figure. Il n'y a **pas** de choix « A — 20 points »
ou « A — 50 points » : le nombre de points n'est pas une notion pour l'utilisateur.

### Pourquoi il y a quand même des points, en interne
Pour dire si l'essaim a bien dessiné la figure, il faut **mesurer** l'écart entre l'essaim et la figure.
L'ordinateur ne compare pas une image : il place des **points de référence** régulièrement sur le tracé
de la figure (rendue en 128 × 128 pixels, puis échantillonnée par « point le plus éloigné »), et calcule
la distance entre les robots et ces points (distance de Chamfer). Ces points sont une **règle graduée**
posée sur la figure, pas une consigne « un robot par point ».

### Pourquoi leur nombre est fixé, et pas choisi
- **Comparabilité** : toutes les campagnes d'évaluation utilisent **50 points de référence**, pour 20, 40
  ou 100 robots. Si la résolution changeait d'un essai à l'autre, les distances ne seraient plus
  comparables. La démo utilise donc la même valeur, fixée dans `config/web-demo.yaml`
  (`target_resolution: 50`) : visible et versionnée (règle 4), mais pas un bouton de l'interface.
- **Assez de points** pour que la figure reste reconnaissable : à 20 points, un « 8 » devient un nuage.
  C'était le défaut du premier prototype (100 points demandés, seulement 20 distincts).
- **Conséquence à connaître** : la distance mesure aussi la **couverture** (chaque point de référence doit
  avoir un robot proche). Avec 20 robots pour 50 points, la couverture parfaite est impossible ; il y a un
  plancher de distance, et les scores à 20 robots sont moins bons pour tous les contrôleurs.

Le catalogue `outputs/shapes/` contient encore plusieurs résolutions (20, 40, 50) pour les études internes
(entraînement du cercle avec 40 points, par exemple) ; l'interface n'en expose qu'une.

---

## 4. Pourquoi trois contrôleurs : Manuelle, Regroupement, GNN ?

Un résultat chiffré n'a de sens que **comparé à quelque chose**. « Le GNN obtient 0,135 » ne dit rien ;
« le GNN obtient 0,135 là où une règle écrite à la main obtient 0,065 » dit beaucoup.

| Contrôleur | Ce qu'il fait | Rôle dans la comparaison |
|---|---|---|
| **Regroupement** (flocking) | cohésion vers les voisins + répulsion, **sans regarder la cible** | **plancher** : ce qu'on obtient sans aucune consigne de forme. Tout contrôleur sérieux doit faire mieux. |
| **Règle manuelle** | attraction vers le point de contour le plus proche + répulsion entre voisins, gains réglés (27 essais) | **référence honnête** : ce qu'un humain obtient avec une règle simple. Le GNN doit faire **mieux** pour justifier l'apprentissage. |
| **GNN** | réseau de neurones partagé, messages entre voisins à portée, poids appris | le **sujet** du mémoire. |

Et dans les campagnes, s'ajoute **`no_comm`** : le même GNN, sans messages. C'est l'ablation de la
communication (section 2).

### Règles d'équité (AGENT.md, règles 1 et 7)
- Tous les contrôleurs utilisent **la même interface** : mêmes entrées locales (voisins à portée `r_c`,
  consigne partagée), mêmes graines d'épisode, même simulateur, même bornage de vitesse.
- Aucune information globale (position d'un robot hors de portée, affectation robot→point).
- Limite connue : la règle manuelle reçoit le **déplacement complet** vers le contour (direction et
  distance), le GNN seulement la **direction**. Elle a donc un léger avantage d'information, à mentionner.

---

## 5. Qu'est-ce qui nous dit que le modèle donne le bon résultat ?

C'est la question la plus importante. La réponse courte : **une animation ne prouve rien** (règle 6).
La preuve se construit en couches, de la plus simple à la plus exigeante.

### Couche 1 — Le code fait ce qu'il prétend (tests automatiques)
383 tests vérifient des **propriétés**, pas des valeurs au hasard :
- les robots ne communiquent qu'à portée `r_c`, la vitesse ne dépasse jamais `v_max` ;
- mélanger l'ordre des robots ne change pas leurs commandes (invariance par permutation) ;
- décaler tout l'essaim ne change pas les commandes (invariance par translation) ;
- même graine → même épisode, au bit près ;
- les métriques donnent la valeur exacte sur des cas calculés à la main (deux ensembles identiques → 0).

Les tests prouvent que le **banc d'essai** est juste. Ils ne prouvent pas que le **modèle** est bon.

### Couche 2 — Une définition du « bon résultat » fixée à l'avance
- **Distance de Chamfer** : moyenne des distances robot → point cible le plus proche, plus point cible →
  robot le plus proche, divisée par le rayon de la cible. 0 = forme parfaite.
- **Succès** : la distance passe sous **0,15** (15 % du rayon de la forme) **et y reste** jusqu'à la fin.
- Ce seuil est écrit dans `docs/decisions/06-evaluation.md` **avant** de voir les résultats, et ne doit
  jamais être changé après coup.

### Couche 3 — Le modèle est jugé sur des situations qu'il n'a jamais vues
- **Graines d'épisode** différentes de celles de l'entraînement.
- **Découpage des formes** : entraînement / validation / test. Les formes de test (géométrie : étoile ;
  chiffres : 8, 9 ; lettres : L, Y, Z ; nombres : 9, 314, 606, 789, 999) ne servent **jamais** à entraîner
  ni à choisir un modèle. La fuite
  `numbers_9` / `digits_9` a d'ailleurs été corrigée le 2026-10-27.
- **Généralisation à la taille** : entraîné à 40 robots, évalué à 20 et 100 sans réentraînement.

### Couche 4 — Le modèle est comparé aux références, avec incertitude
Le GNN n'est « bon » que s'il bat la règle manuelle et le GNN sans communication **sur les mêmes
épisodes**, avec un écart **plus grand que la dispersion entre graines**.

Résultats de l'évaluation pilote actuelle (40 robots, sans panne, 50 épisodes) :

| Contrôleur | Cercle : distance / succès | Étoile : distance / succès |
|---|---|---|
| Regroupement | 0,413 / 0 % | 0,308 / 0 % |
| **Règle manuelle** | **0,065 / 100 %** | **0,141 / 74 %** |
| GNN 3 tours (1 graine) | 0,135 / 86 % | 0,172 / 12 % |
| GNN sans communication (5 graines) | 0,171 / 58 % | 0,174 / 10 % |

Lecture honnête : le GNN bat le plancher et sa version sans communication, mais **pas** la règle manuelle.
Avec une seule graine pour le GNN 3 tours, aucune conclusion statistique n'est possible. **Aujourd'hui, rien
ne prouve encore que le modèle donne un meilleur résultat qu'une règle simple** ; c'est un résultat
acceptable à présenter tel quel, à condition de le dire.

### Couche 5 — Un humain reconnaît-il la forme ? (lisibilité)
Une faible distance de Chamfer ne garantit pas qu'on reconnaisse un « A » (AGENT.md §6). Le protocole de
la phase 11 montre des images de l'essaim, sans la cible, à des participants qui disent quelle forme ils
voient. On en tire un taux de reconnaissance avec son intervalle de confiance, et une **matrice de
confusion** (quelles formes sont confondues : O/0, I/1, 8/B…). C'est le seul endroit où une matrice de
confusion a du sens, car c'est une vraie tâche de classification.

### En résumé : la chaîne de preuve
1. Tests verts → le banc d'essai est correct.
2. Seuil fixé à l'avance → « bon » a une définition objective.
3. Formes et graines jamais vues → pas de triche par mémorisation.
4. Comparaison appariée aux références, 5 graines, intervalles de confiance → l'apport est réel et pas dû
   au hasard.
5. Ablations (communication, `geometry_prior`) → on sait **quel composant** produit le gain.
6. Lisibilité humaine → la forme est reconnaissable, pas seulement proche en distance.

Le projet a aujourd'hui les étapes 1 à 3 en place ; les étapes 4 à 6 attendent les nouveaux entraînements
et la collecte des réponses humaines.
