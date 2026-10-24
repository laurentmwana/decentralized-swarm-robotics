# Sauvetage de l'entraînement v2 : diagnostic, correctifs et vérification

> Document de suivi — 22 octobre 2026.
> Expliqué simplement : chaque section répond à « quoi ?, pourquoi ?, comment ? ».

## 1. Contexte : que fait ce projet ?

On entraîne un **essaim de robots** à se mettre en forme (un cercle, une lettre, un
pentagone…). Chaque robot est petit et « bête » : il ne voit que deux choses :

- **sa vitesse** actuelle ;
- **la direction du point de la cible le plus proche** (le « vecteur géométrie »),
  et éventuellement ce que disent ses voisins (la « communication », via un réseau
  de neurones sur graphe, ou **GNN**).

Un réseau de neurones (le « contrôleur ») décide de la commande de chaque robot à
chaque pas de temps. On l'entraîne par **descente de gradient** : on déroule une
simulation complète (« rollout »), on mesure à quel point le résultat est mauvais
(une « fonction de perte », ou *loss*), et on ajuste les poids du réseau pour faire
baisser ce score. C'est possible ici parce que le simulateur est entièrement écrit
avec des tenseurs différentiables (PyTorch) : le gradient « remonte » du résultat
final jusqu'aux poids, pas à pas.

La perte est composée de 4 termes qu'on additionne avec des **poids** (des
coefficients qui disent ce qui compte le plus) :

| Terme | Ce qu'il mesure | Poids |
|---|---|---|
| `shape` | distance moyenne robot → point de cible le plus proche (être SUR la forme) | 1.0 |
| `coverage` | distance moyenne point de cible → robot le plus proche (pas de TROUS) | 1.0 |
| `collisions` | pénalité quand deux robots se chevauchent | 0.5 |
| `motion` | énergie cinétique moyenne (trajectoires calmes) | 0.02 |

## 2. Le problème : la campagne v1 a échoué

Les figures de la première campagne (`figures/training_circle_final.png`,
`figures/training_losses_*.png`) montraient trois symptômes :

1. **État final dispersé** : à la fin de l'entraînement, les robots ne formaient
   pas le cercle du tout — ils étaient éparpillés plus ou moins là où le tirage
   aléatoire initial les avait placés.
2. **Courbes de perte bruitées et plafonnées** : la perte `shape` descendait un
   peu (0.4 → 0.25) puis stagnait vers l'itération 600–800 sur 1200.
3. **`no_comm` aussi bon que les GNN** : couper la communication entre robots ne
   changeait rien, donc le GNN n'apprenait rien grâce aux voisins.

Et le pire indicateur : la perte `motion` convergeait vers ~0.01, c'est-à-dire des
robots qui avancent à **10 % de leur vitesse maximale**. Le modèle avait trouvé une
solution paresseuse : **presque ne pas bouger**. Tant qu'on ne bouge pas, on ne
risque rien, les pertes bougent peu… et le cercle ne se forme jamais.

## 3. Le diagnostic : 5 causes racines

En lisant le code (`src/training/`, `config/`), cinq problèmes ont été identifiés,
classés du plus meurtrier au plus subtil.

### 3.1 Le learning rate s'effondrait (cause n°1)

Le **learning rate** (lr) décide de la taille des correctifs appliqués aux poids à
chaque itération. On utilisait une décroissance exponentielle : `lr_gamma: 0.995`,
appliquée **à chaque itération**. Mathématiquement :

```
lr_final = lr_initial × 0.995^1200 ≈ lr_initial / 410
```

Autrement dit, dès l'itération 600 le modèle apprenait 20 fois moins vite qu'au
départ, et vers 1000 il était **paralysé**. C'est exactement là que les courbes
plateauaient. Image : une voiture qui lève le pied chaque tour de roue ; elle ne
verra jamais la ligne d'arrivée.

**Correctif** : `lr_gamma: 0.999` → `0.999^1200 ≈ 0.30`, soit un lr final encore
30 % du lr initial. Fichier : `config/training.yaml`.

### 3.2 Des gradients ultra-bruités

À chaque itération, le modèle ne regardait que **2 simulations** (`batch_episodes: 2`)
pour décider dans quel sens corriger ses poids. Avec si peu d'exemples, le « sens
à suivre » est dominé par le hasard du tirage initial des robots — d'où les courbes
en dents de scie.

**Correctif** : `batch_episodes: 8`. Plus d'exemples = moyenne plus stable =
meilleure direction. Coût : 4× plus lent par itération.

### 3.3 Le modèle démarrait « gelé » et restait au point mort

Deux effets se cumblaient :

- À l'initialisation, les sorties du décodeur (les commandes de vitesse) sont
  **proches de zéro** : les robots ne bougent presque pas au départ.
- Chaque commande ne déplace le robot que de `vitesse × dt` (dt = 0.1), donc
  l'influence d'un poids sur la perte finale est **multipliée par 0.1** à chaque
  pas. Le signal d'apprentissage est donc tout petit, et avec un lr qui s'effondre
  (3.1), le modèle n'a jamais eu la force de « sortir du ralenti ».

**Correctif (le plus intéressant)** : un **a priori structurel** dans le décodeur.
On ajoute au sortie du réseau un petit « résidu » : le vecteur géométrie lui-même.

```
commande = tanh(reseau(h) + 1.0 × vecteur_géométrie)
```

À l'initialisation, `réseau(h)` ≈ 0, donc la commande ≈ « foncer vers le point le
plus proche du contour ». Le modèle part d'une **stratégie naïve mais correcte**
(ce que fait le contrôleur trivial de la phase 1 !) et l'entraînement n'a plus qu'à
l'améliorer (éviter les collisions, répartir les robots) au lieu de devoir tout
inventer depuis l'immobilité. Fichiers : `src/models/decoder.py`, `src/models/gnn.py`,
et le coefficient est un paramètre de config : `geometry_prior: 1.0` dans `config/model.yaml`
(règle du projet : aucun poids ne vit dans le code).

### 3.4 La perte `coverage` avait un plancher inatteignable

La cible d'entraînement était un cercle échantillonné en **50 points**, mais il n'y
a que **40 robots**. Même parfaitement entraîné, au moins 10 points restent vides :
la perte `coverage` ne peut jamais descendre sous ~0.15. On gaspillait une partie du
gradient à lutter contre une limite mathématique, et la courbe plate de `coverage`
qu'on voyait était en partie **normale**, pas un échec.

**Correctif** : entraîner sur `geometries_circle@40` (40 points, 40 robots) — la
résolution 40 a été ajoutée à `config/shapes.yaml`.

### 3.5 Les collisions comptaient le tirage au sort initial

Les robots naissent au hasard dans un disque ; avec 40 robots, il y a toujours ~2
paires qui se chevauchent **avant toute décision du modèle**. La pénalité de
collision au pas 0 est donc du bruit sans coupable, dont le gradient pousse les
robots à s'éparpiller pile au début, quand ils ne savent encore rien faire.

**Correctif** : `collision_skip_steps: 10` — on ignore les 10 premiers pas dans la
perte de collision (`src/training/losses.py`).

## 4. La vérification : ça marche

### 4.1 Test rapide sans entraînement

Avec le seul a priori (3.3), avant toute optimisation :

- vitesse moyenne des robots à l'init : **0.81 × v_max** (avant : 0.1) ;
- alignement des commandes avec le contour : **0.99** ;
- distance au cercle sur un épisode : **0.38 → 0.04**.

### 4.2 Run complet de vérification (`gnn_2_rounds`, graine 11, 1200 itérations)

| Métrique | v1 (avant) | v2 (après) |
|---|---|---|
| Score de validation (Chamfer, plus petit = mieux) | 0.428 | **0.138** |
| Perte `shape` finale | 0.229 | **0.041** |
| Perte `coverage` finale | 0.149 | **0.087** |
| Perte `motion` | 0.009 (gelé) | 0.563 (les robots avancent) |
| Meilleure itération atteinte | 650 (plateau) | 1050 (progresse encore) |

Figure : `figures/training_circle_v2_verify.png` — les robots suivent l'anneau
cible, y compris à mi-épisode. La suite de tests (371 tests unitaires) est verte.

### 4.3 Un nouveau point faible, honnête : les collisions en transit

La perte `collisions` a explosé (0.45 → 7.58). Explication : maintenant que les
robots **roulent**, ils se marchent dessus en allant vers le cercle, et plusieurs
visent le même point (plusieurs robots sur un point, d'autres points libres).
C'est un vrai problème d'essaim, mais c'est un problème de **deuxième étage** : la
formation, elle, fonctionne. Pistes pour la suite : augmenter légèrement le poids
`collisions`, ou allonger l'horizon pour laisser le temps de se répartir.

## 5. Ce qui est en cours

- **Campagne complète v2** : 20 runs (4 variantes × 5 graines), 4 en parallèle sur
  16 cœurs, pilote `outputs/campaign_v2.sh`, sortie `outputs/training/`, logs dans
  `outputs/campaign_logs/`. Durée totale estimée : 5 à 6 h.
- **Version web** (`demo/`) : le serveur lit les checkpoints dans
  `outputs/training/` ; pendant la campagne, `config/web-demo.yaml` pointe
  **temporairement** vers le run de vérification (`outputs/training_v2`) pour que
  le contrôleur GNN reste disponible. À remettre sur `outputs/training` à la fin.

## 6. A faire ensuite

1. À la fin de la campagne : régénérer figures + `summary.csv` + `budget.md`
   (`python -m src.reports.training_summary`), comparer les variantes entre elles
   (le GNN avec communication doit-il enfin battre `no_comm` ?).
2. Remettre `web-demo.yaml` sur `outputs/training` et re-tester la démo.
3. Nettoyer : supprimer `outputs/training_v2` et la figure de vérification
   (devenus redondants avec la campagne).
4. Décider, chiffres en main, si le problème de collisions en transit mérite un
   ajustement (poids ou horizon) et un troisième cycle de réglage.
5. Mettre à jour le manuscrit LaTeX avec les nouveaux résultats.

## 7. Mini-glossaire

- **Perte (loss)** : score de mauvaise performance ; l'entraînement le diminue.
- **Gradient** : direction qui fait baisser la perte le plus vite.
- **Learning rate (lr)** : taille du pas qu'on suit cette direction.
- **Batch** : nombre de simulations moyennées pour estimer le gradient.
- **Roulement / rollout** : une simulation complète du départ des robots jusqu'à
  la fin de l'épisode, écrite en tenseurs différentiables.
- **Chamfer** : distance moyenne entre deux nuages de points ; notre critère de
  « à quel point la forme est réussie ».
- **Checkpoint** : copie des poids du réseau sauvegardée pendant l'entraînement ;
  on garde le meilleur selon la validation.
- **Graine (seed)** : numéro du générateur aléatoire ; même graine = expérience
  reproductible à l'identique.
