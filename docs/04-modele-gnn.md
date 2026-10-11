# Étape 4 — Modèle : réseau de neurones sur graphes

## Objectif
Construire le « cerveau » partagé par tous les robots : il transforme ce qu'un robot perçoit autour de lui en un mouvement.

## Principe en mots
À chaque pas de temps :
1. Chaque robot prépare une **description de lui-même** (sa position, sa vitesse, la consigne de forme).
2. Il **reçoit des messages de ses voisins** : leurs positions relatives et leurs propres résumés.
3. Il **combine** ces messages en un seul résumé, d'une manière qui ne dépend ni du nombre ni de l'ordre des voisins.
4. Il répète les étapes 2 et 3 deux ou trois fois : l'information se propage ainsi un peu plus loin que ses voisins directs.
5. Une dernière couche transforme le résumé en **commande de mouvement** (direction et vitesse, limitée à la vitesse maximale).

## Entrées d'un robot
- position relative de chaque voisin (écart horizontal et vertical, distance)
- sa propre vitesse
- l'information sur la forme (selon le choix de l'étape 3)
- éventuellement : l'écart entre sa position et le point de la forme qui lui correspond, si cette association existe

## Sortie
Un vecteur de mouvement par robot.

## Propriétés indispensables
- **Mêmes paramètres pour tous les robots.**
- **Nombre de voisins variable** : le résumé des voisins doit être calculé par une opération symétrique (somme, moyenne, maximum), donc indifférente à l'ordre.
- **Nombre de robots variable** : le modèle ne doit pas contenir de taille fixe.
- **Invariance utile** : si on déplace tout le groupe, le comportement relatif ne change pas ; utiliser des positions relatives plutôt qu'absolues autant que possible.

## Choix d'architecture à tester
- **Version de base** : une couche d'échange de messages simple, résumé par moyenne.
- **Version avec attention** : le robot pondère l'importance de chaque voisin.
- **Profondeur** : 1, 2 ou 3 tours d'échange.
- **Taille des représentations** : petite (32), moyenne (64), grande (128).
- **Mémoire** (optionnel) : ajouter une mémoire courte pour que le robot tienne compte du passé.

Partir de la version de base ; compliquer seulement si les résultats l'exigent.

## Lien avec les réseaux cellulaires
Dans un réseau cellulaire, chaque cellule est identique et n'interagit qu'avec son voisinage fixe. Ici, le principe est le même, mais le voisinage change avec les déplacements et les messages sont appris. À expliquer dans le mémoire comme justification de l'intitulé d'origine.

## Références à implémenter (pour comparer)
1. **Règles écrites à la main** (étape 2) adaptées à la formation de forme quand c'est possible.
2. **Réseau sans échange** : chaque robot décide seul à partir de sa seule position, sans voisins. Montre l'apport de la communication.
3. **Version à une seule tournée d'échange**, pour voir l'effet de la profondeur.

## Tests à écrire
- Le modèle accepte 5, 10 puis 50 robots sans modification.
- Changer l'ordre des robots ne change pas leurs commandes (au renommage près).
- La sortie respecte la vitesse maximale.
- Un robot isolé (sans voisin) produit quand même une commande valide.

## À faire
- [ ] Module d'échange de messages
- [ ] Modèle complet (entrée vers sortie)
- [ ] Trois références
- [ ] Tests de propriétés ci-dessus
- [ ] Schéma d'architecture pour le mémoire

## Critères de réussite
Le modèle non entraîné s'exécute dans le simulateur pour n'importe quel nombre de robots et produit des mouvements valides.

## Pièges
- Utiliser des positions absolues : le modèle apprend par cœur au lieu de généraliser.
- Oublier le cas du robot isolé.
- Modèle trop gros pour la quantité de données.
