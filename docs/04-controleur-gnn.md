# 04 — Contrôleur GNN

## Entrées d'un robot
Sa vitesse, l'identifiant de forme partagé (consigne de base), les messages de ses voisins à portée.
- Variante (campagne complexe) : ajout du vecteur vers le point de contour le plus proche pour une consigne plus riche.

## Passage de messages
- Chaque robot encode son observation en un état latent initial.
- À chaque tour, un **message** est calculé pour chaque voisin à partir des deux états latents et de la position relative.
- Les messages reçus sont **moyennés** (invariant à l'ordre des voisins, échelle stable quand la densité change). Voisinage vide : agrégat nul, la commande reste valide.
- Une fonction de mise à jour combine l'état courant et l'agrégat.
- Après les tours, un **décodeur partagé** produit une commande de vitesse, bornée par une non-linéarité.

## Profondeur et portée
Avec L tours, un robot intègre au plus l'information à L sauts. Campagnes simples : 2 tours, 32 composantes cachées. Campagne complexe : 3 tours, 64 composantes.

## Propriétés garanties (et testées)
Paramètres partagés (pas de dépendance à l'identité), invariance par permutation des robots, invariance par translation globale, tailles d'essaim et voisinages variables. Ces propriétés ne garantissent **pas** la qualité de formation ni le passage à l'échelle : ils se mesurent.

## Limite assumée de la moyenne
Un voisin défaillant pèse autant que les autres : aucune correction. À comparer à une agrégation tronquée ou à l'attention avant toute conclusion de robustesse au bruit.

## Implémentation
PyTorch, messages écrits directement (pas de bibliothèque de GNN) pour garder l'agrégation visible dans le code.
