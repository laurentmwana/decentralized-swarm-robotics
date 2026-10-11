# Architecture initiale du contrôleur GNN

```text
vitesse du robot + identifiant partagé de la forme
                    │
                    ▼
            encodeur partagé ─── état latent hᵢ
                    │
       ┌────────────┴────────────┐
       │ pour chaque voisin j     │
       ▼                          │
message MLP(hᵢ, hⱼ, positionⱼ − positionᵢ, distanceᵢⱼ)
       │                          │
       └── moyenne des messages voisins ──► mise à jour MLP
                                                │
                                      répéter 1 à 3 fois
                                                │
                                                ▼
                                      décodeur partagé
                                                │
                                                ▼
                                   commande de vitesse 2D bornée
```

Chaque robot applique les mêmes paramètres. L'agrégation par moyenne est insensible à l'ordre des voisins et accepte un nombre variable de voisins. Les positions absolues ne sont pas une entrée directe : seules les différences de position le long des arêtes sont utilisées, ce qui rend la politique invariante par translation globale.

## Variantes de référence

1. Règle locale cohésion-répulsion de l'étape 2.
2. `NoCommunicationController` : même entrée vitesse + forme, sans messages entre robots.
3. `GNNController` avec une seule couche de messages, à comparer au modèle de deux couches.
