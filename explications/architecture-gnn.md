# Architecture du réseau de neurones (SwarmGNN)

Source : `src/models/` (encoder, message, aggregator, decoder, gnn) et `config/model.yaml`.
Valeurs vérifiées le 2026-10-27 en instanciant `SwarmGNN` avec la configuration actuelle.

## En bref

Le nombre de couches dépend du nombre de tours de messages R (`message_rounds`) :

- encodeur : 1 embedding + 1 couche linéaire ;
- chaque tour de messages : 4 couches linéaires ;
- décodeur : 2 couches linéaires.

Soit **3 + 4R couches linéaires** (plus l'embedding). Avec la configuration par défaut
(`message_rounds: 2`, `hidden_dim: 64`) : **11 couches linéaires + 1 embedding**.

## Détail des blocs (2 tours)

| Bloc | Couches | Dimensions |
|---|---|---|
| Encodeur | Embedding + Linear + ReLU | identifiant de forme → 16 ; [vitesse 2 + vecteur vers le contour 2 + embedding 16] = 20 → 64 |
| Tour de messages 1 | MLP d'arête (2 Linear) + MLP de mise à jour (2 Linear) | arête : 130 → 64 → 64 ; mise à jour : 128 → 64 → 64 |
| Tour de messages 2 | idem, poids propres | idem |
| Décodeur | 2 Linear | 64 → 64 → 2 (vx, vy), puis `v_max · tanh` et bornage de la norme à `v_max` |

- Entrée du MLP d'arête : `[h_i, h_j, p_j − p_i]` = 64 + 64 + 2 = 130 (position relative : invariance par translation).
- Agrégation : **moyenne** des messages des voisins à portée `r_c`, sans poids appris ; voisinage vide → agrégat nul.
- Entrée du MLP de mise à jour : `[h_i, agrégat]` = 128 ; sortie ajoutée par **connexion résiduelle** (`h ← h + update`).
- Activations : ReLU.
- Le décodeur ajoute `geometry_prior × vecteur vers le contour` avant le `tanh` : dès l'initialisation, la commande pointe vers la cible.

```
identifiant de forme ─► Embedding(64, 16) ─┐
vitesse (2) ───────────────────────────────┼─► Linear(20 → 64) + ReLU ─► h
vecteur vers le contour (2) ───────────────┘
                                               │
              ┌──── répété R fois (poids propres à chaque tour) ────┐
              │  messages m_ij = MLP_arête([h_i, h_j, p_j − p_i])   │
              │  agrégat_i    = moyenne des m_ij sur les voisins     │
              │  h_i          = h_i + MLP_maj([h_i, agrégat_i])      │
              └──────────────────────────────────────────────────────┘
                                               │
                                               ▼
              Linear(64 → 64) + ReLU ─► Linear(64 → 2) ─► v_max·tanh ─► bornage ─► commande (vx, vy)
```

## Par variante

| Variante | Tours | Couches linéaires | Paramètres (avec vecteur vers le contour) | Paramètres (identifiant seul) |
|---|---|---|---|---|
| `gnn_1_round` | 1 | 7 | 31 618 | 31 490 |
| `gnn_2_rounds` | 2 | 11 | 56 578 | 56 450 |
| `gnn_3_rounds` | 3 | 15 | 81 538 | 81 410 |
| `no_comm` | 2 | 11 | 56 578 | 56 450 |

- `no_comm` a exactement les mêmes couches que `gnn_2_rounds`, mais reçoit un graphe vide :
  l'agrégat est toujours nul, aucun message ne circule. Même taille de modèle → comparaison équitable.
- La variante « identifiant seul » (`use_target_geometry: false`) a 128 paramètres de moins :
  l'entrée de l'encodeur passe de 20 à 18.
- Le nombre de paramètres **ne dépend pas du nombre de robots** : tous les robots partagent les mêmes poids.

## À retenir pour le mémoire

- Les chiffres de `PLAN.md` (31 490 / 56 450 / 81 410) correspondent à la variante **identifiant seul**.
  La configuration actuelle utilise le vecteur vers le contour (`use_target_geometry: true`) :
  citer les chiffres de la variante réellement entraînée.
- Le prior géométrique du décodeur (`geometry_prior: 1.0`) fait qu'un modèle non entraîné se comporte
  déjà comme une règle « aller vers le point de contour le plus proche ». Une ablation
  `geometry_prior: 0` est nécessaire pour montrer ce qu'apportent réellement les couches apprises.
