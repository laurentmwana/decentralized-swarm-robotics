# 02 — Règles du simulateur

Daté : 2026-10-21. Figel avant le code (plan §0.5, règle §0.2-6).

## Arène et dynamique
- Arène **carrée**, côté `arena_size` (config). Repère commun 2D, origine au centre ; un robot est dans l'arène si `|x| <= arena_size/2` et `|y| <= arena_size/2`.
- État par robot : position 2D, vitesse 2D. **La vitesse est directement commandée** (double intégrateur retiré : modèle à simple intégrateur, `x_{t+1} = clip(x_t + v_t * dt)`).
- La commande du contrôleur est **bornée par la vitesse maximale avant la mise à jour de position** : `v = command * min(1, v_max / ||command||)`, puis position, puis clamp aux limites d'arène.
- `dt` et `v_max` vivent dans `config/simulator.yaml`. Aucun nombre de robots n'est codé en dur : la taille d'essaim vient de la configuration d'épisode.

## Graphe de communication
- Arête `(i, j)` si `||x_i - x_j|| <= r_c` (portée de communication, config).
- **Recalculé à chaque pas**, pas d'auto-boucle, pas d'arête vers un robot retiré (voir règle des retirés).
- Un robot sans voisin reste valide : son agrégat de messages est nul (convention reprise en phase 4).

## Règle des robots retirés (pannes) — DÉCISION
Quand un robot tombe en panne à mi-épisode (`failure_fraction`, instancié par le `FailureModel` avec la graine de l'épisode) :

- il est **supprimé du graphe de communication** (ni émetteur ni récepteur d'arête) ;
- il est **exclu des collisions et de toutes les métriques** ;
- il **ne reçoit plus de commande et ne bouge plus** (position figée à son dernier pas actif) ;
- il **reste présent dans l'état** (les tableaux gardent N lignes ; le masque `active[t, i]` dit s'il compte au pas t).

Conséquences : les contrôleurs doivent masquer les robots retirés dans leur propre agrégation ; la fraction de robots actifs en fin d'épisode est rapportée avec chaque score.

Alternative écartée : le robot en panne continue de broadcast des positions aberrantes — c'est une perturbation, traitée en **phase 12** (`FaultyRobot`), pas le mode retrait par défaut.

## Source d'aléa unique par épisode
Toute l'aléa d'un épisode passe par `SeedManager(seed_config, seed_episode)` : positions initiales, tirage des pannes. Rien d'autre n'appelle le hasardeux. Deux exécutions avec la même configuration et les mêmes graines donnent **les mêmes positions, bit à bit**.

## Interface d'épisode (contrat unique pour tous les contrôleurs)
`simulate_episode(target, initial_positions, seed, controller, config) -> EpisodeResult`
- `target` : nuage de points normalisé (N_points, 2), produit par `src/shapes` ;
- `controller` : implémentation de `Controller` (vue locale uniquement : son état, ses voisins à portée, la consigne partagée — AGENT.md règle 1) ;
- `EpisodeResult` : `positions (T+1, N, 2)`, `active (T+1, N)`, `commandes (T, N, 2)`, graines et config effective. Positions **enregistrées à chaque pas** ; les figures et les métriques ne relisent que ces fichiers.
