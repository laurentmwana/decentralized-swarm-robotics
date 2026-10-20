# AGENT.md — Consignes pour l'agent de développement

Projet : **contrôle décentralisé d'un essaim de robots simulés par un réseau de neurones sur graphes (GNN)**, pour la formation de formes (cercle, lettres, chiffres, nombres). Contexte : mémoire de master IA et data science. Le code est un banc d'essai de recherche : la **reproductibilité et l'honnêteté des mesures** comptent plus que les performances brutes.

**Implémentation reprise à zéro** : tout est à construire, dans l'ordre de `PLAN.md` (liste de tâches par phase). Lire d'abord : `PLAN.md`, puis `docs/` (détails). Index : `docs/README.md`.

## 1. Règles d'or (non négociables)

1. **Décentralisation à l'exécution.** Un robot ne voit que : son état, ses voisins à portée `r_c`, la consigne de mission partagée. Aucune position de robot hors portée, aucune affectation globale robot→point calculée pendant l'exécution.
2. **Paramètres partagés.** Un seul jeu de poids pour tous les robots ; aucun identifiant de robot en entrée.
3. **Entraînement centralisé autorisé** (perte globale, simulateur différentiable), exécution locale uniquement.
4. **Aucun paramètre caché.** Tout nombre qui influence un résultat vit dans un fichier de configuration versionné (graine, nombre de robots, portée, pas de temps, vitesse max, poids de perte, seuil de succès…).
5. **Reproductibilité par graine.** Même configuration + même graine = même épisode. Toute source d'aléa passe par la graine de configuration.
6. **Séparer illustration et preuve.** Une animation n'est pas un résultat. Tout résultat chiffré vient de plusieurs épisodes/graines et est accompagné de sa dispersion.
7. **Comparaisons à information équivalente.** Ne jamais présenter un écart GNN vs baseline comme un apport du GNN si la baseline ne reçoit pas la même consigne (voir `docs/07-baselines-ablations.md`).
8. **L'interface web n'exécute aucune règle de contrôle** : elle affiche des trajectoires calculées par le modèle côté serveur.

## 2. Architecture en un coup d'œil

Modules indépendants, échangeant des artefacts explicites (détail : `docs/01-architecture.md`) :

| Module | Rôle |
|---|---|
| Simulation | dynamique, voisinage, arène, pannes, animation |
| Formes cibles | rendu, binarisation, échantillonnage en nuages de points |
| Contrôleurs | GNN et règle locale de référence, même interface d'épisode |
| Entraînement | boucle différentiable, pertes, curriculum, sauvegardes |
| Évaluation | scénarios (taille, pannes), CSV bruts, agrégats |
| Démonstration Web | API locale de visualisation (lecture seule) |
| Génération des figures | figures construites après coup à partir des sorties |

Principe : le simulateur ne connaît pas la politique ; le modèle ne connaît pas la mécanique de l'arène ; les scripts d'évaluation lisent des fichiers, pas des objets en mémoire.

## 3. Conventions de code

- Python 3 + **PyTorch**. Pas de bibliothèque de GNN : l'échange de messages est écrit à la main pour garder l'agrégation lisible.
- Le graphe de voisinage est **reconstruit à chaque pas** (pas de mise à jour incrémentale) ; pas d'auto-boucle dans l'adjacence.
- Agrégation des messages par **moyenne** ; voisinage vide → agrégat nul ; la commande doit rester valide pour un robot isolé.
- Commande de vitesse **bornée** (norme ≤ vitesse max) avant la mise à jour de position.
- Les figures ne pilotent jamais la simulation ; elles lisent les positions enregistrées.
- Sorties d'une campagne : un dossier nommé `<expérience>/<niveau>` contenant modèle final, meilleur modèle, configuration effective, pertes par itération, métriques par cible, liste des cibles vues. Les binaires lourds ne sont pas versionnés.

## 4. Commandes attendues (à aligner avec le dépôt réel)

```
# générer les formes           -> configs/shapes.yaml
# entraîner (cercle)           -> configs/training-circle.yaml
# entraîner (curriculum)       -> configs/curriculum.yaml
# campagne complexe            -> configs/curriculum-complex.yaml
# vérification rapide          -> configs/curriculum-complex-smoke.yaml  (pipeline seulement)
# évaluer (pilote)             -> configs/evaluation-pilot.yaml
# tests                        -> pytest
# démo locale                  -> serveur sur la boucle locale uniquement
```

## 5. Définition de « terminé » pour une tâche

- Tests unitaires à jour et verts (propriétés, pas valeurs figées : voir `docs/09-tests.md`).
- Configuration versionnée et sauvegardée à côté des résultats.
- Au moins 10 graines (pilote) ou **50** (campagne finale) ; moyenne ± écart-type, intervalles de confiance en campagne finale.
- Seuil de succès affiché avec chaque pourcentage de réussite.
- Limites et hypothèses mentionnées dans la doc concernée.

## 6. À ne pas faire

- Changer un hyperparamètre dans le code au lieu de la config.
- Évaluer un modèle avec des poids qui bougent, ou ajuster sur les scénarios de test.
- Déclarer une lettre « lisible » sur la seule distance de Chamfer.
- Mélanger les résultats des campagnes de configurations différentes sans le préciser (information de consigne différente : identifiant de forme seul vs vecteur vers le contour).
- Ajouter bruit, délais, obstacles ou dynamique réaliste **avant** que les baselines équitables existent (voir PLAN, étape 6).

## 7. Points ouverts à connaître

Voir `docs/12-decisions-et-risques.md` : intitulé (« cellulaires » vs « sur graphes »), traitement des robots retirés, définition exacte du succès, comparabilité des mesures entre tailles d'essaim.
