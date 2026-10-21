# 06 — Évaluation : définitions, formule du succès, seuil

Daté : 2026-10-21. Figé **avant** toute campagne (plan §0.5 et §0.2-1 : on ne change plus rien après avoir vu les résultats).

## Distance de Chamfer symétrique
- Formule en mots : moyenne, sur les robots **actifs**, de la distance euclidienne au point de cible le plus proche, **additionnée** à la moyenne, sur les points de cible, de la distance euclidienne au robot actif le plus proche. C'est la **moyenne des distances (non au carré)** — « distance moyenne aller + distance moyenne retour », deux termes.
- Normalisation : le numérateur est divisé par le **rayon de la cible** (= 1.0 par construction, voir 04-shapes.md). La grandeur est donc adimensionnée et **comparable entre 20, 40 et 100 robots** et entre formes.
- Toujours recalculable depuis les seules positions enregistrées (contrat de la phase 2).

## Succès — DÉCISION UNIQUE
Deux options étaient permises par le plan (« distance finale sous le seuil » OU « franchissement à un moment donné »). **Choix figé : franchissement stabilisé.**

- `success_threshold` : valeur dans la config de run ; **sens physique : 0.15 = 15 % du rayon de la forme** (un essaim est déclaré « en forme » quand sa distance moyenne aux deux côtés est sous 15 % de la taille de la forme).
- `convergence_step` : **premier pas k tel que la distance est ≤ seuil à tous les pas de k à T** (franchissement définitif ; si l'essaim repasse au-dessus, k est re-recalculé plus tard).
- `success = true` ssi `convergence_step` existe **et** `convergence_step <= T`.
- Conséquences de la formule :
  - un épisode dont la **distance initiale dépasse le seuil ne peut jamais être déclaré réussi au pas 1** : le pas 1 ne peut être `k` que si la distance y est déjà sous le seuil — ce cas est bloqué par le test obligatoire de la phase 2 ;
  - un essaim qui démarre déjà sous le seuil converge au pas 0, et c'est consigné comme tel (position initiale sous le seuil = cas limite testé, plan phase 2 « juste en dessous, égal, juste au-dessus »).
- Épisode trop court = succès sous-estimé : la durée `horizon` de chaque config est au moins le temps de convergence observé + marge (règle de la phase 6).

## Rapport à deux niveaux de variabilité (structure du rapport d'évaluation)
1. **Entre graines d'entraînement** (≥ 5) : chaque graine = un modèle ; on rapporte moyenne ± écart-type des métriques de campagne **par graine**, puis sur les graines.
2. **Entre épisodes à graine fixée** (50 répétitions par condition) : moyenne ± écart-type et intervalle de confiance 95 % ( Student t, n=50) sur les épisodes appariés.
- Les comparaisons entre contrôleurs sont **appariées** : mêmes positions initiales, mêmes graines d'épisode, mêmes cibles ; la différence est calculée épisode par épisode, puis agrégée (jamais agrégat contre agrégat).
- Chaque pourcentage de réussite est affiché **avec son seuil** (ex. « succès 0.94 (seuil 0.15) »).

## Collisions, régularité, connectivité (données ici pour figer les formules, implémentées phase 2)
- Taux de collisions : fraction des paires (i<j) actives ayant été à distance ≤ `collision_radius` au moins un pas, **normalisée par (nombre de paires actives × durée)**, mesurée sur les robots actifs seulement.
- Régularité de répartition : écart-type des distances au plus proche voisin parmi les robots actifs, en **valeur absolue**, rapporté à l'espacement idéal sur la cible.
- Connectivité : nombre de composantes connexes du graphe de communication **à chaque pas** ; on rapporte la valeur au pas final et la moyenne sur l'épisode.
