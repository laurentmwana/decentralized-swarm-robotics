# Étape 6 — Évaluation et expériences

## Objectif
Mesurer honnêtement ce que le modèle sait faire, le comparer à des références, et tester sa robustesse. C'est le cœur des résultats du mémoire.

## Métriques (en mots)
- **Qualité de la forme** : distance moyenne entre les robots et la forme cible, et inversement (aucun trou, aucun robot hors forme).
- **Régularité** : écart entre les distances aux plus proches voisins (une forme nette a des robots bien répartis).
- **Collisions** : nombre de paires de robots trop proches pendant l'épisode.
- **Temps de convergence** : nombre de pas pour atteindre une qualité donnée.
- **Taux de réussite** : proportion d'épisodes où la qualité dépasse un seuil fixé à l'avance.
- **Lisibilité** (pour lettres et chiffres) : un petit classifieur d'images, ou une évaluation visuelle, reconnaît-il la forme obtenue ?

Toujours rapporter la moyenne **et** la dispersion sur de nombreux départs aléatoires (au moins 50 par condition).

## Expériences à réaliser
1. **Formes de base** : performance par catégorie (géométrique, lettres, chiffres, nombres).
2. **Généralisation aux formes inédites** : formes gardées pour le test.
3. **Pannes** : retirer 5 %, 10 %, 20 % des robots en cours d'épisode ; mesurer la dégradation et la capacité à se réparer.
4. **Passage à l'échelle** : entraîné avec N robots, testé avec 2N, 5N, 10N.
5. **Portée de communication** : réduire le rayon pour voir quand le groupe se fragmente.
6. **Bruit** : ajouter du bruit aux mesures de position.
7. **Ablations** (retirer un élément pour mesurer son utilité) : sans échange de messages, avec 1, 2, 3 tours d'échange, avec ou sans attention.
8. **Comparaison aux références** : règles écrites à la main, réseau sans échange.

## Protocole
- Jeu de test fixé avant de regarder les résultats.
- Graines multiples pour l'entraînement (au moins 3) et pour les départs.
- Toutes les expériences décrites par un fichier de configuration dans `configs/`.
- Résultats sauvegardés en tableau (CSV) puis transformés en figures par un script.

## Tableaux et figures à produire pour le mémoire
- Tableau principal : métriques par catégorie de formes, modèle contre références.
- Courbe : qualité en fonction du pourcentage de pannes.
- Courbe : qualité en fonction du nombre de robots.
- Planche d'images : formes obtenues, réussies et ratées.
- Animation ou séquence d'images : formation d'une forme et auto-réparation.

## Analyse des échecs
Pour chaque type d'échec (forme déformée, robots bloqués, groupe fragmenté), montrer un exemple, proposer une explication, et dire si c'est une limite du modèle, de l'entraînement ou du simulateur.

## Tests logiciels de comportement
- Le simulateur et le modèle donnent le même résultat pour une même graine.
- Les métriques valent la valeur attendue sur des cas simples (forme parfaite, forme vide).

## À faire
- [ ] Module de métriques
- [ ] Scénarios de test (pannes, échelle, bruit, portée)
- [ ] Scripts d'exécution de toutes les expériences
- [ ] Tableaux et figures automatiques
- [ ] Analyse des échecs

## Critères de réussite
Chaque affirmation du mémoire est appuyée par un chiffre ou une figure reproductible ; les comparaisons sont équitables.

## Évaluation pilote exécutée

La configuration `configs/evaluation-pilot.yaml` fixe dix graines et évalue un cercle sur trois tailles d'essaim (20, 40 et 100 robots), quatre taux de pannes (0, 5, 10 et 20 %) et deux méthodes : GNN entraîné et règle locale de regroupement. Elle génère 240 épisodes, `episodes.csv`, `summary.csv` et deux figures dans `results/evaluation-pilot/`.

Cette étude est explicitement **pilote** : elle emploie dix épisodes par condition, alors que le protocole final en demande au moins cinquante. La baseline de regroupement est un contrôle comportemental sans consigne explicite de cercle ; elle ne doit donc pas être présentée comme une comparaison informationnellement équivalente au GNN. La baseline apprise sans communication et les essais de bruit ou de portée réduite restent à exécuter.

## Pièges
- Ne présenter que les meilleurs exemples.
- Comparer à des références mal réglées.
- Conclure sur une seule graine.
