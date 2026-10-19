# PLAN.md — TODO : reprise de l'implémentation à zéro

Tout est à refaire. Cases à cocher : `[ ]` à faire · `[x]` fait. Chaque phase a un **critère d'acceptation** ; ne pas passer à la phase suivante tant qu'il n'est pas atteint.
Le mémoire initial sert de **référence de spécification** (valeurs de départ dans `docs/`), pas de code à recopier.

## Leçons du premier prototype (à appliquer dès le départ)
- Une seule définition du **succès**, écrite avant de coder les métriques.
- Une seule règle pour les **robots retirés**, testée.
- Mesure de distance **comparable entre tailles d'essaim**.
- **Baselines équitables** construites dès le début, pas à la fin.
- Épisodes **assez longs** pour atteindre le seuil ; **50 répétitions** ; plusieurs graines d'entraînement.
- Les paramètres vivent dans les configurations, jamais dans le code.

---

## Phase 0 — Décisions et squelette
- [x] Trancher l'intitulé avec l'encadrant (« cellulaires » ou « sur graphes »).
- [x] Décider : robot retiré = supprimé du graphe **ou** figé et présent. Écrire la règle dans `docs/02-simulateur.md`.
- [x] Définir le succès (distance finale sous le seuil, ou franchissement à un moment) et le seuil. Écrire dans `docs/06-evaluation.md`.
- [x] Décider comment rendre la distance comparable entre tailles (résolution de cible fixe ou normalisation).
- [x] Choisir le repère et la consigne : repère commun 2D, identifiant de forme partagé (option A) ; vecteur vers le contour en variante séparée.
- [x] Créer le dépôt : `configs/`, `src/`, `tests/`, `docs/`, `outputs/` (non versionné), `figures/`.
- [x] Environnement Python + PyTorch figé (fichier de dépendances), outil de tests, formatage.
- [x] Chargeur de configuration (fichier versionné → objet), avec sauvegarde de la configuration effective.
- [x] Gestion centralisée des graines (une seule source d'aléa par épisode).
**Critère** : les décisions ci-dessus sont écrites ; un test vide passe ; une configuration se charge et se réécrit à l'identique.

## Phase 1 — Simulateur 2D
- [x] Arène carrée, état par robot (position, vitesse), vitesse directement commandée.
- [x] Commande bornée par la vitesse max, appliquée avant la mise à jour de position.
- [x] Limites d'arène.
- [x] Graphe de communication par portée, **recalculé à chaque pas**, sans auto-boucle.
- [x] Pannes : sélection aléatoire à mi-épisode, selon la règle décidée.
- [x] Interface d'épisode unique (cible, positions initiales, graine, contrôleur) utilisable par tous les contrôleurs.
- [x] Enregistrement des positions par pas.
- [x] Tests : portée respectée, vitesse bornée, robot isolé valide, robot en panne immobile et hors collisions, même graine = même épisode.
**Critère** : un contrôleur trivial tourne sur 20, 40 et 100 robots, reproductible.

## Phase 2 — Formes cibles
- [x] Rendu 128×128 (géométries tracées, caractères par police).
- [x] Binarisation au milieu de la plage de gris.
- [x] Échantillonnage par point le plus éloigné, graine contrôlée.
- [x] Normalisation dans un cadre commun, à l'échelle du rayon de cible.
- [x] Jeu : 7 géométries, 26 lettres, 10 chiffres, nombres à trois caractères (puis 9 à 999).
- [x] Deux résolutions (20 et 50 points) ; envisager 80 à 100 pour les nombres.
- [x] Index + image d'origine + planche de contrôle.
- [x] Découpage entraînement / validation / test **déclaré dès maintenant** (formes réservées pour la généralisation).
- [x] Tests : normalisation, reproductibilité, présence de l'index.
**Critère** : planche inspectée visuellement ; régénération identique avec la même graine.

## Phase 3 — Métriques
- [x] Distance de Chamfer symétrique (préciser si au carré) ; indépendante de la taille selon la décision de phase 0.
- [x] Taux de collisions (normalisé par paires et durée) ; mesure sur robots actifs.
- [x] Régularité de répartition (valeur absolue).
- [x] Succès et pas de convergence selon la définition unique.
- [x] Connectivité du graphe pendant l'épisode (nombre de composantes).
- [x] Courbe de distance au cours du temps.
- [x] Tests : valeurs exactes sur cas simples, paire en collision, cas limites autour du seuil.
**Critère** : toutes les mesures recalculables depuis les positions enregistrées.

## Phase 4 — Contrôleurs de référence (avant le GNN)
- [x] Règle de regroupement (cohésion–répulsion), sans consigne.
- [x] **Règle manuelle avec consigne de forme** (attraction vers le contour + répulsion).
- [x] Même interface que le GNN ; mêmes tests d'interface.
**Critère** : les deux règles forment un cercle ou échouent de façon documentée ; leurs mesures servent de plancher.

## Phase 5 — Contrôleur GNN
- [x] Encodeur de l'observation (vitesse, identifiant de forme).
- [x] Messages par arête (états des deux robots + position relative), agrégation par moyenne, voisinage vide = agrégat nul.
- [x] Mise à jour, tours configurables (1, 2, 3), décodeur borné.
- [x] Variante **sans communication** (mêmes entrées, pas de messages, taille comparable).
- [ ] Option : agrégation par somme normalisée ; attention (plus tard).
- [x] Écrire les messages à la main (pas de bibliothèque de GNN).
- [x] Tests : invariance par permutation et par translation, tailles variables, robot isolé, communication désactivée, conditionnement par la consigne.
**Critère** : tous les tests passent ; nombre de paramètres indépendant du nombre de robots.

## Phase 6 — Entraînement (cercle)
- [x] Simulateur différentiable (tout en tenseurs).
- [x] Perte à quatre termes : forme, collisions, mouvement, couverture ; poids en configuration.
- [x] Boucle Adam, lots d'épisodes, sauvegarde du meilleur état et du dernier.
- [x] Journal des pertes par itération, configuration effective sauvegardée.
- [x] Tests : pertes proches de zéro sur cas attendus, pas d'auto-boucle, cohérence de la cible.
- [x] Profil de vérification rapide (smoke) du pipeline.
- [x] Entraîner sur le cercle avec **au moins 5 graines**.
**Critère** : la perte décroît sur toutes les graines ; le cercle formé est visuellement correct et la couverture est acceptable.

## Phase 7 — Harnais d'évaluation
- [x] Scénarios : taille d'essaim (20, 40, 100), retraits (5, 10, 20 %), durée d'épisode ≥ temps de convergence observé.
- [x] Mêmes positions initiales réutilisées entre contrôleurs.
- [x] Sortie brute : une ligne par épisode ; agrégats moyenne, écart-type, intervalle de confiance.
- [x] **50 répétitions** par condition.
- [x] Générateur de tableaux et de figures depuis les fichiers bruts.
**Critère** : une commande exécute toute la campagne sur un contrôleur et produit tableaux et figures.

## Phase 8 — Comparaisons à information équivalente
- [x] Évaluer : regroupement, règle manuelle + consigne, GNN sans communication, GNN 1/2/3 tours.
- [x] Budget d'entraînement identique pour les contrôleurs appris.
- [x] Un tableau unique par condition, avec l'information reçue par chaque contrôleur.
- [ ] Option : somme normalisée / attention / agrégation tronquée.
**Critère** : on peut répondre « la communication apporte-t-elle quelque chose à consigne égale ? ».

## Phase 9 — Curriculum de formes
- [ ] Niveaux : géométries → chiffres → lettres → nombres, cibles précédentes conservées.
- [ ] Mesure de la perte **par cible** à chaque niveau (cibles anciennes vs nouvelles) pour détecter l'oubli.
- [ ] Cohérence des journaux (pertes finales et pertes par cible expliquées).
- [ ] Évaluer le modèle spécialisé cercle **et** le modèle du curriculum sur les mêmes épisodes.
**Critère** : pas de dégradation des cibles anciennes ; résultats par forme.

## Phase 10 — Formes complexes (50 robots)
- [x] Configuration dédiée (50 robots, ≥ 50 points, 64 composantes, 3 tours, épisodes longs).
- [x] Variante avec vecteur vers le contour, **séparée** des autres résultats.
- [x] Entraîner tous les niveaux (coût processeur : planifier).
**Critère** : lettres et nombres évalués visuellement et quantitativement.

## Phase 11 — Généralisation et lisibilité
- [ ] Évaluer sur formes réservées, poids gelés.
- [ ] Protocole de lisibilité (évaluateurs humains ou classifieur entraîné sur rendus d'essaims).
- [ ] Exemples favorables **et** défavorables, graines fixes.
**Critère** : chiffre de lisibilité rapporté avec sa méthode.

## Phase 12 — Robustesse
- [ ] Bruit de position, bruit de communication.
- [ ] Réduction systématique de la portée.
- [ ] Robot défaillant qui reste immobile ou transmet des valeurs aberrantes.
- [ ] Suivi de la connectivité et du temps de reformation.
**Critère** : courbes de dégradation avec dispersion.

## Phase 13 — Démonstration Web
- [ ] API locale en lecture seule : `/api/health`, `/api/targets`, `/api/simulate`.
- [ ] Schéma JSON documenté ; rejouable par (forme, graine).
- [ ] Page de visualisation (cible + positions) sans règle de contrôle côté navigateur.
**Critère** : deux lancements identiques donnent les mêmes positions.

## Phase 14 — Figures et rédaction
- [ ] Générer les figures depuis les sorties brutes (style unifié).
- [ ] Mettre à jour les chapitres 3 à 5 avec les nouveaux résultats.
- [ ] Regrouper les limites (une seule section), déplacer l'architecture logicielle en annexe.
- [ ] Page de titre complète, relecture orthographique.
**Critère** : chaque chiffre du texte est traçable jusqu'à un fichier brut.
