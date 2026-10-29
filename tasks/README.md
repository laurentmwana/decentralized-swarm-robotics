# Migration vers PLAN.md

Référence : PLAN.md à la racine, laissé intact. Début : 2026-10-28.
Chaque tâche livre une fonctionnalité, ses tests et ses décisions dans docs/decisions/.
Cycle : réalisation → tests → résumé → validation utilisateur → commit limité aux fichiers de la tâche → tâche suivante.
Règle initiale : commandes d'entraînement remises à l'utilisateur. Le 2026-10-29, l'utilisateur a explicitement autorisé l'agent à lancer l'entraînement de la variante relative et à poursuivre l'étape suivante.
Les modifications présentes avant cette migration sont préservées.
Le 29 octobre 2026, l'utilisateur a demandé explicitement de mettre à jour tâches et décisions, créer les commits puis poursuivre : cette autorisation s'applique au cycle courant sans nouvelle confirmation.

| Tâche | Livraison et critère de fin | État |
|---|---|---|
| 001 | Moteur à accélérations : inertie, bornes, rayon physique, murs ; tests unitaires | Validée, commit b3644eb, 34 tests réussis |
| 002 | Épisodes du nouveau moteur, observations locales et trajectoires position/vitesse/commande ; tests de portée et reproductibilité | Validée, commit 219c9e5, 28 tests nouveaux réussis |
| 003 | Consigne circulaire et professeur local : rayon, espacement, freinage et évitement ; tests de comportement | Validée, commit f60448d, 23 tests nouveaux réussis |
| 004 | Quatre métriques : erreur en mètres, stabilisation en secondes, contacts, réussite ; protocole et tests | Validée, commit 3e23a47, 32 tests nouveaux réussis |
| 004b | Corriger le professeur depuis des positions dispersées sous le protocole 004 ; tests de régression et nouvelles graines | Validée, commit 38b1941 ; pilote 9/10, validation 17/20 |
| 005 | Collecte du professeur et séparation par épisodes entraînement/validation/test ; tests anti-fuite | Validée, commit 164a69e, 27 tests nouveaux réussis ; qualité des données à examiner |
| 006 | GAT partagé à une couche, prédiction d'accélérations et MSE supervisée ; tests sans optimisation réelle | Validée, commit 9f915cf, 36 tests nouveaux réussis |
| 007 | Intégration GAT, commandes d'entraînement à lancer par l'utilisateur, comparaison sur épisodes inédits | Entraînement retrouvé ; comparaison réelle : professeur 5/5, GAT 0/5, 22 collisions ; 21 tests d'intégration réussis lors de la reprise |
| 007b | Diagnostic du cercle appris et correction sur train/validation avant nouveau test indépendant | Entraîné : MSE 0,185083 ; validation 5/5 ; campagne indépendante 11/20, 5 collisions ; suite 007c |
| 007c | Corrections du professeur sur états visités en train, puis évaluation indépendante | Entraîné : MSE 0,161719 ; validation 5/5 ; campagne 600–619 : 17/20, 2 collisions ; robustesse résiduelle documentée |
| 008 | Formations générales : A–Z, 1–10, au moins 15 géométries ; répartition locale et transitions | Catalogue 56 formes, GAT protégé entraîné et transitions évaluées ; détails 008b–008e |
| 008b | Reconnexion initiale et défauts de stabilisation du professeur | Implémenté, 314 tests ; 112/112 conservés puis 168/168 nouveaux ; replis du filtre signalés |
| 008c | Contrat et collecte d'imitation multi-formes | Schéma v2 et pilote 56/56 réussis ; 37 train / 10 validation / 9 test (54 uniques), audit complet passé ; 326 tests |
| 008d | GAT à cible locale et imitation multi-formes | Entraîné et testé, 348 tests ; réservé : protégé 9/9 sans collision, brut 0/9 (54 contacts) ; prototype hybride retenu |
| 008e | Transitions continues entre formations | Implémenté, 401 tests ; pilote 6/6 et validation 2/2 séquences par contrôleur, sans collision observée ; états/poids conservés |
| 009 | Déplacement collectif puis obstacles ; tests de navigation et échecs explicites | Implémenté, 412 tests ; pilote et validation : 6/6 nominaux, 0/4 difficiles par contrôleur/campagne, sans contact ; audit 40 trajectoires passé |
| 010 | Interface et exports adaptés, comparaisons reproductibles ; tests d'intégration | Implémenté, 423 tests ciblés ; catalogue 56 formes, transitions/navigation, comparaison appariée, lecture et exports ZIP/JSON/CSV/NPZ ; limites conservées |
| 011 | Réduire les contacts du GAT brut par corrections sur ses propres états ; nouvel entraînement et test frais | Première expérience évaluée : contacts 102→65, brut 0/18, protégé 18/18 ; autonomie sûre non atteinte, modèle de l’interface conservé |
| 012 | Apprentissage multi-effectifs, dont 24 et 64 ; dimensions, budgets et évaluations par effectif | Pilote entraîné et évalué, 590 tests ; brut 2/3 à 12, 0/3 à 24/64 ; protégé 3/3 à 12, 0/3 à 24/64 ; généralisation non validée |
| 013 | Retirer l’ancienne base inutilisée et ses commandes/configurations/tests | Terminé sur autorisation utilisateur : 96 fichiers obsolètes retirés, 578 tests conservés réussis ; résultats et supports préservés |
| 014 | Corriger l’équilibre et obtenir un mouvement appris brut aux trois effectifs | Objectif atteint dans le domaine contrôlé : 18/18 réservés sans contact, aucun filtre sur mouvements 0/3, audit 36 trajectoires, 597 tests ; départs généraux/navigation/obstacles non couverts |
| 015 | Planifier les détours de toute la formation et clarifier les résultats | Corrigé : carré avec obstacle réussi par les deux contrôleurs, destination impossible détectée avant calcul, métriques de navigation visibles ; 600 tests |
| 016 | Vérifier la navigation corrigée sur de nouveaux départs et auditer les métriques | Terminée : 8/8 par contrôleur sans contact, audit physique et métriques de 16 trajectoires passé ; trois formes, deux départs et parcours complémentaires ; 604 tests |
| 017 | Expliquer le modèle selon l’effectif et proposer cinq à dix obstacles | Terminé : six comptes publics 5–10, 8/8 par contrôleur sans contact avec le carré à 12 ; audit de 16 trajectoires, préparation classique déclarée, protection des murs et port serveur exclusif ; 621 tests |
| 018 | Adapter les dimensions et prévenir densité, déconnexion et dépassement | Terminé : aperçu avant lancement, largeur adaptée, options incompatibles désactivées ; 336 combinaisons examinées, deux régressions classiques réussies sans contact, 634 tests |
| 019 | Clarifier profils et cercle initial, rapprocher les obstacles et documenter chaque commande | Terminé : guide Markdown, messages explicites vérifiés, nouvelle scène proche des cibles ; 8/8 par contrôleur sans contact, audits de 16 trajectoires, 634 tests |
| 020 | Rendre l’animation visible sur petit écran et permettre l’accès direct au résultat | Terminé : défilement et lecture automatiques, index de lecture sécurisé, résultat final visible à 64 robots ; 17 tests ciblés réussis |
| 021 | Reprendre le mémoire sur les résultats actuels, en français simple et avec des figures issues du programme | Terminée : rédaction alignée, cinq figures de données, contrôle documentaire et relecture du PDF de 45 pages ; français simple et résultats limités au domaine testé |

Les métriques sont préparées avant la collecte. Les formations suivantes ne sont entraînées qu'après validation de leur répartition. Les anciens modèles commandant des vitesses ne sont pas réutilisés comme modèles à accélérations.

Point qualité : la collecte brute 005 conserve l’échec 113. La version revue 007 l’exclut explicitement de train, avec provenance, sans modifier validation/test. Les corrections train conservent les échecs du collecteur. Sur les mêmes graines 600–619, le modèle relatif initial réussit 70 % et le corrigé 85 % ; les collisions résiduelles restent explicites. La géométrie 008 progresse indépendamment ; l'affectation locale doit être validée avant tout apprentissage des nouvelles formes.
