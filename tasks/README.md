# Migration vers PLAN.md

Référence : PLAN.md à la racine, laissé intact. Début : 2026-10-28.
Chaque tâche livre une fonctionnalité, ses tests et ses décisions dans docs/decisions/.
Cycle : réalisation → tests → résumé → validation utilisateur → commit limité aux fichiers de la tâche → tâche suivante.
Aucun entraînement n'est lancé par l'agent ; les commandes seront remises à l'utilisateur.
Les modifications présentes avant cette migration sont préservées.

| Tâche | Livraison et critère de fin | État |
|---|---|---|
| 001 | Moteur à accélérations : inertie, bornes, rayon physique, murs ; tests unitaires | Validée, commit b3644eb, 34 tests réussis |
| 002 | Épisodes du nouveau moteur, observations locales et trajectoires position/vitesse/commande ; tests de portée et reproductibilité | Validée, commit 219c9e5, 28 tests nouveaux réussis |
| 003 | Consigne circulaire et professeur local : rayon, espacement, freinage et évitement ; tests de comportement | Validée, commit f60448d, 23 tests nouveaux réussis |
| 004 | Quatre métriques : erreur en mètres, stabilisation en secondes, contacts, réussite ; protocole et tests | Validée, commit 3e23a47, 32 tests nouveaux réussis |
| 004b | Corriger le professeur depuis des positions dispersées sous le protocole 004 ; tests de régression et nouvelles graines | Validée ; pilote 9/10, validation 17/20 |
| 005 | Collecte du professeur et séparation par épisodes entraînement/validation/test ; tests anti-fuite | À faire |
| 006 | GAT partagé à une couche, prédiction d'accélérations et MSE supervisée ; tests sans optimisation réelle | À faire |
| 007 | Intégration GAT, commandes d'entraînement à lancer par l'utilisateur, comparaison sur épisodes inédits | À faire |
| 008 | Ligne et V : affectation locale, collecte adaptée et transitions ; tests | À faire |
| 009 | Déplacement collectif puis obstacles ; tests de navigation et échecs explicites | À faire |
| 010 | Interface et exports adaptés, comparaisons reproductibles ; tests d'intégration | À faire |

Les métriques sont préparées avant la collecte. Les formations suivantes ne sont entraînées qu'après validation de leur répartition. Les anciens modèles commandant des vitesses ne sont pas réutilisés comme modèles à accélérations.
