# 001 — Noyau physique à accélérations

Statut : réalisation terminée ; 34 tests unitaires réussis ; validée par l’utilisateur ; commit autorisé.
Référence : PLAN.md §2 et §7. Décisions : docs/decisions/07-migration-plan-acceleration.md.

## Livraison

- DynamicsConfig explicite et validée : dt, arène, vitesse, accélération, rayon.
- AccelerationDynamics.step retourne positions et vitesses sans mutation des entrées.
- Inertie, bornage vectoriel, confinement des disques et arrêt normal aux murs.
- Tests unitaires physiques et de validation ; aucune optimisation ni collecte d'entraînement.

## Critères de réception

- À accélération nulle, la vitesse persiste hors contact.
- Les normes de vitesse et d'accélération respectent leurs bornes.
- Le rayon physique est pris en compte aux deux murs opposés.
- Une entrée invalide produit une erreur explicite.
- Les tests du simulateur historique restent valides.

## Suite après validation et commit

Tâche 002 : raccorder des épisodes, un état incluant les vitesses, une configuration et des observations locales au nouveau moteur. L'ancien moteur reste le parcours historique jusqu'à cette intégration.

## Vérification du 2026-10-28

`tests/test_dynamics.py` : 34 tests réussis (Python 3.12, dépendances temporaires dans /tmp/swarm-gnn-test-deps).
La vérification historique `tests/test_simulator.py` est bloquée à la collecte par l’absence de PyTorch dans cet environnement. Aucun résultat de non-régression ne peut donc être affirmé. Le .venv du projet contient une structure Windows, sans interpréteur Linux.
Aucun entraînement exécuté.
