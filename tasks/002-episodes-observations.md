# 002 — Épisodes, observations locales et trajectoires

Statut : validée par l’utilisateur. Commit autorisé par l’utilisateur.
Référence : PLAN.md §2 et §7 ; décisions : docs/decisions/08-episodes-observations-locales.md.

## Fonctionnalité

- Configuration YAML dédiée aux unités SI et chargement explicite.
- Épisodes à accélérations avec état position/vitesse, initialisation reproductible et état initial optionnel.
- Contrôleur appelé par robot avec observations locales copiées, sans vue globale.
- Graphe dynamique incluant les robots superposés et les voisins exactement à portée.
- Historique position/vitesse, accélérations demandées/bornées, communications et temps.
- Export NPZ autonome avec paramètres, graine et unités.

## Vérification

Tests ciblés : tests/test_dynamics.py et tests/test_acceleration_episode.py.
Aucun entraînement n'est exécuté. Les tests historiques restent hors vérification : PyTorch absent de l'environnement Linux disponible.

## Suite après validation et commit

Tâche 003 : consigne circulaire partagée et professeur local avec freinage et espacement. Le nouveau moteur est maintenant utilisable par épisodes Python ; la démonstration historique utilise encore l'ancien parcours.

Résultat : 62 tests réussis (34 du moteur + 28 des épisodes). Commit précédent : b3644eb (tâche 001, auteur Labeya).
