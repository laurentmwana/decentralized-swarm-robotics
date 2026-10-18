# 11 — Reproductibilité et traçabilité

- **Graine** : pivot de tout. Elle commande positions initiales, ordre des cibles, scénarios de retrait. Même graine + même code = même épisode.
- **Une graine n'est pas un résultat général** : cumuler plusieurs graines (10 en pilote, 50 en campagne finale) et plusieurs graines d'**entraînement**.
- **Sortie d'une campagne** (`<expérience>/<niveau>`, ex. geometry, digits, symbols, numbers) : modèle final, meilleur modèle, configuration effective, pertes par itération, courbe, métriques par cible, liste des cibles vues. Les niveaux coexistent sans s'écraser (comparaisons a posteriori).
- **Versionné** : code, configurations, documentation, résumés. **Non versionné** : modèles, animations, tableaux bruts, données générées (régénérables).
- **Évaluation** : poids gelés ; fichiers bruts à une ligne par épisode, pour recalculer moyennes, écarts-types et figures.
- **Figures** construites après coup depuis les positions enregistrées.
- Consigner pour chaque résultat : configuration, graine, seuil τ, nombre d'épisodes, modèle utilisé.
