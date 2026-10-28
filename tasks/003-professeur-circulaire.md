# 003 — Formation circulaire et professeur local

Statut : validée par l’utilisateur ; commit autorisé.
Références : PLAN.md §3–4 ; docs/decisions/09-professeur-circulaire.md.

## Livraison

- Consigne commune : centre, rayon, effectif et espacement par corde.
- Commande locale d'accélération : rappel radial, amortissement, espacement tangentiel, répulsion de sécurité.
- Paramètres YAML et vérification de faisabilité géométrique et de portée.
- Commande de démonstration sans entraînement : `python -m src.reference.demo`.
- Archive des trajectoires et fichier de consigne/gains pour reproduire le scénario.
- Tests : direction des forces, freinage, invariance à l'ordre des voisins, équilibre, convergence, séparation, configuration et export.

## Limites et suite

L'évitement reste heuristique et les superpositions parfaites ne sont pas résolues. La réussite depuis des positions dispersées doit être mesurée ; elle n'est pas affirmée par ces seuls tests. La tâche 004 définit les métriques et permettra une évaluation du professeur avant la collecte d'imitation.
Aucun entraînement exécuté. Les tests historiques nécessitant PyTorch ne sont pas inclus dans la vérification disponible.

## Vérification

85 tests réussis : 34 du moteur, 28 des épisodes et 23 du professeur circulaire et de sa démonstration.
