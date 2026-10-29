# 016 — Valider la navigation corrigée et auditer ses métriques

Statut : terminé. Suite de 015, conformément à la demande de poursuivre après documentation et commit.

Suite complète finale : 604 tests réussis. Vérification navigateur : campagne courante visible, détour apparié réussi, destination occupée refusée avant calcul avec explication.

## Protocole enregistré avant exécution

- 12 robots, poids de référence inchangés ; classique et GAT protégé sur départs appariés.
- Obstacle sur la route : carré, rectangle et triangle, graines 8600 et 8601, soit six cas par contrôleur.
- Régressions complémentaires : carré, départ 8600, ligne droite et contournement, soit deux autres cas par contrôleur.
- Seuils inchangés : erreur finale/destination ≤ 0,10 m, déformation en déplacement ≤ 0,20 m, vitesse ≤ 0,05 m/s pendant 1 s ; aucun contact.

## Résultats

Classique : 8/8 réussis. GAT protégé : 8/8 réussis. Aucun contact robot/obstacle/mur sur les seize trajectoires complètes.

GAT : déformation maximale 0,0100614 m ; distance finale maximale à destination 0,0069176 m. Audit indépendant : intégration physique, bornes, horloge, contacts balayés, appariement des départs, métriques et maintien de stabilisation vérifiés sur 16 archives, aucune divergence.

Les huit cas réutilisent des placements selon forme/parcours : ils ne sont pas huit tirages indépendants. La marge minimale aux obstacles sur l'expérience complète est 0,137991 m pendant les phases incluant la préparation ; ne pas la présenter comme une garantie uniforme de 0,15 m ou de la marge planifiée 0,35 m.

Preuves versionnées : `evidence/016-planned-navigation-{evaluation,audit}.json`. Archives physiques et exports : `outputs/navigation-fix/fresh-8600-8601/`. L'interface affiche cette campagne séparément des archives 009.

Reproduction : `python -m src.reference.evaluate_planned_navigation --output outputs/navigation-fix/nouveau-dossier`.

Décision : [32-validation-navigation-planifiee](../docs/decisions/32-validation-navigation-planifiee.md). Étape suivante proposée : élargir les scènes et effectifs de navigation ; le GAT structuré brut et la découverte d'obstacles inconnus demandent une évaluation distincte.
