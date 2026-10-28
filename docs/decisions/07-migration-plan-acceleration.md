# 07 — Migration vers le plan à accélérations

Date : 2026-10-28. Statut : implémenté pour la tâche 001, validé par l’utilisateur.

## Constat

PLAN.md demande des accélérations, un GAT à un échange et une imitation supervisée.
L'existant utilise un simple intégrateur commandé en vitesse (décision 02), un GNN configurable et une perte géométrique sur trajectoires (src/training/losses.py).
AGENT.md est vide au début de la migration. De nombreuses modifications utilisateur sont déjà présentes.

## Décisions et raisons

- PLAN.md est la référence de la nouvelle réalisation. Les anciens documents décrivent le fonctionnement historique ; cette décision remplace la règle de dynamique de 02 pour le nouveau parcours uniquement.
- Conserver provisoirement l'ancien parcours et ajouter AccelerationDynamics dans src/simulator/dynamics.py : remplacer silencieusement des vitesses par des accélérations rendrait les anciens contrôleurs et modèles incorrects. Le raccordement des épisodes est la tâche 002 ; l'application existante ne change pas encore de moteur.
- Paramètres explicites dans DynamicsConfig : côté d'arène et rayon en mètres, dt en secondes, vitesse en m/s, accélération en m/s². Aucun ancien résultat normalisé ne doit être renommé comme résultat SI. Le chargement de configuration sera raccordé avec les nouveaux épisodes.
- Euler semi-implicite : borner la norme d'accélération, intégrer et borner la norme de vitesse, puis avancer la position. Choix simple et déterministe, qui conserve l'inertie quand la commande est nulle.
- Aux murs, garder le disque entier dans l'arène et annuler la composante normale de vitesse, sans rebond ; conserver le mouvement tangent. Les collisions entre robots seront détectées et comptabilisées séparément, sans correction qui masquerait les échecs du contrôleur.
- Refuser les états non finis, les formes incompatibles, les positions hors limites et les vitesses initiales excessives, pour rendre les erreurs visibles dès leur origine.
- Livrer par tâches avec tests unitaires et validation avant chaque commit. Aucun entraînement automatique, y compris dans les tests de cette migration.

## Limites de la tâche 001

Ce noyau ne fournit encore ni épisodes, ni observations, ni détection de contacts entre robots, ni contrôleur de cercle. Il ne constitue pas à lui seul le simulateur complet du plan. Ces éléments sont explicitement suivis dans tasks/README.md.
