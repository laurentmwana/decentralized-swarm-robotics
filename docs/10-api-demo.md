# 10 — API locale de démonstration

Outil de visualisation **local**, pas un système de contrôle temps réel : pas d'accès Internet, pas de WebSocket, pas de base de données, écoute sur la boucle locale uniquement.

| Route | Objet | Réponse |
|---|---|---|
| GET /api/health | disponibilité du modèle | statut et chemin du modèle chargé |
| GET /api/targets | cibles connues | identifiants et fichiers sources |
| GET /api/simulate | exécution d'un épisode | cible, modèle employé, positions par pas |

## Règles
- Le calcul est fait par le modèle PyTorch côté serveur ; la page web dessine seulement (cible + positions successives). **Interdit** de réimplémenter des règles de contrôle en JavaScript.
- Le modèle chargé est celui du dernier niveau du curriculum.
- Rejouabilité : le couple (forme, graine) donne exactement les mêmes positions.
- Le contrat JSON est documenté ici pour permettre de réécrire l'interface sans toucher au modèle.

## À compléter
Schéma JSON précis de `/api/simulate` (champs, unités, repère), paramètres acceptés (forme, graine, nombre de robots, retrait éventuel).
