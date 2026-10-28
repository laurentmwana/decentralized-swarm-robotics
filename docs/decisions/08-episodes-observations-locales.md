# 08 — Épisodes à accélérations et observations locales

Date : 2026-10-28. Statut : validée par l’utilisateur (tâche 002).

## Contrat et raisons

- Le nouveau parcours est `AccelerationEpisode`, configuré par `config/acceleration-simulator.yaml`. Le moteur historique et ses interfaces restent disponibles pendant la migration. La démonstration n'est pas encore raccordée au nouveau parcours.
- Chaque appel au contrôleur reçoit uniquement la position et vitesse propres, les positions et vitesses relatives des voisins à portée et le temps courant. Aucun tableau global, aucune affectation ni position d'un robot distant n'est fourni. Une future consigne de formation partagée sera définie avec le professeur (tâche 003).
- Un contrôleur est une fonction `observation -> accélération (2,)`, sans dépendance à PyTorch. Le même état physique est utilisé pour tous les robots avant la mise à jour simultanée. Les observations sont des copies en lecture seule, pour protéger le moteur et permettre leur conservation.
- Graphe recalculé à chaque pas et au dernier état ; seuil inclusif, sans auto-boucles. Deux robots superposés sont voisins : exclure une distance nulle ferait disparaître précisément une situation de collision. Cette règle corrige ce cas dans le nouveau parcours sans modifier le graphe historique.
- Tirage uniforme en aire dans un disque centré, via SeedManager. La graine et la configuration rendent les épisodes reproductibles pour un contrôleur déterministe sans état caché. L'appelant doit réinitialiser un éventuel contrôleur avec mémoire entre épisodes. Les positions et vitesses initiales peuvent être fournies pour les expériences appariées.
- L'état initial est contrôlé avant tout appel à la politique. Les vitesses initiales sont nulles par défaut. Les superpositions initiales sont autorisées et ne seront pas cachées aux futures métriques.
- L'archive NPZ contient les états T+1, les commandes T demandées et bornées, le graphe T+1, les temps, la graine, les paramètres et les unités SI ; lecture possible sans pickle. Conserver les commandes demandées permet de diagnostiquer les saturations. Les accélérations bornées sont les entrées d'intégration, pas une estimation de l'accélération effective lors d'un contact avec un mur.

## Périmètre restant

Les contacts entre robots ne sont pas résolus mécaniquement. Leur détection et leur comptage seront ajoutés avec les métriques de la tâche 004. Les pannes historiques ne sont pas reprises dans ce premier parcours : elles ne sont pas un prérequis du nouveau PLAN.md. Les observations, trajectoires et exports sont disponibles par l'API Python ; l'interface de démonstration est prévue en tâche 010.
