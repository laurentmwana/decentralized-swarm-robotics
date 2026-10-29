# 008e — Transitions continues entre formations

Statut : transitions continues implémentées et validation ciblée terminée ; 401 tests réussis. Suite : navigation collective et obstacles en 009.

## Livraison et résultat

- Positions, vitesses et temps conservés entre missions ; aucun état physique réinitialisé artificiellement.
- Versions de mission par occurrence, y compris retours A→B→A ; anciennes revendications rejetées et comptées.
- Trajectoires NPZ continues avec phases, affectations, cibles locales et communications. Collisions globales dédupliquées aux frontières ; échecs et délais trop courts conservés.
- Pilote sur deux départs : professeur et GAT protégé réussissent chacun 6/6 séquences (18 stages), sans collision observée.
- Validation mixte C→7→triangle→C sur deux nouveaux départs : chacun 2/2 séquences (8 stages), sans collision observée. Erreur finale moyenne du GAT protégé 1,17 cm.
- Audit de 16 trajectoires/52 stages : limites physiques respectées, états continus, poids inchangés et 4 670 anciens paquets rejetés. Les quatre positions initiales sont partagées entre les séquences/contrôleurs ; aucune garantie universelle n'est affirmée.
- Le freinage n'impose pas un arrêt artificiel : jusqu'à 0,268 m/s au début de l'affectation, mais aucun changement d'arête observé pendant sa diffusion. Les limites d'infaisabilité du filtre restent dans les rapports.

Décision : `docs/decisions/24-transitions-continues.md`.
Preuves : `tasks/evidence/008e-transitions-{pilot,validation,audit}.json`.
Configurations : `config/formation-transitions.yaml` et `formation-transitions-validation.yaml`.

Les sections suivantes conservent le périmètre ayant guidé l'implémentation.

## Base retenue

Comparer le professeur classique et le prototype `LocalTargetGAT` protégé, avec politiques, filtres et modules d'affectation explicitement identifiés. Le brut reste disponible comme comparaison expérimentale : il n'est pas validé pour la sécurité.

## Travail attendu

1. Définir une séquence de missions du catalogue avec instants de changement explicites.
2. Préserver positions, vitesses et temps entre missions ; ne pas concaténer des épisodes qui remettent artificiellement les vitesses à zéro.
3. Réaffecter localement les nouvelles places en gérant les versions de mission et les anciens messages.
4. Compter rendez-vous éventuel, freinage, communication, déplacement et contacts pendant toute la transition.
5. Exporter des trajectoires continues et les métriques par transition ainsi que globales ; ne pas compter deux fois un contact prolongé à une frontière de mission.
6. Tester des lettres, nombres et figures, y compris retours à une forme précédente, délais trop courts et configurations infaisables.
7. Évaluer sur des séquences et départs nouveaux, sans entraîner ni régler les poids sur le test déjà examiné en 008d.

Le prototype actuel est limité au domaine étudié : 12 robots, portée 3 m et formations de cadre 6 m. La navigation avec obstacles demeure le bloc 009 après les transitions.
