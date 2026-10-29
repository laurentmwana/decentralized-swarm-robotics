# 015 — Corriger les trajets avec obstacles et expliquer les métriques

Statut : terminé. Autorisation utilisateur du 29 octobre 2026 : corriger la navigation, actualiser tâches/décisions puis créer des commits.

## Problème et correction

La référence déplaçait le centre sans vérifier les trajectoires des places de la formation. Un filtre local évitait les contacts mais pouvait déformer durablement le carré. Le cas « destination obstruée » occupait une place finale : son objectif était incompatible avec l'obstacle.

- Planification classique sur la carte statique déclarée de la scène : obstacles translatés par chaque place, recherche sur grille, contrôle des segments complets, murs et destination.
- Même destination, même formation et mêmes seuils ; détour commun à vitesse maximale de référence 0,20 m/s, marge géométrique de planification 0,35 m, délai ajusté à la longueur du trajet et 12 s de stabilisation.
- Destination incompatible refusée avant simulation avec explication ; aucun déplacement silencieux de l'objectif.
- Le GAT conserve ses poids et observations locales. Le planificateur utilise explicitement la carte commune ; ce supplément n'est pas un apprentissage d'obstacles par le réseau.
- Interface : trois commandes et deux comparaisons regroupées, distance à destination/déformation maximale visibles, stabilisation « Non atteinte », campagnes antérieures identifiées comme archives.

## Vérifications

Régression carré, 12 robots, départ 7700, obstacle sur la route : classique et GAT protégé réussissent, aucun contact. GAT : erreur finale 0,010015 m, destination 0,005550 m, déformation maximale 0,010061 m, arrivée stabilisée à 67,6 s.

Suite complète : 600 tests réussis. Tests spécifiques : garde géométrique de chaque place sur les segments, vitesse de référence, destination bloquée, impossibilité de contourner en traversant les murs. Vérification de l'interface réelle et export de la trajectoire réussie.

Décision : [31-navigation-planifiee](../docs/decisions/31-navigation-planifiee.md).
Suite : 016, campagne sur nouveaux départs et audit indépendant des états et des métriques. La grille finie et les marges conservatrices peuvent refuser un trajet réalisable par rotation/déformation ; aucune garantie générale sur toutes les scènes.
