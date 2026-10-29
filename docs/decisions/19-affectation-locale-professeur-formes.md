# 19 — Affectation locale par revendications et professeur multi-formes

Date : 2026-10-29. Tâche 008, après le catalogue de 56 formes.

## Contrat d'information explicite

Les observations physiques initiales ne contiennent ni identifiants des voisins ni messages. Une affectation avec conflits ne peut donc pas être présentée comme déjà fournie par le simulateur. Le nouveau protocole possède un agent par robot, un identifiant unique connu dans 0…N−1, une géométrie de places commune et la position propre initiale. Les identifiants servent uniquement au départage déterministe ; leur disponibilité est une nouvelle hypothèse explicite.

Chaque agent classe les places selon sa distance initiale propre. Il diffuse des revendications `(propriétaire, révision, place)`, dans un paquet contenant son identifiant et l'empreinte de la géométrie. Les paquets ne contiennent ni position distante ni coût d'un autre robot. La table des revendications est relayée de proche en proche ; une révision plus ancienne ne remplace jamais une nouvelle. Deux contenus pour la même révision, une mission incompatible, un identifiant hors consigne et une révision propre inventée par un relais sont refusés.

En cas de conflit, le plus petit identifiant garde sa place ; l'autre choisit sa meilleure place libre parmi les revendications connues de priorité supérieure. Ce mécanisme ne minimise pas le déplacement total et peut défavoriser les identifiants élevés.

## Tours de communication et portée de la résolution

Le transport prend une photographie de tous les paquets avant tout avancement d'agent : un message ne traverse qu'une arête par tour. Aucune condition globale n'est fournie aux agents pour choisir leur place ou arrêter les échanges.

Le protocole initial utilise un graphe **statique connecté**, des messages fiables synchrones et des robots stationnaires. Par induction sur la priorité, la revendication du premier robot devient fixe puis se propage ; celle du suivant se fixe après réception des priorités précédentes, etc. Avec au plus N−1 arêtes de diamètre, N² tours constituent un budget conservateur de fixation et de propagation des N revendications. Les tests couvrent chaînes, anneaux et graphes complets, y compris les conflits initiaux sur les 56 formes. Un graphe déconnecté n'est pas déclaré résolu.

`locally_consistent` est un état de connaissance locale ; il ne constitue pas une détection générale de consensus sur un réseau mobile. La vérification globale finale du simulateur (places distinctes, mêmes revendications et connaissance complète) est une métrique hors contrôleur. Elle n'est pas utilisée pour guider les agents.

La table nécessite jusqu'à N revendications par robot et par paquet. Les tours ne sont pas gratuits : à 12 robots, le budget de 144 tours à 0,1 s consomme 14,4 s. À 24 robots il consommerait 57,6 s ; un horizon trop court est refusé explicitement. Ce protocole est une base fonctionnelle, pas une solution de communication optimisée ou tolérante aux pertes.

## Raccordement au déplacement

`execute_shape` représente réellement les tours stationnaires dans les trajectoires physiques, à vitesse nulle. Ensuite les revendications sont gelées pour cette mission. Chaque robot disposant d'une table locale complète sans conflit rejoint sa place ; les autres restent immobiles et l'échec est rapporté.

La commande nominale combine attraction vers la place propre, amortissement de la vitesse et répulsion des seuls voisins directs. Les accélérations et vitesses respectent les limites du moteur. Les agents ne reçoivent jamais les positions des robots hors portée pour calculer leurs commandes. Le résultat enregistre positions/vitesses, accélérations demandées/bornées, communications, places, historique des revendications et tailles de connaissance.

L'API originale `AccelerationEpisode` reste inchangée ; le nouveau parcours explicite est distinct. Une future imitation pourra partager le même module classique d'affectation, puis apprendre le mouvement avec la place attribuée comme entrée locale. Les poids du cercle et son seul contrat centre/rayon ne suffisent pas.

## Métriques et pilote

L'erreur de forme est le maximum des deux distances de couverture : robot vers place la plus proche et place vers robot le plus proche. Elle pénalise les places vides et ne dépend pas d'une permutation d'identités. La réussite exige simultanément consensus d'affectation, erreur ≤0,1 m, vitesse maximale ≤0,05 m/s pendant 1 s, et aucune collision pendant l'épisode. Tout contact initial ou intermédiaire reste un échec ; son temps de stabilisation est null.

Nouveau protocole multi-formes : 40 s, incluant les 14,4 s d'affectation ; 12 robots, cadre de construction 6 m (rectangle 6×4 m), portée 3 m. Ces changements de scénario sont explicites : ne pas comparer directement les temps avec les campagnes cercle de 10 s/portée 2 m.

Le premier pilote utilise une même configuration en anneau de séparation 0,45 m, rotation déterministe par graine 7000, pour les 56 formes. Il résout les 56 affectations et finit sur les places, mais réussit seulement 37/56 épisodes (66,07 %) en raison de 30 événements de collision. Rapport conservé : `tasks/evidence/008-shape-teacher-pilot.json`.

Ce pilote confirme le raccordement fonctionnel et expose le défaut de sécurité ; il ne valide pas une collecte d'imitation.
