# 25 — Transport de formation et obstacles locaux

Date : 2026-10-29. Tâche 009, poursuite autorisée par « d’accord, continue ».

## Contrat de commande

Une formation du catalogue est préparée pendant 40 s par le protocole existant. Sa position et sa vitesse finales sont transmises à la navigation, sans téléportation ni remise à zéro. Chaque robot conserve la place issue de ses propres revendications locales ; il ne suit la trajectoire que si sa photographie locale contient toutes les places distinctes. Le centre de référence commun suit des tronçons à interpolation quintique, avec vitesse et accélération nulles aux points de passage. Les durées et points sont fournis par la mission, sans planificateur global automatique.

Le contrôleur utilise explicitement un repère en translation : position propre et cible moins la position de référence, vitesse propre moins la vitesse de référence. Les positions et vitesses relatives des voisins restent identiques. L’accélération commune de référence est ajoutée à la commande, puis le filtre classique travaille dans le repère physique. Cette adaptation fournit une information de mouvement commune supplémentaire ; elle n’est pas un apprentissage de la navigation. Même une sortie neuronale nulle peut déplacer l’essaim par cette anticipation explicite.

Le professeur utilise le même repère et la même anticipation. Le GAT n’est pas réentraîné : poids SHA-256 `a1e3b88b39386a3d490798b5c7717e3939416e52572a058045d7402ec6cf6d84`. Le prototype reste hybride ; la sécurité du GAT brut n’est pas réhabilitée par ces essais.

## Obstacles et observation

Les obstacles sont des disques statiques présents dès la préparation. Chaque capteur reçoit uniquement la position propre ; il retourne les centres relatifs des disques dont la surface est à moins de 3 m. Le rayon transmis au filtre vaut rayon obstacle + rayon robot + marge de 0,15 m. Les obstacles lointains et les objectifs privés des autres robots ne sont jamais fournis au contrôleur neuronal.

Les contraintes entre robots restent réciproques ; face à un disque immobile, le robot mobile assume la contrainte entière. Les contraintes robot/robot et robot/obstacle sont projetées simultanément sous la borne d’accélération. Les replis en cas d’infaisabilité et les interventions sont comptés. Le filtre utilise la vitesse réelle du robot pour les obstacles statiques. Aucune garantie universelle de sécurité ne découle du filtre discret.

Le générateur de départs connaît la scène et rejette les contacts initiaux ; cette connaissance appartient au dispositif d’expérience. Les deux contrôleurs reçoivent exactement les mêmes départs. La route de contournement est une consigne explicite, et non une preuve de recherche autonome de chemin par le GAT.

## Protocole enregistré avant les essais

12 robots, formations de largeur 6 m, portée de communication 3 m, arène 10 m, dt 0,1 s, vitesse maximale 1 m/s et accélération maximale 2 m/s². La formation initiale est préparée pendant 40 s. Les tronçons ont des durées de 10 à 20 s ; une période finale permet l’arrêt.

Les métriques distinguent :

- l’erreur de forme après recentrage sur le centre effectivement observé ;
- l’erreur de suivi des places autour du centre de référence ;
- la distance du centre réel à la destination ;
- les contacts physiques robot/robot, robot/disque et murs sur toute la simulation.

La réussite impose une préparation réussie, aucun contact, une erreur de forme maximale pendant le transport ≤ 0,20 m, puis une erreur finale ≤ 0,10 m, une distance à destination ≤ 0,10 m et une vitesse de chaque robot ≤ 0,05 m/s, maintenues une seconde après la fin de la référence. Une formation correcte restée au départ échoue. Un dépassement transitoire reste un échec même si la forme se reconstitue ensuite. Le temps d’arrivée est absolu, préparation comprise ; il reste null en cas d’échec.

Les contacts avec les disques sont calculés sur les segments parcourus entre états, avec un seul événement à l’entrée d’un contact prolongé. La distance minimale aux obstacles est également calculée sur les segments dans l’audit. Le premier pilote a enregistré cette distance sur les seuls états ; ses contacts continus sont corrects et l’audit fournit la distance continue de référence. Les murs sont comptés lorsque le moteur atteint sa limite physique.

## Campagnes et limites

Le pilote utilise les graines 7600 et 7601 : carré en ligne droite, A sur deux tronçons, carré sur un contournement fourni, carré sur une route directe rencontrant le disque, puis carré avec une place finale obstruée. Chaque contrôleur réussit les six épisodes nominaux et échoue aux quatre épisodes difficiles, sans collision observée. Au total, la réussite est donc 6/10 par contrôleur, et non 100 %. L’erreur maximale de forme nominale du GAT protégé est 0,010056 m. Les cas difficiles atteignent 0,873909 m pour le GAT protégé et 0,650813 m pour le professeur.

La validation conserve les paramètres et les poids : nouvelles graines 7602 et 7603, rectangle, chiffre 7, nouveaux points de passage et disque décalé. Chaque contrôleur réussit encore 6/6 épisodes nominaux et 0/4 difficiles, soit 6/10 au total, sans collision observée. L’erreur maximale nominale du GAT protégé est 0,010056 m, et sa distance finale maximale à destination est 0,006918 m. Dans les cas difficiles, l’erreur de forme atteint 0,824904 m pour le GAT protégé et 0,970680 m pour le professeur. Aucun échec n’est supprimé et aucun poids n’est ajusté après examen.

L’audit indépendant des 40 trajectoires ne trouve aucune incohérence physique, temporelle, de limites ni de contacts. Il tient compte de l’écrêtage réel de vitesse du moteur. Les minima de distance physique aux disques sur les segments sont 0,111189 m au pilote et 0,113653 m en validation ; la marge de sécurité demandée de 0,15 m n’est donc pas garantie à chaque instant, même sans contact physique. L’empreinte du modèle reste identique. Ces campagnes emploient quatre graines avec des scènes parfois différentes ; les 40 trajectoires ne représentent pas 40 départs indépendants.

Preuves conservées : `tasks/evidence/009-navigation-pilot.json`, `009-navigation-validation.json` et `009-navigation-audit.json`. Les interventions et infaisabilités du filtre restent disponibles pour chaque épisode.

Cette étape évalue un transport avec référence commune, obstacles statiques et filtrage local. Elle ne résout pas la planification autonome, les passages étroits arbitraires, les obstacles mobiles, les pertes de communication ni toutes les tailles et formes du catalogue.

## Vérification et reproduction

412 tests ciblés réussis, incluant les 401 régressions antérieures. Les nouveaux cas couvrent le capteur local, les contacts traversants et prolongés, les contraintes statiques, le repère mobile, les frontières physiques, le faux succès d’une formation immobile et les délais trop courts.

```powershell
python -m src.reference.evaluate_navigation --config config/formation-navigation.yaml --model-run outputs/imitation/training-multiform-expanded-v1 --output outputs/navigation/task009-pilot-v1
python -m src.reference.evaluate_navigation --config config/formation-navigation-validation.yaml --model-run outputs/imitation/training-multiform-expanded-v1 --output outputs/navigation/task009-validation-v1
python -m src.reference.audit_navigation outputs/navigation/task009-pilot-v1 outputs/navigation/task009-validation-v1 --checkpoint outputs/imitation/training-multiform-expanded-v1/best.pt --output tasks/evidence/009-navigation-audit.json
```

Les sorties de simulation existantes sont refusées. Sans `--model-run`, le professeur seul est évalué. Les archives NPZ contiennent les états continus, consignes, références, places, observations d’obstacles comptées et interventions du filtre ; les configurations et empreintes sont enregistrées avant chaque campagne.
