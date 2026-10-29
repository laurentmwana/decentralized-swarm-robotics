# 24 — Transitions continues et versions de mission

Date : 2026-10-29. Tâche 008e, autorisée par « vas-y ».

## État physique et calendrier

Chaque `FormationStage` fournit une consigne du catalogue et une durée entière en pas physiques. `execute_sequence` transmet la position **et la vitesse finales** d'un segment au suivant. Le moteur physique étant sans état caché, cette transmission compose une intégration continue. Aucun robot n'est téléporté, aucune vitesse n'est remise à zéro par le changement de mission.

Les observations des contrôleurs reçoivent le temps absolu de la séquence. L'archive globale contient un seul axe temporel, sans état physique ni intervalle dupliqué aux frontières. Les états internes d'affectation sont en revanche réinitialisés à la nouvelle mission. À une frontière, la ligne de politique globale décrit la mission qui va fournir la prochaine commande ; les métriques du segment précédent gardent son état interne juste avant cette réinitialisation.

Les durées CLI doivent être des multiples de dt. Une mission trop courte pour achever le rendez-vous ou l'affectation est exécutée comme telle et conservée comme échec, avec stabilisation null. Les effectifs variables en cours de séquence ne sont pas supportés et sont refusés avant simulation. Les configurations géométriques infaisables sont également refusées avant exécution.

## Versions d'affectation et messages anciens

L'empreinte de revendication combine la géométrie des places et la version d'occurrence `stage:i`. A→B→A produit ainsi trois versions distinctes, même lorsque le troisième tracé est identique au premier. Les nouvelles préférences sont calculées depuis la position propre actuelle au démarrage de l'affectation.

`advance_current_mission` ignore et compte les paquets d'une autre version avant toute fusion de revendications. Les contrôles existants sur les paquets de la version courante demeurent stricts. Un ancien message avec une révision élevée ne peut donc pas réintroduire une ancienne place dans le nouvel état.

La campagne ajoute un contrôle de rejeu explicite : au premier tour d'affectation de la nouvelle mission, chaque récepteur reçoit, de ses voisins alors visibles, le nouveau paquet et au plus une photographie précédente mise en cache. Les paquets anciens rejetés sont inclus dans le trafic mesuré. Cette injection éprouve le rejet des anciennes versions ; elle n'est pas un modèle complet de latence, de perte ou de bande passante radio. Le transport courant reste synchrone à un saut par tour.

## Métriques de transition et collisions

La couverture des places, le seuil de vitesse et la durée de maintien restent ceux des formations statiques. La stabilisation est donnée relativement au début de chaque mission et sur l'axe temporel absolu. Tous les délais (rendez-vous 8 s, freinage 1 s, diffusion 14,4 s à 12 robots, mouvement) restent dans les 40 s accordées à chaque mission.

`collision_timeline` fournit les débuts de contact par intervalle et les contacts initiaux. Le bilan global compte une seule fois chaque entrée de contact, même si ce contact dure sur plusieurs missions. Chaque stage indique séparément `contact_present` : un contact déjà commencé à sa frontière invalide sa réussite même si son nombre de nouveaux événements vaut zéro. La somme des événements attribués aux stages correspond au total global ; aucun temps de stabilisation n'est imputé à un échec.

Une séquence est réussie seulement si toutes ses missions sont réussies et si elle ne présente aucun contact. Le taux par stage et le taux par séquence sont rapportés distinctement.

## Comparaison sans réentraînement

Poids figés de 008d : `a1e3b88b39386a3d490798b5c7717e3939416e52572a058045d7402ec6cf6d84`. Aucun poids ni hyperparamètre n'est changé pour les transitions. Le professeur est comparé au GAT **protégé par le filtre classique**, dont les interventions et infaisabilités restent comptées. Le résultat de sécurité insuffisante du GAT brut en 008d n'est pas effacé.

Pilote, départs dispersés 7400 et 7401, 12 robots, portée 3 m, cadre 6 m :

- A→B→A ;
- 1→10→3 ;
- carré→rectangle→cercle.

Chaque contrôleur réussit les six séquences (18 stages), sans collision observée. Le GAT protégé a une erreur finale moyenne de 0,014143 m et une stabilisation moyenne de 30,561 s par stage, contre ≈0 m et 29,439 s pour le professeur. Les paquets anciens rejetés sont respectivement 1 532 et 1 584 ; les infaisabilités du filtre sont 87 et 151.

Validation à paramètres figés : C→7→triangle→C depuis deux nouveaux départs 7402 et 7403. Chaque contrôleur réussit deux séquences (huit stages) sans collision observée. Le GAT protégé atteint une erreur finale moyenne de 0,011706 m et une stabilisation moyenne de 30,563 s par stage, contre ≈0 m et 29,613 s pour le professeur. Les infaisabilités du filtre sont 21 et 48, conservées dans les rapports.

Ces campagnes utilisent quatre positions initiales distinctes, partagées entre les séquences et contrôleurs. Elles ne constituent pas 52 départs indépendants ni une garantie sur toutes les combinaisons possibles du catalogue.

## Audit et limites

L'audit des 16 trajectoires (52 stages, deux contrôleurs confondus) confirme :

- aucune violation des limites de vitesse, accélération, confinement ou déplacement par pas à la tolérance numérique 10⁻⁹ ;
- un seul axe temporel et conservation exacte des états physiques aux frontières ;
- aucune incohérence entre le total global et la somme des événements de contact par stage ;
- 4 670 paquets anciens effectivement rejetés ;
- empreinte des poids inchangée avant/après toutes les campagnes.

La durée prévue de freinage ne prouve pas à elle seule l'arrêt de tous les robots : une vitesse maximale de 0,268 m/s est observée à l'entrée de la phase d'affectation dans cette campagne. Aucune vitesse n'a été forcée à zéro pour masquer cela. Les graphes d'adjacence restent néanmoins inchangés pendant les tours d'affectation observés, les préférences sont figées et les consensus finaux sont vérifiés hors contrôleur. Pour des réseaux dont la topologie varie pendant cette phase, la borne de diffusion du cas statique ne constitue pas une garantie ; les échecs devront être mesurés.

Les changements imposés pendant un mouvement et les délais trop courts sont couverts par les tests, mais les campagnes nominales donnent 40 s à chaque formation. D'autres orientations, tailles, pertes de messages, obstacles et changements rapides restent à évaluer. Aucun obstacle ni navigation collective vers une destination ne sont encore ajoutés ici.

## Vérification et preuves

401 tests ciblés réussis : physique, observations, métriques cercle, catalogue, affectation, contrôle, reconnexion, données, GAT et transitions. Les tests incluent une vitesse non nulle transmise au segment suivant, les temps absolus, A→B→A, les paquets anciens à révision élevée, une collision prolongée sur une frontière, les délais plus courts que le maintien et l'export autonome sans pickle.

Preuves : `tasks/evidence/008e-transitions-pilot.json`, `008e-transitions-validation.json`, `008e-transitions-audit.json`. Trajectoires : `outputs/formations/transitions-pilot-task008e/` et `transitions-validation-task008e/`.

## Commandes

```powershell
python -m src.reference.evaluate_transitions --config config/formation-transitions.yaml --model-run outputs/imitation/training-multiform-expanded-v1 --output outputs/formations/transitions-pilot-task008e
python -m src.reference.evaluate_transitions --config config/formation-transitions-validation.yaml --model-run outputs/imitation/training-multiform-expanded-v1 --output outputs/formations/transitions-validation-task008e
```

Les sorties existantes sont refusées. Sans `--model-run`, la campagne utilise uniquement le professeur. `--modes teacher guarded raw` ajoute explicitement le mode brut expérimental. Les chemins Python et dépendances Windows documentés précédemment restent nécessaires si le `.venv` transféré est inutilisable.
