# 21 — Rendez-vous local, freinage et correction des détours

Date : 2026-10-29. Tâche 008b, autorisée par « vas-y » après le constat du robot isolé et des neuf échecs de stabilisation.

## Diagnostic préalable

Le départ 7001 comporte deux composantes (11 robots et 1 isolé), ce qui empêche l'affectation stationnaire complète. Sur 7002, les neuf échecs sont A, L, V, W, Y, 1, étoile, ligne et V géométrique. Plusieurs erreurs de position sont déjà inférieures à 0,1 m mais les vitesses restent au-dessus de 0,05 m/s. Les robots continuent à recevoir une composante de détour même quand le filtre n'en a plus besoin. Rapport conservé : `tasks/evidence/008b-stabilization-diagnostic.json`.

## Correction isolée des détours

Conserver la version antérieure par défaut et ajouter `--risk-detours` (`ShapeGains.detour_only_on_constraint=True`). Le détour n'est actif que si la commande nominale viole effectivement la contrainte de barrière du voisin, en plus des conditions antérieures de distance et de direction. Le test de risque utilise la même commande nominale initiale pour tous les voisins, afin de préserver l'invariance à leur ordre.

Sur les mêmes départs 7001–7002 et les mêmes 56 formes, sans rendez-vous : 56/112 réussites, zéro collision. Les 56 formes de 7002 réussissent : les neuf défauts antérieurs sont corrigés. Les 56 épisodes du départ déconnecté restent des échecs d'affectation. Les seuils et l'horizon sont inchangés.

## Phase de rendez-vous locale

Le nouveau module `ReconnectionConfig` définit une consigne intermédiaire commune : un anneau centré à l'origine, avec espacement de corde 0,6 m. Chaque robot calcule sa place intermédiaire depuis son seul identifiant et l'effectif partagé. Aucune connaissance des composantes du réseau, des positions distantes ou d'un centroïde de l'essaim n'est utilisée dans ses commandes.

Tous les robots suivent le même calendrier explicite, même si le départ était déjà connecté :

1. 8 s pour rejoindre les places intermédiaires, avec observations locales et filtre de sécurité.
2. 1 s de freinage local borné, visant `a = −v/dt` lorsque cette commande est admissible. Aucune position ou vitesse n'est réinitialisée artificiellement.
3. Initialisation de nouvelles revendications depuis la position propre **à cet instant**, puis N² tours de diffusion d'affectation. Le freinage reste actif ; les anciens états de revendication ne sont pas réutilisés.
4. Déplacement vers les places de la forme, si l'état local d'affectation est complet et sans conflit.

La synchronisation du calendrier est une hypothèse partagée du simulateur ; la tolérance à une dérive d'horloge, aux pertes de messages ou aux fautes d'identifiant n'est pas implémentée. La reconnexion est opt-in `--reconnect` et exige `--safety-filter`. L'anneau intermédiaire doit tenir dans l'arène, respecter la séparation et être compatible avec la portée.

## Temps et vérification de l'hypothèse stationnaire

À 12 robots et dt=0,1 s : 80 pas de rendez-vous, 10 de freinage et 144 de diffusion. La commande vers la forme commence à 23,4 s, laissant 16,6 s sur le même horizon total de 40 s. Le seuil d'erreur reste 0,1 m, celui de vitesse 0,05 m/s et le maintien 1 s. La simulation compte les collisions de toutes les phases.

Les traces enregistrent les phases, les positions, vitesses, commandes, revendications, messages et infaisabilités du filtre. Les métriques auditent la vitesse maximale et les changements d'arêtes pendant l'affectation. Aucun verdict global de connectivité ou de stationnarité n'est envoyé aux contrôleurs. La borne de diffusion du graphe statique n'est donc pas supposée gratuitement après un déplacement : sa condition doit être vérifiée sur les trajectoires. Une réussite empirique de consensus sur un graphe mobile ne constituerait pas une preuve de la même borne.

Les matrices de revendications portent la valeur sentinelle −1 avant l'initialisation de l'affectation et ne doivent pas être interprétées comme des places valides pendant le rendez-vous.

## Comparaison sur les départs conservés

| Variante, 56 formes × deux départs | Réussites | Collisions |
|---|---:|---:|
| Professeur filtré initial | 47/112 | 0 |
| Détours conditionnés, sans rendez-vous | 56/112 | 0 |
| Détours conditionnés et rendez-vous | 112/112 | 0 |

La version complète atteint une erreur finale moyenne de 8,7×10⁻¹¹ m et une stabilisation moyenne de 29,35 s (maximum 34,4 s). Les 112 affectations sont résolues, avec vitesse nulle et aucun changement d'arête pendant la diffusion. Le temps supplémentaire du rendez-vous est inclus : cette version n'est pas présentée comme plus rapide.

Le filtre recourt au freinage de repli pour 784 infaisabilités robot/pas dans l'ensemble de la campagne. Les phases précédant l'affectation étant identiques pour les 56 formes d'un même départ, cet agrégat ne représente pas 784 situations initiales indépendantes. Les traces restent disponibles ; l'absence de collision observée n'est pas une garantie universelle de sécurité.

Preuves : `tasks/evidence/008b-risk-detours-evaluation.json` et `008b-reconnection-evaluation.json`. Les résultats historiques 008 restent inchangés.

## Vérification

314 tests ciblés réussis (physique, épisodes, catalogue, affectation, professeur, filtre et reconnexion). Ils incluent une régression du robot isolé de la graine 7001, l'absence de téléportation, les bornes physiques, la latence réelle, la stationnarité de l'affectation et l'absence de détour perturbateur près d'une place déjà accessible.

La validation sur les nouvelles graines 7003–7005 est terminée avec tous les paramètres figés. Aucun entraînement neuronal multi-formes n'est lancé dans cette tâche.

## Résultat de validation indépendante

Les 168 épisodes (56 formes × trois nouveaux départs) réussissent, sans collision observée. L'erreur finale moyenne est 5,51×10⁻⁹ m ; la stabilisation moyenne est 29,396 s, toutes phases incluses. Les 168 affectations sont résolues. L'audit de toutes les trajectoires ne détecte aucune violation des bornes physiques à la tolérance numérique 10⁻⁹, aucune variation de graphe et aucune vitesse non nulle pendant l'affectation.

Le filtre rencontre 3 815 infaisabilités robot/pas : 3 808 pendant le rendez-vous et 7 pendant le mouvement final ; aucune pendant le freinage ou l'affectation. Le freinage de repli a été appliqué et toutes ces situations restent dans les traces. Les scénarios de rendez-vous sont communs aux 56 formes d'un même départ : ces agrégats ne constituent pas autant de situations indépendantes. La campagne couvre trois positions initiales distinctes, pas 168 tirages indépendants.

Les défauts reproduits de 008b sont corrigés sur ce domaine. D'autres effectifs, portées, vitesses initiales, pertes de messages et dimensions de formes restent à évaluer. La collecte peut être préparée avec ces paramètres explicites et son propre contrôle de qualité ; aucune garantie universelle de sûreté ou de réussite n'est déclarée.

Preuves : `tasks/evidence/008b-reconnection-validation.json` et `008b-reconnection-validation-audit.json`. Les rapports et trajectoires antérieurs restent conservés.

## Commandes reproductibles

```powershell
python -m src.reference.evaluate_shapes --output outputs/formations/teacher-risk-detours-task008b --seeds 7001 7002 --robots 12 --width 6 --horizon 400 --communication-radius 3 --safety-filter --risk-detours --initialization dispersed
python -m src.reference.evaluate_shapes --output outputs/formations/teacher-reconnection-task008b --seeds 7001 7002 --robots 12 --width 6 --horizon 400 --communication-radius 3 --safety-filter --risk-detours --reconnect --initialization dispersed
python -m src.reference.evaluate_shapes --output outputs/formations/teacher-reconnection-validation-task008b --seeds 7003 7004 7005 --robots 12 --width 6 --horizon 400 --communication-radius 3 --safety-filter --risk-detours --reconnect --initialization dispersed
```

Utiliser des dossiers neufs pour reproduire. Sous Windows, les variables et le Python du runtime documentés en 007b restent nécessaires si le `.venv` transféré est inutilisable.
