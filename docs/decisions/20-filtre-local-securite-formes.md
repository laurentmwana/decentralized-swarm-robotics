# 20 — Filtre d'accélération local et validation des formes

Date : 2026-10-29. Correction du défaut de collision révélé par le pilote 008.

## Filtre et limites

Avec `r = position_propre − position_voisin`, `w = vitesse_propre − vitesse_voisin`, `h = ||r||² − distance_sécurité²`, imposer pour chaque voisin :

`r · a_propre ≥ −0,5 ||w||² − α r·w − 0,25 α² h`.

Lorsque les deux robots satisfont leur moitié de la contrainte, leur différence d'accélération satisfait la condition réciproque de barrière d'ordre deux `h'' + 2α h' + α² h ≥ 0` dans le modèle continu correspondant. Cette relation algébrique ne prouve pas une garantie dans le simulateur discret avec vitesse bornée, murs et contraintes parfois infaisables.

Projeter la commande nominale sur l'intersection des demi-plans locaux et du disque d'accélération maximale. En 2D, la projection s'obtient en énumérant les projections sur les frontières et leurs intersections, sans dépendance à un solveur QP externe. Une préférence de détour de même chiralité chez les deux robots départage les face-à-face. Le calcul est invariant à l'ordre des voisins.

Si l'intersection est vide, appliquer un freinage borné et enregistrer explicitement une infaisabilité. Une superposition exacte est également signalée. Ce repli n'est pas annoncé comme sûr. La portée doit couvrir la distance de détour.

Paramètres figés avant le pilote filtré : α=3, distance de sécurité 0,4 m, distance de détour 1 m, gain de détour 3. Attraction 6, amortissement 4 et répulsion nominale 60 sont conservés. Le filtre est opt-in `--safety-filter` ; les rapports du professeur nominal restent inchangés.

## Comparaison appariée du pilote

Même graine 7000, même anneau initial et mêmes 56 formes, limites et horizon de la décision 19 :

| Mesure | Nominal | Filtré |
|---|---:|---:|
| Affectations résolues | 56/56 | 56/56 |
| Réussites | 37/56 (66,07 %) | 47/56 (83,93 %) |
| Événements de collision | 30 | 0 |
| Erreur finale moyenne | ≈0 m | 0,022076 m |
| Stabilisation moyenne des seules réussites | 19,289 s | 20,285 s |

Le filtre rencontre 10 infaisabilités robot/pas, conservées dans les traces et les métriques. Neuf formes ne se stabilisent pas dans le délai. La faible erreur finale du nominal ne masque pas ses contacts ; l'absence de contact observée du filtré ne masque ni ses échecs ni ses infaisabilités.

## Validation sans réglage sur nouveaux départs dispersés

Conserver les paramètres du filtre et utiliser les graines 7001 et 7002, chacune partagée entre les 56 formes. Positions tirées par rejet dans le disque de rayon 4 m, séparation minimale 0,45 m, budget 20 000 tirages. Aucune modification du professeur après cette validation.

- 112 épisodes demandés et évalués ; 47 réussites (41,96 %), aucune collision observée.
- Graine 7001 : zéro affectation résolue, zéro réussite ; le réseau initial est déconnecté et les agents restent stationnaires.
- Graine 7002 : 56 affectations résolues, 47 réussites ; quatre infaisabilités robot/pas du filtre.
- Erreur finale moyenne de l'ensemble, incluant le départ non résolu : 1,333928 m. Stabilisation moyenne des seules réussites : 21,498 s.

Ces 112 épisodes utilisent seulement deux configurations initiales distinctes, pas 112 tirages indépendants. Le résultat révèle un problème de reconnexion, pas un motif pour éliminer la graine 7001. Les neuf échecs restants de la graine connectée doivent aussi être examinés.

Preuves : `tasks/evidence/008-shape-teacher-safety-pilot.json`, `008-shape-teacher-safety-validation.json` et `008-connectivity-audit.json`. Les trajectoires et revendications sont conservées dans les dossiers correspondants de `outputs/formations/`.

## Vérification et suite

303 tests ciblés physique/épisodes/catalogue/affectation/contrôle/sécurité réussis, puis un test supplémentaire de reproductibilité du domaine dispersé ajouté et réussi (304 cas ciblés au total). Aucun entraînement multi-formes lancé.

L'affectation stationnaire et le professeur constituent une base fonctionnelle ; le jalon de robustesse depuis des positions dispersées n'est pas atteint. La prochaine étape doit traiter la reconnexion par des actions et messages locaux, compter son temps et ses collisions, puis réévaluer sans supprimer les départs difficiles. La collecte d'imitation attend ce travail ainsi que le diagnostic des stabilisations échouées.

## Commandes

```powershell
python -m src.reference.evaluate_shapes --output outputs/formations/teacher-pilot-task008 --seeds 7000 --robots 12 --width 6 --horizon 400 --communication-radius 3
python -m src.reference.evaluate_shapes --output outputs/formations/teacher-safety-pilot-task008 --seeds 7000 --robots 12 --width 6 --horizon 400 --communication-radius 3 --safety-filter
python -m src.reference.evaluate_shapes --output outputs/formations/teacher-safety-validation-task008 --seeds 7001 7002 --robots 12 --width 6 --horizon 400 --communication-radius 3 --safety-filter --initialization dispersed
```

Choisir des sorties neuves pour reproduire. `--shapes letters_A numbers_10 geometries_square` sélectionne un sous-ensemble explicite ; aucune forme n'est entraînée par cette commande.
