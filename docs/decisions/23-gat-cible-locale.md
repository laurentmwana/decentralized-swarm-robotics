# 23 — GAT à cible locale : architecture, premier essai et extension du domaine

Date : 2026-10-29. Tâche 008d ; entraînement explicitement demandé par « vas-y next ».

## Modèle et frontière d'information

`LocalTargetGAT` est un type de modèle distinct du GAT cercle. Il reçoit une observation propre à huit composantes du schéma v2, les quatre composantes physiques relatives de chaque voisin direct et un masque. Aucun argument ne contient une cible privée voisine.

Encoder en deux couches denses la cible relative propre, la vitesse propre, la phase et l'effectif normalisés. Encoder séparément les positions/vitesses relatives et la distance dérivée des arêtes. Une attention par produit scalaire entre requête propre et clé relative agrège ces messages en un seul échange. Le décodeur dense combine l'encodage propre, le message, la cible/vitesse relatives et le degré local. Les poids sont partagés ; une sortie à deux composantes est bornée en norme par l'accélération maximale.

La normalisation utilise les échelles fixées dans `config/local-target-model.yaml`, sans ajustement sur test. Les voisins masqués n'influencent pas le résultat ni ses dérivées, même si leur remplissage contient des NaN. Un robot isolé reçoit un message nul. Le checkpoint enregistre le type, sa version, le schéma v2, les noms de caractéristiques, les échelles et les limites physiques. Les sauvegardes cercle sont refusées.

## Apprentissage et sélection

`local_target_train` charge uniquement train/validation, conserve les phases de mouvement 0/3 et exclut les épisodes `duplicate_within_split_of` de l'optimisation et de la sélection. Les lots sont des exemples robot/pas à voisinage relatif masqué. Mélanger ces exemples ne change pas la séparation des simulations et formes établie avant l'apprentissage.

La MSE pondère les composantes de tous les exemples. Adam, arrêt anticipé et meilleur checkpoint utilisent uniquement validation. Le run conserve les archives sélectionnées, l'empreinte du manifeste, les paramètres, les versions et les pertes. Le mode à blanc ne crée ni optimiseur ni sortie.

## Rejeu apparié : brut et protégé

Le simulateur accepte une politique de mouvement locale. Elle remplace les commandes du professeur dans les phases 0 et 3 et ne reçoit que l'observation propre/relative, la cible propre, la phase et l'effectif. Les mêmes consignes de rendez-vous et le même module classique d'affectation sont utilisés. Le freinage et la diffusion des phases 1/2 restent classiques pour toutes les variantes.

- `raw` : le mouvement du GAT n'est pas filtré ; seules les limites physiques sont appliquées. Le professeur n'est jamais consulté pour produire les commandes de ces phases.
- `guarded` : la prédiction est projetée par le filtre local classique déjà utilisé par le professeur. Ce résultat est une politique hybride ; il ne démontre pas une sécurité apprise.

Le nombre d'interventions compte les changements au-delà du simple bornage physique de la commande, avec un seuil numérique 10⁻⁶. Les infaisabilités et replis restent également comptés. Le temps total, les contacts de toutes les phases et les quatre métriques de mission sont conservés. Les états initiaux et paramètres sont identiques entre professeur, brut et protégé.

L'évaluation utilise validation par défaut et exige `--split test` explicitement pour la réserve finale. Les alias sont exclus pour éviter une double pondération des mêmes trajectoires. La MSE sur les états du professeur est rapportée séparément du contrôle en boucle fermée.

## Première expérience sur le pilote

Corpus 008c : 36 épisodes train uniques, 106 272 exemples ; 9 validations uniques, 26 568 exemples. Aucun fichier test chargé pour l'optimisation. Maximum 120 époques, lots de 512, taux 0,001, patience 15, graine 42.

L'entraînement s'arrête à 16 époques : meilleure validation dès l'époque 1 (MSE 0,205807). La perte train diminue ensuite jusqu'à 0,029403 tandis que validation se dégrade. Ce constat montre une insuffisance du premier essai sur ce domaine ; il ne suffit pas à prouver quelle modification de modèle serait optimale.

Sur les neuf formes validation uniques, départ 7200 :

| Variante | Réussite | Collisions | Erreur finale moyenne | Interventions filtre pendant mouvement |
|---|---:|---:|---:|---:|
| Professeur | 9/9 | 0 | ≈0 m | 6 047 |
| GAT brut | 0/9 | 45 | 0,032889 m | 0 |
| GAT protégé | 9/9 | 0 | 0,033524 m | 5 426 |

Les 45 contacts du brut se répartissent en 36 pendant le rendez-vous et 9 pendant le mouvement final. Le rendez-vous est commun aux neuf formes de ce départ : ces événements ne sont pas des configurations initiales indépendantes. Une faible erreur finale ne masque pas l'échec de sécurité du brut. Le protégé rencontre aussi 423 infaisabilités robot/pas du filtre dans l'ensemble du scénario ; ce repli reste signalé.

Le corpus pilote ne comporte qu'un départ train : les scènes communes de rendez-vous sont répétées pour toutes les formes. Cela justifie d'étendre le domaine initial plutôt que de conclure sur une seule configuration. Les résultats et poids de cette première expérience restent conservés.

## Deuxième expérience préparée

La configuration `config/multiform-data-expanded.yaml` conserve exactement les partitions de formes et les archives test. Elle passe à cinq départs train (7100–7104) et deux validation (7200–7201), en conservant le départ test 7300.

L'extension copie les archives existantes octet pour octet, vérifie leurs empreintes et collecte seulement les nouveaux épisodes. Son manifeste référence le manifeste source. Ajouter ou retirer des formes, modifier le professeur/protocole ou changer les graines test est refusé. Tous les nouveaux échecs restent enregistrés et empêchent une déclaration automatique `passed`.

Le nouvel essai repart des mêmes paramètres initiaux du modèle, avec un budget explicite de 80 époques, lots de 1 024, taux 0,0005 et patience 12. Les différences de données et de budget sont déclarées ; une amélioration éventuelle ne sera pas attribuée expérimentalement à la seule extension du corpus. Les poids restent sélectionnés par validation ; le test n'a pas encore été utilisé pour mesurer un modèle appris.

## Corpus élargi effectivement collecté

214 épisodes du professeur collectés ou copiés, tous réussis sur ce domaine. Train : 185 épisodes, dont 180 uniques ; validation : 20 épisodes, dont 18 uniques ; test : 9 uniques. Le second entraînement utilise 531 360 exemples train et 53 136 validation après masque et déduplication. Les neuf archives test sont identiques octet pour octet à celles du pilote, avec preuve d'empreintes conservée.

Le statut de qualité est `passed` et l'audit train/validation est réussi. Ce corpus comporte huit départs distincts (cinq train, deux validation et un test), chacun partagé par les formes de son ensemble. Les nombres d'exemples ne constituent donc pas autant de situations initiales indépendantes.

Preuves : `tasks/evidence/008d-expanded-manifest.json`, `008d-expanded-audit.json` et `008d-test-preservation.json`. Les résultats du premier essai sont dans `008d-local-target-run-v1.json`, `008d-local-target-validation-v1.json` et `008d-v1-collisions-by-phase.json`.

## Vérification logicielle avant le second entraînement

348 tests ciblés réussis : physique, épisodes, catalogue, affectation, professeur, reconnexion, données et extension immuable, modèle à cible locale et comparaison appariée. Ils vérifient notamment que la politique brute ne consulte pas le professeur pendant ses phases de mouvement, que les cibles transmises au callback sont des copies, que le test n'est pas chargé pour l'optimisation et que l'extension conserve ses archives.

## Résultats du second entraînement

Arrêt anticipé après 23 époques ; meilleur checkpoint à l'époque 11, MSE validation 0,1125160754. Les 18 épisodes validation uniques donnent 18/18 réussites au professeur et au GAT protégé, contre 0/18 au brut (70 événements de collision). Erreurs finales moyennes : ≈0 m, 0,018264 m et 0,018264 m respectivement. Le filtre du protégé intervient sur 9 246 commandes robot/pas de mouvement et rencontre 99 infaisabilités sur le scénario entier.

Le checkpoint sélectionné uniquement par validation est figé avant la première évaluation neuronale de test. Son empreinte est `a1e3b88b39386a3d490798b5c7717e3939416e52572a058045d7402ec6cf6d84`. Aucun poids ni hyperparamètre n'est modifié après ce test.

## Test final réservé : neuf formes inédites pour l'apprentissage

Test : ellipse, cœur, rectangle, H, K, T, Y, 3, 8, depuis le départ commun 7300. Les neuf géométries et cette graine sont exclues de train/validation. Il s'agit d'un seul départ partagé, pas de neuf tirages indépendants.

| Mesure | Professeur | GAT brut | GAT protégé |
|---|---:|---:|---:|
| Réussite | 9/9 (100 %) | 0/9 (0 %) | 9/9 (100 %) |
| Événements de collision | 0 | 54 | 0 |
| Erreur finale moyenne | ≈0 m | 0,012690 m | 0,012690 m |
| Stabilisation moyenne, réussites seulement | 29,833 s | null | 30,433 s |
| Interventions du filtre pendant mouvement | 5 751 | 0 | 5 323 |
| Infaisabilités du filtre, toutes phases | 45 | 0 | 207 |

Les 27 simulations (trois variantes × neuf formes) obtiennent une affectation complète, à vitesse nulle et graphe inchangé pendant les tours de diffusion. Le temps de toutes les phases reste inclus. L'empreinte du checkpoint est identique avant/après l'évaluation.

## Conclusion de ce jalon et limites

Le modèle apprend à rejoindre les cibles locales avec une faible erreur finale sur ces formes test. Il **n'a pas appris une sécurité suffisante** : ses contacts rendent tous les épisodes bruts infructueux. La version protégée est retenue comme prototype pour les fonctionnalités suivantes, avec son filtre explicitement classique. Le taux de réussite du prototype hybride ne sera pas attribué au seul GAT et il n'est pas plus rapide que le professeur dans cette campagne.

La sécurité du brut, les infaisabilités du filtre et la couverture limitée de huit départs du corpus restent des limites de recherche. Une amélioration future nécessitera de nouveaux protocoles et réserves finales ; ce test déjà examiné ne devra pas devenir un jeu de sélection. Aucune garantie universelle sur d'autres tailles, portées, obstacles ou transitions n'est formulée.

Preuves finales : `tasks/evidence/008d-local-target-expanded-run.json`, `008d-local-target-expanded-validation.json`, `008d-local-target-expanded-test.json`. Les poids et trajectoires sont dans `outputs/imitation/training-multiform-expanded-v1/`, `validation-multiform-expanded-v1/` et `test-multiform-expanded-v1/`.

## Commandes

```powershell
python -m src.imitation.local_target_train --dataset outputs/imitation/multiform-expanded-task008d --training config/local-target-training-expanded.yaml --output outputs/imitation/training-multiform-expanded-v1
python -m src.imitation.local_target_evaluate --run outputs/imitation/training-multiform-expanded-v1 --dataset outputs/imitation/multiform-expanded-task008d --output outputs/imitation/validation-multiform-expanded-v1 --split validation
python -m src.imitation.local_target_evaluate --run outputs/imitation/training-multiform-expanded-v1 --dataset outputs/imitation/multiform-expanded-task008d --output outputs/imitation/test-multiform-expanded-v1 --split test
```

Les dossiers existants sont refusés ; choisir une nouvelle sortie pour reproduire. Le module d'entraînement ne lit jamais test. L'évaluation test doit rester une mesure finale et ne pas guider une reprise de l'optimisation.
