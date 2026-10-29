# 27 — Imitation corrective du mouvement sans filtre à l’inférence

Date : 2026-10-29. Tâche 011 autorisée par l’utilisateur après l’interface.

## Hypothèse et périmètre

Le GAT brut 008d forme correctement les figures à la fin mais rencontre des contacts pendant ses mouvements. Une imitation évaluée seulement sur des états visités par le professeur couvre mal les écarts de trajectoire du modèle. Le candidat conserve l’architecture, les observations locales et la limite d’accélération de 008d ; il apprend des commandes corrigées sur ses propres états. Aucun filtre classique n’est ajouté aux phases apprises pendant les essais bruts. Freinage, allocation et protocole de communication restent classiques, comme dans la comparaison 008d.

La démonstration continue d’utiliser le modèle protégé validé, sans changement de poids. Aucun candidat n’est automatiquement promu à partir de cette expérience.

L’architecture compte un bloc d’attention de graphe et huit modules linéaires appris : deux pour l’état propre, deux pour les messages, deux projections query/key et deux pour la sortie. Ils forment plusieurs branches, pas huit couches toutes successives. La largeur cachée est 64, distincte du nombre de robots ; chaque robot exécute le même réseau et partage ses poids. L’effectif entraîné reste 12 pour cette expérience.

## Collecte et absence de fuite

Le collecteur lit seulement les archives train/validation de `multiform-expanded-task008d`, à leur première graine respective, sur leurs géométries uniques : train 7100 et validation 7200. Les archives test ne sont pas chargées. Les données initiales restent intactes. Les formes alias ne sont pas dupliquées ; le regroupement initial reste toutefois identique entre formes d’une même graine et les épisodes ne constituent pas autant de départs indépendants.

Chaque mouvement brut est exécuté et conservé, y compris ses contacts. Depuis l’état avant commande, le professeur utilise uniquement position, vitesse, cible propres et positions/vitesses relatives des voisins visibles. Son attraction, son évitement et sa projection bornée sont identiques au professeur multi-formes. Les nouvelles commandes supervisées sont stockées dans `teacher_labels`, distinctes des accélérations effectivement exécutées. Aucun objectif privé voisin n’entre dans le modèle.

Les labels pour lesquels les contraintes de sécurité sont infaisables restent archivés et comptés, mais leur masque individuel d’apprentissage vaut faux. Aucune trajectoire entière n’est exclue. Les phases apprises sont 0 et 3 ; freinage et allocation ne deviennent pas artificiellement des exemples appris.

## Optimisation préenregistrée

Les données initiales train/validation sont complétées par leurs propres corrections. Une observation pèse 8 si un voisin visible est à moins de 1 m, ou si un rapprochement linéaire prédit atteint moins de 0,6 m dans la seconde suivante ; sinon elle pèse 1. Les exemples correctifs ont en plus un multiplicateur 3. Cette importance n’est pas une garantie de sécurité.

Reprise du checkpoint 008d, Adam à 0,0003, lots de 1024, 30 époques maximum, patience 8, graine 42. Le checkpoint est sélectionné par MSE pondérée de validation ; les états de validation ne sont jamais optimisés. Le manifeste des corrections enregistre tous ces paramètres et la réserve finale avant apprentissage.

La réserve emploie les neuf formes test de la partition initiale mais deux nouveaux départs 7800 et 7801, disjoints des graines des données. Les résultats test 008d ont déjà été observés ; ils ne redeviennent pas une réserve inédite. Le nouveau protocole évalue le professeur, le modèle brut initial et le candidat brut, ainsi que le candidat protégé prévu au manifeste, sur les mêmes nouveaux départs, sans lire les anciens états test. Les résultats seront gelés après une seule campagne et ne serviront pas à retoucher ce candidat.

## Reproduction

```powershell
python -m src.imitation.multiform_corrections --run outputs/imitation/training-multiform-expanded-v1 --dataset outputs/imitation/multiform-expanded-task008d --output outputs/imitation/multiform-corrections-task011-v1
python -m src.imitation.train_multiform_corrections --dataset outputs/imitation/multiform-expanded-task008d --corrections outputs/imitation/multiform-corrections-task011-v1 --source-run outputs/imitation/training-multiform-expanded-v1 --output outputs/imitation/training-multiform-corrected-v2
python -m src.imitation.evaluate_multiform_fresh --dataset outputs/imitation/multiform-expanded-task008d --corrections outputs/imitation/multiform-corrections-task011-v1 --baseline outputs/imitation/training-multiform-expanded-v1 --candidate outputs/imitation/training-multiform-corrected-v2 --output outputs/imitation/multiform-corrected-fresh-task011
```

Chaque dossier de sortie doit être nouveau. Les poids source et manifestes sont identifiés par SHA-256, et les deux modèles restent séparés. Une baisse de MSE seule ne permet pas de déclarer le mouvement autonome sûr : les contacts et échecs en boucle fermée décident du résultat.

Le premier lancement `training-multiform-corrected-v1` s’est interrompu lors de la sauvegarde d’une amélioration, car l’écriture exclusive de checkpoint refusait un fichier déjà présent. Il est conservé comme lancement échoué. La sauvegarde utilise désormais un fichier distinct par époque améliorée puis une copie finale `best.pt`, comme le formateur initial. Le test d’optimisation couvre plusieurs époques. Le lancement `training-multiform-corrected-v2` reprend les mêmes poids, données et paramètres, sans réglage scientifique après cette interruption.

L’utilisateur a demandé pourquoi l’apprentissage se limitait à 12 alors que 24 et 64 sont visés. Cette expérience reste à effectif constant pour isoler l’effet des corrections ; elle ne valide pas ces autres effectifs. La tâche 012 décrit l’apprentissage multi-effectifs et les adaptations physiques/de communication nécessaires.

## Résultats figés

La collecte conserve 36 trajectoires train et neuf de validation : respectivement 106 096 et 25 979 exemples correctifs utilisables, dont 43 874 et 10 660 critiques. Les 176 et 589 labels infaisables sont comptés et masqués individuellement. Les trajectoires brutes comportent respectivement 100 et 46 contacts et restent toutes conservées.

Le lancement corrigé termine les 30 époques. Le meilleur score de validation pondérée est 0,382023 à l’époque 29 ; cette métrique utilise une distribution et une pondération différentes de la MSE 008d et n’est pas directement comparable à celle-ci. Empreinte du candidat : `e8d40571613479b093f123b978cd966d2fc3f32e676ca0da86449428885d514e`.

| Contrôleur, nouveaux départs 7800/7801 | Réussites | Contacts | Erreur finale moyenne | Stabilisation moyenne réussie |
|---|---:|---:|---:|---:|
| Professeur | 18/18 | 0 | ≈ 0 m | 29,689 s |
| Ancien GAT brut | 0/18 | 102 | 0,012690 m | absente |
| Candidat brut | 0/18 | 65 | 0,017829 m | absente |
| Candidat protégé | 18/18 | 0 | 0,017782 m | 30,867 s |

Les contacts diminuent de 36,3 %, mais le candidat brut échoue encore dans tous les scénarios. Sa faible erreur finale ne corrige pas les contacts survenus auparavant. Le résultat ne valide donc pas un mouvement autonome sûr. La version protégée réussit, sans attribuer la sécurité au seul réseau ; aucun remplacement du modèle d’interface n’est effectué. Le test n’est pas utilisé pour réajuster les poids ou paramètres de cette expérience.

L’audit des 72 trajectoires confirme les comptes de contacts balayés entre états, les bornes physiques, l’intégration, le temps, l’identité des départs appariés et les empreintes des deux modèles. Il s’agit de deux départs distincts réutilisés sur neuf formes, pas de 72 conditions initiales indépendantes. Les preuves sont conservées dans `tasks/evidence/011-corrections-manifest.json`, `011-corrected-run.json`, `011-interrupted-run.json`, `011-fresh-evaluation.json` et `011-fresh-audit.json`.
