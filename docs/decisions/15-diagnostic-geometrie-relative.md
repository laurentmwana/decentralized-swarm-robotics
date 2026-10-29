# 15 — Diagnostic du cercle appris et variante à arêtes relatives

Date : 2026-10-29. Préparation vérifiée sans entraînement ; qualité de contrôle du nouveau modèle non mesurée.

## Mesures reproductibles

Le diagnostic `src.imitation.diagnose` charge exclusivement les archives train/validation. Il contrôle le run terminé, les empreintes du manifeste et du checkpoint, ainsi que les métadonnées du modèle. Sur validation, il rejoue le GAT depuis les états initiaux archivés, sauvegarde les trajectoires et calcule les commandes bornées que le professeur aurait demandées sur les états effectivement visités. Il ne crée aucun optimiseur. La distance au plus proche robot est une mesure globale d'analyse hors ligne, pas une entrée supplémentaire du contrôleur.

Preuve : `tasks/evidence/007b-circle-diagnostic.json`. Protocole conservé : erreur 0,1 m, vitesse 0,05 m/s, maintien 1 s, délai 10 s.

| Mesure | Train, états professeur | Validation, états professeur | Validation, états GAT |
|---|---:|---:|---:|
| Exemples robot/pas | 22 800 | 6 000 | 6 000 |
| MSE des commandes, m²/s⁴ | 0,261176 | 0,297486 | 1,743084 |
| Erreur quadratique radiale | 0,051474 | 0,070140 | 0,183752 |
| Erreur quadratique tangentielle | 0,470879 | 0,524832 | 3,302417 |
| Exemples sous la distance de sécurité | 92 | 0 | 2 665 |

Les deux erreurs projetées sont des carrés de composantes ; leur somme vaut deux fois la MSE cartésienne. La composante tangentielle représente environ 88 % de l'erreur quadratique en validation sur les états professeur.

La moitié des exemples validation se situe à partir de 5 s : leur MSE vaut 0,005258, contre 1,040817 dans les deux premières secondes. Une faible perte moyenne ne démontre donc pas un contrôle correct lors de la mise en formation. Le jeu train ne comporte que 0,40 % d'exemples sous la distance de sécurité ; les états GAT validation en comportent 44,42 %.

Les cinq rollouts GAT validation échouent et comptent 19 événements de collision. Les démonstrations professeur validation réussissent toutes sans collision. Le RMSE final des positions par identité se situe entre 0,823 et 1,289 m ; cette mesure de dérive par robot n'est pas l'erreur de formation, qui reste calculée séparément. Ces constats démontrent une erreur tangentielle importante et un déplacement de distribution ; ils n'établissent pas à eux seuls la cause exclusive de l'échec.

## Limite d'attention et correction candidate

L'attention initiale calcule `LeakyReLU(query_i + key_j)`. Comme l'activation est strictement croissante, deux récepteurs partageant les mêmes voisins classent ceux-ci dans le même ordre. Les poids normalisés peuvent différer, mais le classement ne dépend pas de la proximité au récepteur. Les messages ne contiennent pas non plus de géométrie explicite par paire. Le décodeur dispose toujours de l'état propre : cette limite ne prouve pas une impossibilité générale d'apprendre le professeur.

Ajouter une variante opt-in `relative_edges: true` : déplacement relatif, vitesse relative et distance (5 scalaires normalisés) pour chaque arête directe. Un terme linéaire géométrique complète le score d'attention et un MLP géométrique complète son message. Le modèle conserve un seul échange local, les deux couches d'encodage, le décodeur, les bornes physiques et la MSE inchangée. Le même nombre de couches d'échange n'implique pas le même nombre de paramètres : la variante possède deux modules supplémentaires.

Les arêtes hors portée sont masquées. Les poids restent partagés. Les observations, le professeur et la collecte restent inchangés. Le fichier `config/imitation-model-relative.yaml` choisit explicitement cette variante. Les anciens fichiers YAML gardent le calcul et les métadonnées du schéma 1 ; les sauvegardes relatives portent le schéma 2 et les versions incohérentes sont refusées.

Ne pas changer simultanément l'objectif ou les données pour ce premier essai : cela permettra d'observer l'effet de la représentation. Si validation reste insuffisante, étudier séparément la pondération des transitoires et une collecte de corrections sur états visités, sans utiliser le test pour choisir les modifications. La variante relative n'offre aucune garantie de sécurité, notamment dans les situations peu représentées par la collecte.

## Vérification et suite

68 tests réussis sur modèle, raccordement, diagnostic et variante relative. Les tests vérifient le classement dépendant de la distance, la localité des dérivées, les gradients finis, les permutations, les translations communes, les lots, les bornes, les deux formats de sauvegarde et l'inférence par étoile locale. Aucun poids n'est mis à jour.

Le mode à blanc de la commande relative charge 1 900 graphes train et 500 validation, sans créer de dossier ni d'optimiseur. L'empreinte du `best.pt` initial reste `d778a2d9de10f40f5b17a7385ac54bcd0e067661587b8e032eea35fa2d43a6b4`.

L'utilisateur doit exécuter le nouvel entraînement, puis le diagnostic validation. Les graines test 300–304 ayant déjà été examinées, elles ne doivent pas guider la sélection de cette variante. Une évaluation finale ultérieure exigera un protocole figé et de nouveaux épisodes indépendants, conservés même en cas d'échec. Le jalon cercle et la tâche 007b restent ouverts jusqu'à validation réelle du modèle appris.
