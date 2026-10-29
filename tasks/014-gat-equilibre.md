# 014 — Mouvement appris brut, équilibre et interactions locales

Statut : objectif mesurable atteint dans le domaine contrôlé annoncé. Réserve finale : 18/18 réussites brutes sans contact, six essais par effectif. Audit des 36 trajectoires appariées passé et 597 tests logiciels réussis.

Le modèle structuré est disponible dans l’interface sous « GAT structuré brut · 12/24/64 ». Les anciens poids sont conservés comme référence. Cette validation ne couvre pas les départs dispersés généraux, toutes les formes du catalogue, la navigation et les obstacles avec ce nouveau modèle.

## Objectif mesurable

Réussir le contrôle de formation en mode brut sur une nouvelle réserve à 12, 24 et 64 robots : aucun contact robot/robot ni mur, erreur ≤ 0,10 m et vitesse ≤ 0,05 m/s maintenues une seconde avant le délai du profil. Le domaine immédiat reste celui du pilote 012 (départs proches du rendez-vous). Les phases de freinage et d’attribution restent classiques et communes ; « brut » porte sur les phases de mouvement 0/3, sans filtre de sécurité caché.

La réserve est enregistrée avant apprentissage : départs 8500/8501, rectangle/C/3, trois effectifs. Elle n’est pas exécutée avant validation en boucle fermée. Aucun remplacement des poids de l’interface avant l’atteinte des critères et l’audit.

## Diagnostic établi

Le candidat 012 protégé réussit l’attribution à 12 mais pas à 24/64. Les graphes d’attribution sont stationnaires et les vitesses nulles ; le regroupement appris a néanmoins perturbé la géométrie et les revendications restent non résolues. Le filtre ne répare pas cette absence de convergence. Ajouter seulement des tours d’allocation ne constitue pas la correction ciblée.

## Candidat distinct

`VectorBasisGAT` accepte les mêmes observations locales mais utilise des bases vectorielles avec coefficients appris par attention. Sa sortie est exactement nulle lorsque cible, vitesse propre et vitesses relatives sont à l’équilibre. Les rotations et translations du plan sont respectées par construction. Le mouvement nominal emploie deux gains positifs appris sur des exemples isolés du professeur, puis gelés ; ils ne sont pas remplacés par des constantes dans l’inférence. Une correction de messages apprise complète cette base, avant bornage physique de la sortie.

Il s’agit d’un modèle structuré, et non de la simple continuation des huit couches linéaires du modèle 008d. Sa structure apparentée à une commande proportionnelle/amortie est explicitement déclarée ; aucune projection de contraintes ni fonction professeur n’est appelée pendant son inférence brute.

12 000 exemples locaux synthétiques train (8400) et 3 000 validation (8401) complètent les données CSR 012. Les labels infaisables sont comptés et exclus individuellement. Une pénalité de contraintes réciproques est calculée pendant l’apprentissage seulement. Les exemples de validation ne reçoivent pas de gradients.

Les choix et résultats de chaque tentative sont conservés. Si la validation échoue, la méthode peut être revue avant le nouveau test réservé ; un test déjà consulté ne sera pas réutilisé pour annoncer une validation finale inédite.

## Résultat final

Le premier candidat obtient 6/9 réussites en validation et six contacts. Le second renforce la pénalité d’apprentissage (poids 200 et distance visée 0,50 m), sans ajouter de filtre à l’inférence. Les gains nominaux appris valent 6 et 4. Sélection à l’époque 41, arrêt à 53, sur la seule validation. Les neuf scénarios de validation réussissent ensuite sans contact, avant l’ouverture de la réserve finale.

| Effectif | Réussites brutes réservées | Contacts robot/mur | Stabilisation moyenne | Distance minimale balayée entre centres |
|---|---|---|---|---|
| 12 | 6/6 | 0/0 | 29,433 s | 0,545886 m |
| 24 | 6/6 | 0/0 | 85,233 s | 0,515774 m |
| 64 | 6/6 | 0/0 | 465,800 s | 0,410773 m |

L’audit confirme zéro intervention du filtre dans les phases 0/3, les seuils maintenus, les bornes et l’intégration physiques, les mêmes départs du professeur et du candidat et les poids figés. Les deux départs par effectif sont réutilisés sur trois figures : six dispositions initiales distinctes, pas 36. La distance de contact physique est 0,20 m ; la marge d’apprentissage 0,50 m n’est pas une garantie puisque le minimum observé à 64 est inférieur à cette marge.

Poids figés : `outputs/imitation/training-vector-gat-v2/best.pt`, SHA-256 `411d902b0fbf52d60f79ca1c115fb07c4563602d3dc2ce03f6885c4824bc952a`. Preuves : `tasks/evidence/014-vector-{augmentation,run-v1,run-v2,validation-v2,fresh-test,final-audit}.json`. Décision : `docs/decisions/30-gat-structure-validation.md`.
