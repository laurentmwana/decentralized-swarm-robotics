# 30 — Mouvement brut du GAT structuré validé sur la réserve annoncée

Date : 2026-10-29. L’utilisateur demande de poursuivre jusqu’à l’atteinte de l’objectif. Le critère annoncé est une formation stabilisée sans contact sur de nouveaux essais réservés à 12, 24 et 64 robots. Le domaine immédiat est celui du pilote 012, avec départs proches du rendez-vous.

## Diagnostic et architecture

Le candidat 012 produit une petite erreur d’imitation mais perturbe le rendez-vous. À 24/64, les graphes sont ensuite stationnaires et les vitesses nulles pendant l’allocation, mais les revendications restent non résolues. Le filtre évite les contacts sans rétablir la mission. Cette absence d’équilibre du mouvement appris est corrigée dans un candidat distinct, sans supprimer les échecs précédents.

`VectorBasisGAT` conserve des observations locales : état/cible propres, phase/effectif communs et messages physiques relatifs des voisins visibles. Les messages n’incluent aucune cible privée voisine. Les coefficients d’attention sont calculés à partir de caractéristiques scalaires invariantes ; les messages sont recombinés sur des bases vectorielles relatives. La politique respecte les translations et les rotations du plan, et la permutation des voisins.

La commande nominale a une structure proportionnelle/amortie à deux gains positifs appris par régression sur 8 192 exemples isolés du professeur (graine 8402), pendant 1 200 pas d’optimisation. Les valeurs finales 6 et 4 sont stockées comme paramètres ; l’inférence ne les remplace pas par des constantes du professeur. Les gains sont ensuite gelés pour apprendre la correction d’attention. La commande nominale est bornée, complétée par les messages appris, puis rebornée à 2 m/s².

Le modèle contient huit modules linéaires de largeur interne 48 et deux gains scalaires appris. L’activité des messages est nulle lorsque cible et position coïncident, que la vitesse propre est nulle et que les voisins sont immobiles. Cette structure impose l’équilibre et évite une force résiduelle arbitraire à cible atteinte. Ce n’est pas la continuation inchangée du GAT 008d : le biais inductif de commande est déclaré. Aucune projection de sécurité, aucun professeur ni solveur QP n’est appelé dans le mouvement brut de ce réseau. Les bornes physiques du simulateur restent actives.

## Données et sélection

Les données CSR train/validation 012 sont conservées, avec 6 000 exemples sélectionnés par effectif et partition. Elles sont complétées par 12 000 exemples locaux synthétiques train (8400) et 3 000 validation (8401), comprenant positions relatives, vitesses et objectifs variés. Les 100 et 28 états de labels infaisables sont exclus et comptés ; les tentatives et empreintes sont archivées. Il y a au total 30 000 exemples train et 21 000 validation pour l’apprentissage des messages.

Les labels sont produits par le professeur local filtré, uniquement pendant la préparation des données. La perte comporte une imitation pondérée et une pénalité de contraintes réciproques d’accélération, calculée pendant l’apprentissage seulement. Les états de validation ne reçoivent aucun gradient.

Version 1 : 80 époques maximum, perte de barrière pondérée 20 et distance 0,40 m. La validation en boucle fermée atteint 6/9 sans contact sur les six réussites, mais trois scénarios échouent avec six contacts au total. Ces échecs ne sont pas utilisés comme données train. Une ablation de validation sur un face-à-face de deux robots actifs parmi douze montre un contact avec les seuls gains appris, contre zéro avec l’attention du candidat v1 et une cible finale atteinte. Cela illustre l’apport de l’interaction apprise sur ce cas, sans constituer une preuve générale.

Version 2 : reprise des poids v1, taux 0,0003, maximum 60 époques, patience 12, poids de barrière 200 et distance d’apprentissage 0,50 m. Les autres données restent séparées. Sélection à l’époque 41, arrêt à 53. La perte composite de validation vaut 0,239340 et n’est pas directement comparable à la perte v1, car la pondération et la distance visée changent. Les neuf scénarios de validation 8200, triangle/B/7 aux trois effectifs, réussissent sans contact avant le test final.

## Réserve finale et audit

La réserve est enregistrée dans le manifeste d’augmentation avant apprentissage : rectangle/C/3, départs 8500/8501, effectifs 12/24/64. Les poids v2 sont figés avant son ouverture. Il n’y a aucun réglage après observation de ce test.

| Effectif | Réussites du GAT brut | Contacts robots/murs | Erreur finale moyenne | Stabilisation moyenne |
|---|---:|---:|---:|---:|
| 12 | 6/6 | 0/0 | 1,7876e-7 m | 29,433 s |
| 24 | 6/6 | 0/0 | 3,4413e-7 m | 85,233 s |
| 64 | 6/6 | 0/0 | 8,6717e-7 m | 465,800 s |

Le professeur apparié réussit aussi 18/18 sans contact, avec stabilisations moyennes 30,000 / 85,233 / 465,933 s. Les temps incluent le rendez-vous, le freinage et les N² tours, dont 409,6 s d’allocation à 64. Ils ne représentent pas le temps réel de calcul sur le processeur.

L’audit des 36 trajectoires vérifie : états finis, limites physiques, intégration à dt 0,1 s, continuité temporelle, contacts balayés entre états, stabilité dans les seuils pendant une seconde, identité des départs appariés, consensus final, empreintes de poids et zéro intervention du filtre dans les phases de mouvement 0/3. Les phases de freinage 1 et d’allocation 2 restent classiques et filtrées, comme dans les expériences précédentes. « Brut » ne signifie donc pas que tout le protocole collectif est neuronal.

Les distances minimales balayées entre centres sont 0,545886 / 0,515774 / 0,410773 m, pour une distance de contact physique de 0,20 m. La distance 0,50 m de la perte d’apprentissage n’est pas une garantie formelle : elle n’est pas toujours maintenue. Le contrôle brut est validé sur cette réserve finie, pas certifié pour tous les états.

Les dix-huit scénarios réutilisent deux départs par effectif, soit six dispositions distinctes. Le regroupement les rapproche du même rendez-vous. Les départs dispersés généraux, pertes de communication, bruit de mesure, obstacles et navigation avec ce nouveau modèle, ainsi que les autres figures du catalogue, restent à évaluer. Le but mesurable de cette étape est atteint ; ces limites ne sont pas présentées comme résolues.

## Interface et conservation

L’interface propose le candidat sous « GAT structuré brut · 12/24/64 » et une comparaison avec le professeur. Il utilise les profils contrôlés 012 et ne bénéficie pas de filtrage de sécurité pendant les mouvements appris. Le mode est limité à la formation. Le modèle 008d protégé reste disponible séparément, avec ses poids inchangés ; ses résultats historiques ne sont pas réétiquetés comme ceux du nouveau modèle.

L’option du candidat ne s’active que si les dix-huit cas attendus sont réussis, si l’audit est positif, si ses mouvements n’ont pas été filtrés et si l’empreinte du checkpoint correspond. Les graphes identiques sont stockés dans une banque avec indices de trames pour réduire le JSON sans supprimer d’image physique. Un test vérifie leur reconstruction exacte et l’absence de filtrage du mouvement dans l’export. Les exportations complètes antérieures sont restaurées au redémarrage.

Une comparaison réelle de rectangle à douze robots, départ 8501, est vérifiée dans le navigateur : professeur et GAT brut réussissent sans contact ; le GAT se stabilise à 29,2 s. Il s’agit d’un essai de démonstration, distinct du calcul des taux réservés. Capture : `outputs/demo-reference/014-structured-gat.png`. La suite complète réussit 597 tests.

Poids finaux : `outputs/imitation/training-vector-gat-v2/best.pt`, SHA-256 `411d902b0fbf52d60f79ca1c115fb07c4563602d3dc2ce03f6885c4824bc952a`. Le modèle de référence conserve `a1e3b88b39386a3d490798b5c7717e3939416e52572a058045d7402ec6cf6d84`.

## Reproduction et preuves

```powershell
python -m src.imitation.vector_experiment augment --output outputs/imitation/vector-augmentation-v1
python -m src.imitation.vector_experiment train --dataset outputs/imitation/multisize-pilot-task012 --augmentation outputs/imitation/vector-augmentation-v1 --output outputs/imitation/training-vector-gat-v1
python -m src.imitation.vector_experiment train --dataset outputs/imitation/multisize-pilot-task012 --augmentation outputs/imitation/vector-augmentation-v1 --source-run outputs/imitation/training-vector-gat-v1 --conservative --output outputs/imitation/training-vector-gat-v2
python -m src.imitation.vector_experiment rollout --run outputs/imitation/training-vector-gat-v2 --validation --output outputs/imitation/vector-validation-v2
python -m src.imitation.vector_experiment rollout --run outputs/imitation/training-vector-gat-v2 --output outputs/imitation/vector-fresh-test-v2
python -m src.imitation.audit_vector --run outputs/imitation/training-vector-gat-v2 --folder outputs/imitation/vector-fresh-test-v2 --output tasks/evidence/014-vector-final-audit.json
```

Les sorties existantes sont refusées. Les décisions de correction v2 ont été prises sur validation, pas sur la réserve finale. Les preuves conservées comprennent les manifestes, les deux entraînements, les contacts v1, la validation v2, le test final et son audit dans `tasks/evidence/014-*`.
