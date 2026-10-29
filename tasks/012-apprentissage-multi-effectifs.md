# 012 — Généraliser à plusieurs effectifs, dont 24 et 64

Statut : premier pilote multi-effectifs entraîné et évalué ; généralisation à 24/64 non validée. Besoin précisé par l’utilisateur : 12 robots ne suffisent pas pour couvrir 24 et 64.

Mise à jour : le candidat structuré de la tâche 014 atteint ensuite 18/18 réussites brutes sur une nouvelle réserve contrôlée aux trois effectifs. Le présent fichier conserve les résultats du premier pilote ; les départs dispersés généraux et les autres missions restent non validés.

## Pilote livré

Un seul réseau partagé est entraîné sur les trois effectifs. Profils : arènes 10/18/32 m, largeurs 6/12/28 m pour 12/24/64 robots. Départs contrôlés proches du rendez-vous propre, pas des départs dispersés généraux. Trois formes train, trois validation et trois test, départs 8100/8200/8300 séparés. Les 18 épisodes de collecte réussissent sans contact robot/robot observé.

Stockage CSR local, sans voisins invisibles ; 8 640 exemples sélectionnés par effectif dans train et validation, soit 25 920 par partition. Modèle de largeur 64, normalisations spatiale 28 et effectif 64, 40 époques maximum ; meilleur checkpoint 38, MSE validation 0,046263. Le modèle de référence reste inchangé.

| Effectif, trois scénarios réservés chacun | Professeur | GAT brut | GAT protégé | Contacts brut |
|---|---|---|---|---|
| 12 | 3/3 | 2/3 | 3/3 | 1 |
| 24 | 3/3 | 0/3 | 0/3 | 65 |
| 64 | 3/3 | 0/3 | 0/3 | 127 |

Les modèles protégés n’ont pas de contact robot/robot, mais ne convergent pas vers les grandes formations. Aucun temps de réussite n’est imputé aux échecs. Un seul départ test est réutilisé sur trois formes par effectif : ce n’est pas une validation statistique large. Deux formes (E et 10) sont refusées à 24/64 pour défaut de connexité avec la portée de 3 m ; ces refus sont conservés dans l’audit des 56 formes.

Les optimisations gardent tous les tours et messages du protocole. Le benchmark à 64 passe de 69,6 à 31,5 s avec positions, vitesses et accélérations strictement identiques. 590 tests passent sur la suite complète. L’interface prépare un mode multi-effectifs explicitement expérimental, limité à la formation ; il ne remplace pas le modèle de référence.

Preuves : `tasks/evidence/012-multisize-{benchmark,manifest,run,evaluation}.json`. Les critères ci-dessous restent les objectifs de généralisation au-delà de ce pilote.

## Constats

Le réseau accepte des voisinages variables mais les données multi-formes actuelles utilisent uniquement 12 robots. Les effectifs 8/16/24 de l’interface sont exploratoires ; 64 n’est pas encore proposé dans cette nouvelle démonstration. La capacité architecturale à accepter N différent ne constitue pas une validation à cet effectif.

Le protocole courant réserve N² tours d’affectation à dt 0,1 s, soit 14,4/57,6/409,6 s pour N=12/24/64, avant construction finale. Il faut adapter les durées ou étudier un protocole local plus efficace, sans verdict global caché. Le rendez-vous de 64 robots sur un anneau espacé de 0,6 m ne tient pas dans l’arène actuelle de 10 m ; certaines figures de largeur 6 m sont également trop denses pour les marges de sécurité.

## Critères de livraison

1. Déclarer les effectifs train/validation/test et leur distribution, incluant 12, 24 et 64 dans l’apprentissage demandé. Séparer les départs et groupes géométriques, conserver une réserve indépendante par effectif.
2. Définir dimensions d’arène, forme, départ et rendez-vous physiquement réalisables pour chaque effectif ; mesurer et conserver les cas irréalisables au lieu de simplement augmenter N.
3. Adapter le budget d’allocation et de contrôle. Aucun arrêt de diffusion basé sur une information globale indisponible aux robots.
4. Auditer coût CPU/mémoire et représentation des messages avant collecte à 64 ; éviter les duplications denses inutiles sans modifier la localité.
5. Collecter et entraîner un candidat distinct ; déclarer les informations d’effectif et normalisations. Conserver le modèle de démonstration actuel comme référence.
6. Rapporter réussite, erreur, stabilisation, contacts et interventions du filtre séparément pour chaque effectif, en modes brut et protégé.
7. N’ajouter 64 dans l’interface avec un statut validé qu’après ces évaluations. Un mode exploratoire éventuel doit être nommé comme tel et vérifier la faisabilité avant calcul.

La correction 011 à 12 reste une expérience de comparaison à effectif constant ; elle ne remplit pas ces critères de généralisation.
