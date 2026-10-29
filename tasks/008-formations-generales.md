# 008 — Formations générales : géométrie, répartition locale et transitions

Statut : catalogue, professeur, reconnexion, corpus, GAT protégé et transitions continues implémentés et évalués sur le domaine ciblé. Le brut reste limité par ses collisions ; suite fonctionnelle en 009.

Périmètre confirmé par l'utilisateur le 2026-10-29 : A–Z, nombres 1–10 et au moins quinze géométries. Le catalogue livré comprend 56 entrées : 26 lettres, 10 nombres et 20 géométries.

## Réalisation actuelle

- Chemins ouverts/fermés métriques et missions multitraits.
- Alphabet A–Z, nombres 1–10, et vingt géométries : cercle, ellipse, carré, rectangle, triangle, pentagone, hexagone, heptagone, octogone, décagone, losange, trapèze, parallélogramme, étoile, croix, flèche, demi-cercle, cœur, ligne et V.
- Identifiants explicites, alias français et transformations de taille/orientation/position.
- Places de référence distinctes, projection sur les traits et contrôles de confinement, densité et portée.
- 147 tests géométriques réussis, couvrant les 56 entrées à 12 et 24 robots ; décisions dans `docs/decisions/17-formations-lettres-figures.md` et `18-catalogue-complet.md`.
- Export JSON et planche SVG : `outputs/formations/catalog-complete-v1/` ; les configurations physiquement incompatibles sont signalées.
- Protocole local de revendications versionnées : identifiants uniques, priorité déterministe, un saut par tour, aucun partage de positions distantes ou de coûts.
- Professeur classique à places attribuées, filtre local d'accélération et métriques incluant le temps réel d'affectation.
- 303 tests ciblés physique/catalogue/affectation/contrôle/sécurité réussis, puis le test ajouté du domaine dispersé réussi : 304 cas au total.
- Pilote nominal : 37/56 réussites et 30 collisions. Même pilote filtré : 47/56 et zéro collision, 10 infaisabilités robot/pas conservées.
- Validation dispersée, deux départs et 56 formes chacun : 47/112 réussites sans collision. Un départ possède deux composantes (11 robots et 1 isolé) : ses 56 affectations échouent explicitement, sans exclusion.

La reprise 008b ajoute le rendez-vous et corrige les détours : 112/112 réussites sur les départs conservés, puis 168/168 sur trois nouveaux départs, sans collision observée ; 314 tests ciblés réussis. Les replis du filtre restent signalés et la conclusion reste limitée au domaine testé. Aucun modèle multi-formes n'est entraîné. Détails : décisions 19 à 21 dans `docs/decisions/`.

La reprise 008c ajoute le schéma de données v2 et une collecte pilote complète des 56 entrées : train 37 épisodes, validation 10, test 9 ; deux alias géométriques signalés dans leur propre partition, soit 54 simulations uniques. Les 56 épisodes du professeur réussissent, l'audit complet passe et 326 tests ciblés sont réussis. Le contrat local n'expose pas les cibles privées des voisins. Décision 22 et preuves `tasks/evidence/008c-multiform-*`.

La reprise 008d construit un GAT distinct et élargit le corpus à 214 épisodes du professeur. Après apprentissage et sélection sur validation uniquement : prototype protégé 9/9 sur les formes test, zéro collision observée, erreur moyenne 1,27 cm ; brut 0/9 avec 54 contacts. Le rôle du filtre classique est mesuré et ne relève pas d'une sécurité apprise. 348 tests ciblés réussis ; décision 23 et preuves `tasks/evidence/008d-*`.

La reprise 008e ajoute les transitions sans remise à zéro physique, les versions de mission et le comptage unique des contacts aux frontières. Pilote : 6/6 séquences pour chaque contrôleur ; validation mixte : 2/2 (huit stages) chacun, sans collision observée. 401 tests ciblés réussis ; décision 24 et preuves `tasks/evidence/008e-*`.

## Prochain travail

1. Implémenter et mesurer la navigation collective/obstacles en 009, en conservant les limites du brut et des filtres.
2. Réserver de nouveaux tests pour toute nouvelle amélioration de sécurité et mesurer l'interface/campagnes finales en 010.

Le prototype cercle corrigé atteint 85 % sur les graines finales 600–619, avec deux événements de collision ; cette limite reste suivie en 007c et ne constitue pas une garantie pour les nouvelles formes.
