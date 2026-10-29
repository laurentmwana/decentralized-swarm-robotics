# 017 — Choix de 5 à 10 obstacles et message selon le modèle

Statut : terminé et validé. Demande utilisateur du 29 octobre 2026.

Suite finale : **621 tests réussis**. Vérification navigateur : messages distincts à 24 robots selon le modèle, sélecteur 5–10, simulation réelle à dix obstacles réussie par les deux commandes avec zéro contact. Serveur unique relancé avec la version à jour.

## Livraison

- Explication de l'effectif liée au contrôleur effectivement choisi : référence entraînée à 12, GAT structuré entraîné à 12/24/64 en formation contrôlée, classique sans réseau.
- Nouveau parcours « Naviguer avec plusieurs obstacles », nombre entier de 5 à 10. La scène est fixe et reproductible, avec trois obstacles sur la bande traversée et des obstacles supplémentaires autour de la formation. L'utilisateur règle le nombre, pas une génération aléatoire de scènes.
- Le planificateur examine tous les obstacles et toutes les places ; les destinations et passages incompatibles restent refusés.
- Protection des murs intégrée à la projection d'accélération pendant la préparation et la navigation, avec freinage anticipé et marge physique. Les bornes du moteur et la détection des contacts ne sont pas changées.
- Pour ce nouveau parcours, préparation classique explicite, puis navigation par la commande sélectionnée (classique ou GAT de référence protégé). Le réseau n'est pas réentraîné et ne reçoit pas la carte des obstacles.

## Diagnostic conservé

Pilote 8700 avec dix obstacles : l'ancien déroulement donne trois contacts muraux au GAT pendant la préparation. La protection des murs seule évite ces contacts mais le robot peut rester bloqué près d'un obstacle : préparation échouée, résultat conservé. La préparation classique suivie du déplacement appris protégé réussit avec zéro contact. Ces diagnostics ne sont pas comptés comme validation générale.

## Validation

Protocole enregistré : carré, 12 robots, tous les nombres 5–10 sur le départ 8701, dix obstacles sur 8702, et régression à un obstacle sur 8702. Deux contrôleurs appariés, seuils inchangés, audit physique et métriques. Les trajectoires et résultats restent conservés même en cas d'échec.

Résultats : 8/8 par contrôleur, zéro contact robot/obstacle/mur. Sept cas multi-obstacles et une régression simple par contrôleur ; déformation maximale du GAT 0,010388 m et distance finale maximale 0,005550 m. Audit physique et métriques : seize trajectoires sans divergence. Les sept préparations classiques sont identiques entre les deux variantes. Les scènes partagent leurs placements et ne constituent pas huit scènes aléatoires indépendantes. Preuves : `evidence/017-multiple-obstacles-{evaluation,audit,diagnostics}.json`.

Une ancienne instance du serveur pouvait encore répondre avec l'ancienne API alors que le fichier HTML était à jour. Le serveur réserve désormais son port exclusivement sous Windows ; une seconde instance ne peut plus démarrer sur la même adresse. Test de régression ajouté.

Reproduction : `python -m src.reference.evaluate_multiple_obstacles --output outputs/navigation-multiple/nouveau-dossier`.

Décision : [33-obstacles-multiples](../docs/decisions/33-obstacles-multiples.md).
