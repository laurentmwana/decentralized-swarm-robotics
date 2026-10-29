# 018 — Prévenir les erreurs de densité, de portée et de dimensions

Statut : terminé. **634 tests réussis**, 336 combinaisons géométriques examinées, vérification de l'interface réelle et export d'une flèche adaptée réussie. Demande utilisateur : gérer correctement « Formation trop dense en distance euclidienne » et les incompatibilités similaires.

## Correction

La démonstration imposait souvent une largeur de 6 m à tous les effectifs. Désormais, un aperçu calcule une largeur compatible avec le minimum de 0,40 m entre places, la portée de communication et les limites de l'arène. La forme garde ses proportions. Les dimensions originales sont conservées lorsqu'elles sont valides ; les autres sont agrandies ou réduites si l'intervalle autorisé n'est pas vide.

Lorsqu'un agrandissement est nécessaire, une marge préférée de 0,50 m entre places est recherchée si elle tient dans l'arène. Le délai de formation est porté à au moins 40 s après l'attribution pour les formes de référence agrandies. Ni le seuil de réussite ni la distance minimale de sécurité ne sont réduits.

L'interface affiche les dimensions avant simulation, désactive les options incompatibles et le lancement tant que l'aperçu n'est pas valide. Le prochain état est vérifié aussi pour les transitions. En navigation, le départ, la destination et les segments planifiés sont vérifiés pour la forme sélectionnée avant tout lancement. Les requêtes directes restent contrôlées par le serveur.

Les dimensions effectivement utilisées sont écrites dans le manifeste exporté. Une modification de dimension du GAT structuré reste identifiée comme expérimentale ; elle ne modifie pas les résultats réservés antérieurs ni les poids.

## Vérifications

- Catalogue complet : sécurité, connectivité et limites vérifiées pour les options proposées à 8/12/16/24/64, ainsi que le profil structuré 24.
- Dimensions exactes des trois formes réservées du GAT structuré préservées à 12/24/64.
- Aperçu HTTP sans création de tâche ni consommation du quota de simulations.
- Combinaisons irréalisables refusées avant calcul, avec conseil de changer forme/effectif.
- Simulations classiques sur nouveau départ 8801 : flèche à 24 robots, largeur 9,3762 m, réussite sans contact ; X à 8 robots, largeur 4,2335 m, réussite sans contact.

Diagnostic conservé : la première flèche redimensionnée au plus près du minimum de sécurité n'atteignait pas le seuil final. La marge préférée et le délai augmenté ont corrigé cet exemple ; cela ne vaut pas validation universelle de toutes les trajectoires.

Décision : [34-compatibilite-geometrique](../docs/decisions/34-compatibilite-geometrique.md). Archives : `outputs/geometry-fit/`.

Preuve versionnée : `evidence/018-geometry-compatibility.json`. La compatibilité géométrique ne constitue pas une validation dynamique de toutes ces combinaisons ; les deux simulations de validation concernent le contrôleur classique.
