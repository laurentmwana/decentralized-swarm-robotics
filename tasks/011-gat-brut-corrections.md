# 011 — Corriger le GAT brut sur ses propres états

Statut : première expérience corrective entraînée et évaluée ; autonomie sûre non atteinte. Les poids de démonstration restent inchangés.

Collecte : 36 trajectoires train et 9 de validation, 106 096 et 25 979 exemples correctifs utilisables ; 176 et 589 labels infaisables exclus individuellement, épisodes conservés. Le lancement initial a été interrompu par un conflit de sauvegarde, corrigé sans modifier les paramètres ; reprise dans `training-multiform-corrected-v2`.

Entraînement terminé à 30 époques, meilleur checkpoint 29, MSE pondérée validation 0,382023. Nouveau test figé : 18 scénarios, deux nouveaux départs réutilisés sur neuf formes, 72 trajectoires avec les quatre contrôleurs. Professeur 18/18 sans contact ; ancien GAT brut 0/18, 102 contacts ; candidat brut 0/18, 65 contacts ; candidat protégé 18/18 sans contact. Réduction observée des contacts de 36,3 %, mais aucune réussite brute. Le candidat n’est pas promu dans l’interface et aucune retouche n’est faite après ce test.

Audit des 72 trajectoires passé : contacts continus, physique, temps, départs appariés et poids figés. Preuves : `tasks/evidence/011-{corrections-manifest,corrected-run,fresh-evaluation,fresh-audit}.json`. La suite complète conservée après nettoyage du code historique réussit 578 tests.

Le modèle protégé de la démonstration reste inchangé. Cette expérience conserve son architecture et ses observations locales ; elle modifie les données et la pondération d’imitation. Elle ne remplace pas le GAT par un filtre caché.

## Protocole fixé avant entraînement

- Source : `multiform-expanded-task008d`, poids `training-multiform-expanded-v1/best.pt` figés.
- Collecte du GAT brut sur la première graine de train et de validation, toutes leurs géométries uniques ; aucun chargement d’archive test.
- Le professeur recalcule la commande filtrée sur l’observation propre, la cible propre et les voisins visibles de chaque état visité. Les commandes réellement exécutées restent distinctes de ces nouveaux labels.
- Les épisodes avec contacts restent conservés. Les labels pour lesquels la projection du professeur est infaisable sont comptés et exclus individuellement de l’optimisation ; aucun épisode n’est supprimé.
- Train et validation restent séparés, phases apprises 0/3 seulement. Les phases 1/2 restent classiques et communes aux variantes.
- Poids d’importance 8 pour un voisin visible à moins de 1 m ou un rapprochement linéaire prédit à moins de 0,6 m sous une seconde ; poids 1 autrement. Les exemples correctifs ont un multiplicateur 3, en complément des données d’origine.
- Reprise des poids initiaux, Adam 0,0003, lots 1024, maximum 30 époques, patience 8, graine 42. Sélection par MSE pondérée de validation seulement.
- Réserve finale enregistrée dans le manifeste avant apprentissage : neuf formes test d’origine, nouveaux départs 7800 et 7801. Comparaison appariée professeur, ancien GAT brut, candidat brut et candidat protégé ; aucune sélection ou retouche après ce test.

L’amélioration doit se juger en boucle fermée : une diminution de MSE ou une formation finale correcte ne suffit pas si des contacts ont eu lieu. Les résultats seront conservés même si le candidat échoue. Aucune promotion automatique dans l’interface.

## Lancer le serveur

Depuis la racine du projet, sous Windows : `./demo/start-reference.ps1`. Puis ouvrir `http://127.0.0.1:8765/`. Dans un environnement Python prêt : `python -m demo.reference_server`.
