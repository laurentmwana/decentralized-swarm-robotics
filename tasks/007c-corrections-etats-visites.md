# 007c — Robustesse par corrections sur les états visités

Statut : cycle de corrections, entraînement et évaluation terminés ; résultats partiels de robustesse, sans garantie de sûreté. Prototype cercle corrigé conservé ; base géométrique des nouvelles formes démarrée en 008. Aucun contrôle des lettres/figures encore validé.

Décisions : `docs/decisions/16-campagne-independante-corrections.md`.

## Pourquoi cette étape

Le modèle relatif réussit les cinq validations sans collision mais seulement 11/20 nouvelles simulations (graines 400–419), avec cinq événements de collision. Le professeur réussit 19/20, avec un événement. Le jalon de robustesse du cercle reste ouvert ; ne pas annoncer la validation des formations ligne/V sur cette base.

## Réalisation

- Collecte des commandes du professeur sur les états GAT des 19 épisodes train, sans charger validation/test.
- 1 900 graphes supplémentaires ; échecs du collecteur conservés ; actions exécutées et labels attendus distincts.
- Supplément séparé et contrôle d'intégrité/provenance ; aucune modification des archives de la collecte source.
- Entraînement depuis les poids relatifs, MSE inchangée, 3 800 graphes train, 500 validation, taux 0,0003 et 100 époques maximum.
- Modèle initial inclus dans la sélection sur validation, avec traçabilité des poids et des corrections.

Preuve de collecte : `tasks/evidence/007c-circle-corrections-manifest.json`.

## Résultat réel

- 100 époques exécutées ; meilleur checkpoint à l'époque 80 ; MSE validation 0,16171945.
- Validation : 5/5 réussites, aucune collision, erreur finale entre 0,95 et 1,99 cm.
- Nouvelle campagne 600–619 : professeur 19/20, GAT corrigé 17/20 (85 %) ; 2 événements de collision GAT et erreur finale moyenne 0,022025 m.
- Comparaison appariée sur ces mêmes graines : ancien GAT relatif 14/20 (70 %), 3 collisions et erreur finale moyenne 0,042657 m. Aucun ajustement ultérieur sur ces graines.
- 87 tests du pipeline réussis. Les trois échecs de la variante corrigée restent dans le rapport ; le jalon de robustesse universelle n'est pas déclaré atteint.

Rapports dans `tasks/evidence/007c-circle-*` ; poids : `outputs/imitation/training-circle-corrections-v1/best.pt`.

## Suite

Étudier les échecs résiduels sans sélectionner de nouveaux poids sur les campagnes finales déjà examinées. Pour de nouveaux essais d'apprentissage, revenir à train/validation et réserver de nouvelles graines finales. Avancer la géométrie 008 sans confondre des points cibles avec un contrôle décentralisé réussi.
