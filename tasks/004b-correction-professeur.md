# 004b — Corriger le professeur sur positions dispersées

Statut : validée par l’utilisateur ; commit autorisé.
Décisions et raisons : docs/decisions/11-correction-professeur-disperse.md.

## Réalisation

- Diagnostic des traces : rayon atteint, répartition angulaire trop lente.
- Correction des gains dans config/circle-teacher.yaml : radial 12, amortissement 6, espacement 60, évitement 60.
- Loi locale, seuils, délai et conditions initiales du pilote préservés.
- Tests unitaires existants adaptés au nouveau paramétrage ; 11 tests de régression supplémentaires (simulations classiques sans entraînement).
- Rapports de réévaluation et de validation versionnés dans tasks/evidence/.

## Résultats

- Pilote 0–9 : 9/10 réussites ; erreur moyenne 0,00504 m ; un contact initial, conservé comme échec.
- Validation indépendante 10–29 : 17/20 réussites ; erreur moyenne 0,00941 m ; trois contacts initiaux, conservés comme échecs.
- 26/26 épisodes initialement sans chevauchement réussis ; aucun nouveau contact détecté. Pas de garantie universelle ni de conclusion sur d'autres tailles/portées.

## Vérification

128 tests réussis : 117 précédents et 11 tests de régression supplémentaires.
Les tests historiques nécessitant PyTorch restent hors du périmètre vérifié dans l'environnement disponible.
Aucun entraînement neuronal exécuté.

## Suite après validation et commit

Tâche 005 : définir et tester le domaine des initialisations de collecte (notamment les chevauchements), générer les données du professeur et garantir la séparation par épisodes entre entraînement, validation et test. Aucun entraînement ne sera lancé par l'agent.
