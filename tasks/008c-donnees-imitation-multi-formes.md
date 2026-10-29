# 008c — Contrat et collecte d'imitation multi-formes

Statut : contrat et collecte pilote implémentés ; 56 épisodes du professeur réussis, audit complet passé ; 326 tests ciblés réussis. Aucun entraînement neuronal multi-formes lancé. Suite : 008d.

## Livraison réelle

- Schéma de données v2 distinct du cercle : état propre, cible propre, phase, effectif ; messages physiques relatifs sans cible privée voisine.
- Trajectoires et délais de toutes les phases archivés, masque d'apprentissage des mouvements 0/3, freinage/affectation classiques.
- Configuration figée : `config/multiform-data.yaml`, partition d'épisodes/graines et groupes de formes équivalentes.
- Jeu `outputs/imitation/multiform-pilot-task008c` : train 37 épisodes (36 uniques), validation 10 (9 uniques), test 9 uniques. 56/56 réussites du professeur dans ce pilote.
- Deux doublons d'alias conservés et signalés dans leur propre ensemble, aucun entre ensembles. Après déduplication : 54 simulations uniques, 159 408 exemples robot/pas avec masque appris.
- Audit par défaut limité à train/validation ; audit complet avec test exécuté explicitement et réussi. Statut `passed` limité à la qualité des épisodes collectés.

Preuves : `tasks/evidence/008c-multiform-manifest.json` et `008c-multiform-audit.json`.
Décision : `docs/decisions/22-donnees-imitation-multi-formes.md`.

Les sections suivantes conservent les exigences à respecter lors de l'entraînement ; le corpus reste un pilote à un départ distinct par ensemble.

## Périmètre à conserver

Catalogue de 56 entrées : A–Z, 1–10 et vingt géométries. Protocole courant : 12 robots, dt 0,1 s, vitesse 1 m/s, accélération 2 m/s², portée 3 m, départs sans chevauchement dans le disque de rayon 4 m, horizon total 40 s. Conserver le temps et les contacts du rendez-vous, du freinage et de l'affectation.

## Contrat d'information

- Le professeur et le futur GAT utilisent le même module classique de rendez-vous/affectation, avec les mêmes observations, messages et délais.
- La commande de mouvement reçoit la place propre attribuée ou la place intermédiaire propre. Ne pas remplacer celle-ci par le centre/rayon du modèle cercle.
- Les messages physiques donnent accès seulement aux voisins directs. Leurs positions/vitesses relatives sont disponibles ; leur place privée ne doit pas être injectée dans les caractéristiques du GAT sans ajouter explicitement cet échange au contrat.
- Le GAT doit avoir un schéma/checkpoint distinct, avec une consigne de cible locale. Les anciens poids cercle ne constituent pas un modèle multi-formes entraîné.
- Définir explicitement le rôle du filtre de sécurité pendant l'imitation et l'évaluation. Si le GAT est protégé par un filtre classique, mesurer ses interventions et distinguer ce résultat du contrôle brut appris.

## Collecte et intégrité

1. Enregistrer les états avant commande, cible locale, phase, voisins/graphes, accélérations demandées et appliquées, trajets complets, revendications et infaisabilités du filtre.
2. Définir quelles phases sont apprises et lesquelles restent classiques ; conserver toutes les phases dans les métriques de scénario.
3. Séparer les épisodes complets et les graines entre train/validation/test. Figer aussi une partition de formes pour mesurer la généralisation.
4. Regrouper les entrées géométriquement identiques (notamment lettre V/géométrie V, lettre O/cercle) dans une même partition ; les identifiants distincts ne suffisent pas à prévenir une fuite.
5. Contrôler les empreintes, les duplications et le statut de qualité. Conserver les échecs du professeur et les signaler, sans requalifier silencieusement une collecte comme prête.
6. Lancer la collecte avec budgets explicites et produire un manifeste reproductible avant tout entraînement.

## Vérification attendue

Tests d'alignement cible/phase/commande, absence d'accès aux cibles privées des voisins, séparation par épisodes et formes équivalentes, intégrité, conservation des échecs, et comparaison sur les mêmes conditions initiales. La validation empirique limitée de 008b n'est pas une garantie de sûreté hors de son domaine.
