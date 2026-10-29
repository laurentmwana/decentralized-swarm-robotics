# 010 — Interface, exports et comparaisons finales

Statut : implémentée ; 423 tests ciblés réussis. Vérification interactive du calcul apparié, de la lecture, d’une transition continue et d’une destination obstruée conservée en échec. Audit de six trajectoires exportées passé.

## Livraison

- Interface locale française sur `http://127.0.0.1:8765`, lanceur Windows et commande Python.
- 56 formes ; choix de 8, 12, 16 ou 24 robots, avec effectifs hors apprentissage signalés.
- Formation, transition vers une seconde forme, transport, contournement fourni et cas difficiles conservés.
- Professeur classique, GAT protégé ou comparaison des deux sur le même départ.
- Calcul puis lecture avec pause, remise au début, curseur et vitesse ; communications, trajectoires, places et numéros.
- Mesures finales, réussite/échec, raisons de navigation ; temps réussi absent en cas d’échec.
- ZIP avec configuration et métriques JSON, trajectoires CSV, états et traces NPZ ; aucun entraînement ni chemin de modèle libre depuis le navigateur.
- Tableau des campagnes conservées, taux avec dénominateurs et rapport CSV/JSON régénérable à partir des preuves.

## Vérification

Les tests comprennent un calcul physique réel, l’égalité des états exportés JSON/NPZ/CSV, l’intégrité du ZIP, les paramètres reproductibles, le cycle des calculs HTTP, les paramètres refusés et les cas difficiles conservés dans les comparaisons finales.

Les expériences interactives restent des démonstrations distinctes des campagnes de validation. Aucun poids ni hyperparamètre d’apprentissage n’est modifié. Les limites scientifiques restantes sont décrites dans les décisions 23–26 ; l’échec du GAT brut reste explicite.

Décision : [26-interface-demonstration](../docs/decisions/26-interface-demonstration.md). Rapport final : `tasks/evidence/010-final-comparison.json`. Audit des exports : `tasks/evidence/010-demo-export-audit.json`.
