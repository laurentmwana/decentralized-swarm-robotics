# Rédaction du mémoire — version du 29 octobre 2026

Le fichier principal est `main.tex`. Les pages personnelles et les réglages de présentation existants sont conservés. Les chapitres sont rédigés en français simple ; les termes techniques nécessaires sont expliqués lors de leur introduction.

## Cohérence scientifique

La rédaction décrit la version à accélérations du simulateur et l'imitation supervisée. L'ancienne étude à commande de vitesse, ses 9 600 épisodes et ses figures ne sont pas utilisés pour décrire la version actuelle. Les anciens fichiers d'images restent disponibles pour l'historique.

Trois résultats distincts sont présentés :

- GAT de référence à douze robots : neuf cas réservés, 9/9 avec protection et 0/9 sans filtre pendant le mouvement, 54 contacts.
- GAT structuré v2 : dix-huit cas réservés à 12/24/64 robots, sans contact ni filtre pendant les phases de mouvement apprises. Six départs distincts seulement, trois formes et préparation contrôlée.
- Navigation hybride : huit cas finaux par contrôleur, sans contact ; carré à douze robots, carte connue, planificateur et protection classiques. Le GAT structuré brut n'est pas évalué dans cette campagne.

Les échecs intermédiaires et les limites sont conservés. Le nombre d'observations, le nombre d'épisodes et le nombre de départs différents sont distingués. Les tests du logiciel ne servent pas à annoncer un taux de réussite du modèle.

## Figures issues du programme

Aucune image générée par un modèle d'images n'est ajoutée. Les cinq figures `images/current_*` sont des tracés Matplotlib en noir et blanc : graphe de voisinage, courbes d'entraînement, séquence de formation, décomposition des durées et trajectoires près des obstacles. Les fichiers PDF vectoriels sont intégrés au mémoire ; les PNG permettent une vérification rapide.

Les tableaux de résultats sont reconstruits depuis les rapports JSON. Les données lues et leurs empreintes SHA-256 figurent dans `generated/sources.json`.

Depuis la racine du dépôt :

```powershell
python latex/tools/prepare_current_results.py
python latex/tools/check_manuscript.py
```

Le premier script nécessite NumPy et Matplotlib. Il ne lance aucun entraînement. Le second utilise uniquement la bibliothèque standard et vérifie notamment citations, renvois, figures, empreintes des sources et des poids, ainsi que les principaux chiffres rapportés.

## Sources des résultats

| Mention dans le texte | Rapport conservé à la racine du dépôt |
|---|---|
| 008d — référence multi-formes | `tasks/evidence/008d-local-target-expanded-test.json` |
| 011 — correction | `tasks/evidence/011-fresh-evaluation.json` |
| 012 — pilote multi-effectifs | `tasks/evidence/012-multisize-evaluation.json` |
| 014 — modèle structuré | `tasks/evidence/014-vector-fresh-test.json` |
| 014 — audit des trajectoires | `tasks/evidence/014-vector-final-audit.json` |
| 019 — navigation finale | `tasks/evidence/019-near-targets-evaluation.json` |
| 019 — audit de navigation | `tasks/evidence/019-near-targets-audit.json` |

Les journaux d'entraînement utilisés dans les courbes sont `outputs/imitation/training-multiform-expanded-v1/run.json` et `outputs/imitation/training-vector-gat-v2/run.json`. Les trajectoires illustrées sont enregistrées dans le manifeste de sources.

## Sauvegarde et prochaine relecture

Une copie des sources avant cette reprise se trouve dans `outputs/thesis-redaction/source-before/`. Cette version est une rédaction fondée sur les résultats disponibles. Les consignes supplémentaires du directeur et les corrections personnelles pourront ensuite être appliquées sans changer les conclusions expérimentales.

## Rendu vérifié

La version du 29 octobre 2026 compile avec pdfLaTeX et Biber en 45 pages. Elle contient cinq figures scientifiques, sept tableaux et dix références citées. Les renvois sont résolus et aucun débordement n’est signalé. Les pages rendues ont été examinées. Le PDF à relire est `output/pdf/memoire-swarm-gnn-redaction.pdf` depuis la racine du dépôt ; `latex/main.pdf` contient le même document.
