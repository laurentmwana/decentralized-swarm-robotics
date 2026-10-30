# Rédaction du mémoire — lecture allégée, 65 pages

Le fichier principal est `main.tex`. Les pages personnelles et les réglages de présentation existants sont conservés. Les chapitres sont rédigés en français simple ; les termes techniques nécessaires sont expliqués lors de leur introduction.

## Cohérence scientifique

La rédaction décrit la version à accélérations du simulateur et l'imitation supervisée. L'ancienne étude à commande de vitesse, ses 9 600 épisodes et ses figures ne sont pas utilisés pour décrire la version actuelle. Les anciens fichiers d'images restent disponibles pour l'historique.

Trois résultats distincts sont présentés :

- GAT de référence à douze robots : neuf cas réservés, 9/9 avec protection et 0/9 sans filtre pendant le mouvement, 54 contacts.
- GAT structuré v2 : dix-huit cas réservés à 12/24/64 robots, sans contact ni filtre pendant les phases de mouvement apprises. Six départs distincts seulement, trois formes et préparation contrôlée.
- Navigation hybride : huit cas finaux par contrôleur, sans contact ; carré à douze robots, carte connue, planificateur et protection classiques. Le GAT structuré brut n'est pas évalué dans cette campagne.

Les échecs intermédiaires et les limites sont conservés. Le nombre d'observations, le nombre d'épisodes et le nombre de départs différents sont distingués. Les tests du logiciel ne servent pas à annoncer un taux de réussite du modèle.

## Figures issues du programme

Aucune image générée par un modèle d'images n'est ajoutée. Les huit figures `images/current_*` sont des tracés Matplotlib en noir et blanc : graphe de voisinage, catalogue, courbes d'entraînement, contacts après correction, séquence de formation, décomposition des durées, convergence et trajectoires près des obstacles. Les fichiers PDF vectoriels sont intégrés au mémoire ; les PNG permettent une vérification rapide.

Les tableaux de résultats sont reconstruits depuis les rapports JSON. Les données lues et leurs empreintes SHA-256 figurent dans `generated/sources.json`.

Depuis la racine du dépôt :

```powershell
python latex/tools/prepare_current_results.py
python latex/tools/check_manuscript.py
python latex/tools/check_professor_plan.py
```

Le premier script nécessite NumPy et Matplotlib. Il ne lance aucun entraînement. Les scripts de contrôle utilisent uniquement la bibliothèque standard et vérifient notamment les citations, les renvois, les figures, les empreintes des sources et des poids, ainsi que les principaux chiffres rapportés.

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

## Plan et rendu vérifiés

La correction 022 rétablit les rubriques de `RULES.md`. Les dossiers `chapters/` et `frontmatter/` et les quatre chapitres numérotés sont conservés. Les trois travaux connexes conservent les quatre rubriques sous forme de tableaux, à la demande de l’auteur. Discussion et Conclusion générale et perspectives sont séparées.

La version courante compile en **65 pages au total**, avec huit figures scientifiques, quatorze tableaux et onze références citées. La table des matières et les légendes sont rétablies selon le plan. La répartition est donnée dans `PAGINATION.md`.

Le PDF à relire est `output/pdf/memoire-swarm-gnn-redaction.pdf` depuis la racine du dépôt ; `latex/main.pdf` contient le même document. Une sauvegarde locale de la version intermédiaire de 45 pages se trouve dans `outputs/thesis-redaction/version-45-pages/`. Les preuves 021 restent celles de cette version intermédiaire ; les preuves 022 décrivent la version historique de 70 pages.

La révision 023 réduit l’introduction à deux pages et présente le plan par une phrase suivie d’une énumération. Les preuves 023 décrivent la révision de lecture allégée ; les preuves 022 restent celles de la version précédente de 70 pages.

La révision 024 privilégie les phrases directes sans deux-points explicatifs au milieu des phrases. Les titres des pages liminaires sont centrés ; les paragraphes restent justifiés et le style des chapitres reprend dès l’introduction générale. Le PDF conserve 65 pages. Les preuves 024 décrivent cette version courante.
