# Cible de pagination du mémoire

La version compilée du 27 octobre 2026 compte **78 pages** dans une structure à **quatre chapitres numérotés**, avec 17 figures, 13 tableaux et 17 entrées bibliographiques.

Décompte relevé dans `main.toc` (numérotation arabe, après les 8 pages liminaires) :

| Partie | Pages |
|---|---:|
| Page de titre (non numérotée) + pages liminaires (épigraphe, dédicace, remerciements, abréviations, listes, résumés) | i à viii |
| Table des matières | ix à xii |
| Introduction générale | 1 à 7 |
| 1. Concepts de base | 8 à 12 |
| 2. État de l'art | 13 à 23 |
| 3. Méthodologie et solution proposée | 24 à 37 |
| 4. Implémentation, expérimentations et résultats | 38 à 55 |
| Discussion | 56 à 61 |
| Conclusion générale et perspectives | 62 à 63 |
| Bibliographie (17 entrées) | 64 à 65 |

Soit 13 pages liminaires et 65 pages numérotées en chiffres arabes, pour 78 pages physiques.

Structure conforme aux consignes du directeur : chaque chapitre numéroté s'ouvre sur une section *Introduction* et se clôt par une *Conclusion partielle* qui résume le chapitre en un paragraphe et annonce le suivant. Les pages liminaires suivent l'ordre imposé : Épigraphe, Dédicace, Remerciements (les deux signés du nom complet), Liste des abréviations (GNN, LiDAR, ReLU, par ordre alphabétique), Liste des figures, Liste des tableaux, Résumé (8 mots clés), Abstract (8 keywords). L'ordre des chapitres reprend le plan exigé : concepts de base, état de l'art, méthodologie, implémentation et résultats.

## Chapitres

- Le chapitre 1 *Concepts de base* a été ajouté le 27 octobre 2026 ; il était déjà annoncé par le « Plan sommaire » de l'introduction générale mais n'existait pas. Il ne contient aucune valeur de paramètre, et un seul visuel, la Figure 1.1 (un réseau de neurones en couches d'un côté, un voisinage de robots à portée de l'autre), qui est un schéma conceptuel sans résultat d'expérience. Elle ne porte aucun titre interne, seulement les repères (a) et (b), cinq mots de légende et les lettres des nœuds.
- Le chapitre 2 *État de l'art* suit le plan imposé : Introduction, Méthodes traditionnelles, Méthodes avancées, Quelques travaux connexes (trois études, chacune en Contexte de l'étude, Méthodologie adoptée, Aperçu sur les résultats, Critique), Positionnement, Conclusion partielle. La section *Quelques travaux connexes* a été retirée le 27 octobre 2026 puis rétablie le même jour, sur demande de l'auteur : elle est exigée par la consigne 16 du directeur et porte la critique sur laquelle s'appuie le Positionnement.
- La *Discussion* et la *Conclusion générale et perspectives* restent des chapitres non numérotés après le chapitre 4, comme l'annoncent la consigne 16 du directeur et le « Plan sommaire » de l'introduction générale (« le mémoire compte quatre chapitres »). Leurs flottants suivent la convention déjà utilisée pour l'introduction générale (figures `I.n`) : la Discussion porte le préfixe `D.n`, la conclusion `C.n`. Le bloc `D.n` est posé dans `main.tex` avant `\include{chapters/discussion}`, le bloc `C.n` dans `chapters/discussion.tex` juste avant le chapitre de conclusion. Les sections de la Discussion, non numérotées, apparaissent dans la table des matières grâce à `\addcontentsline`, comme celles de l'introduction générale ; la numérotation continue donc sans trou apparent entre 4.18 et la bibliographie. Les anciens `\label{chap:discussion}` et `\label{chap:conclusion}`, qui résolvaient au chapitre 4, ont été supprimés ; aucun `\ref` ne les utilisait.

## Illustrations

17 figures dans le manuscrit, toutes annoncées dans le texte avant leur flottant, toutes en `width=\linewidth`, produites par `generate_manuscript_figures.py` et `generate_scientific_figures.py` (matplotlib 300 dpi, Latin Modern, aucun titre interne). Aucune figure n'est empruntée à un article, donc aucune légende n'appelle de citation.

Chapitre 1 (1) : `neuron_and_message_passing.png` (Figure 1.1).
Introduction (2) : `formation_principle.png`, `experimental_chain.png` (numérotées I.1 et I.2).
Chapitre 3 (7) : `swarm_communication_graph.png`, `target_preparation_pipeline.png`, `gnn_current_architecture.png`, `shape_loss_geometry.png`, `differentiable_training_flow.png`, `success_criterion_comparison.png`, `software_modules.png`.
Chapitre 4 (7) : `historical_pilot_summary.png`, `historical_chamfer_vs_failures.png`, `failure_timeline.png`, `historical_scaling.png`, `threshold_crossing_schema.png`, `historical_seed_dispersion.png`, `historical_curriculum_per_target.png`.

13 tableaux : 1 au chapitre 2 (synthèse comparative des familles), 3 au chapitre 3 (notations, hyperparamètres, curriculum), 8 au chapitre 4, 1 en Discussion (`D.1`, matrice des contrôleurs), les légendes sont placées au-dessus du tableau.

`latex/images/` contient 21 PNG : 17 figures du manuscrit, `logo_unikin.png` pour la page de garde, et 3 visuels sans usage (`controller_information_matrix.png`, `message_aggregation.png`, `activation_functions.png`). Les diapositives de `../presentation/soutenance-swarm-gnn.tex` chargent leurs propres copies dans `presentation/images/`. Les 4 PNG sans usage (règles de regroupement, consensus, portée de l'information, fonctions d'activation) avaient été supprimés le 27 octobre 2026 ; `activation_functions.png` est réapparu avec la Figure 1.1, car le script le régénère à chaque exécution.

**Attention, à lire avant de relancer un générateur.** Les six figures de `generate_scientific_figures.py` telles qu'elles sont commitées n'ont pas les dimensions que ce script produit (`gnn_current_architecture.png` commité : 2007×861, soit 2,33 de rapport ; le canevas du script : 3150×2790, rapport 1,13). Lancer le script entier remplace donc ces six visuels par d'autres rendus. Pour n'ajouter qu'une figure, appeler le constructeur ciblé :
`.venv/Scripts/python.exe -c "import sys; sys.path.insert(0,'latex'); import generate_scientific_figures as g; g.save_figure(g.neuron_and_message_passing(), 'neuron_and_message_passing')"`
Le script écrit aussi des `.pdf` à côté des `.png` ; le manuscrit n'inclut que des `.png`.

## Fichiers morts

`chapters/preliminaries.tex` n'est pas inclus par `main.tex` ; il double le Résumé et l'Abstract de `frontmatter/sumaires.tex`. `page_de_garde.tex` est un modèle externe (autre auteur), conservé pour référence de mise en page.
