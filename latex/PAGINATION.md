# Cible de pagination du mémoire

La version compilée du 18 octobre 2026 compte **78 pages** dans une structure à **cinq chapitres**, avec 27 figures et 12 tableaux.

Décompte relevé dans `main.toc` :

| Partie | Pages |
|---|---:|
| Pages préliminaires (titre, résumés, listes) | 1 à 9 |
| 1. Introduction | 10 à 15 |
| 2. État de l'art | 16 à 29 |
| 3. Méthode et architecture logicielle | 30 à 50 |
| 4. Expériences et résultats | 51 à 67 |
| 5. Discussion et conclusion | 68 à 75 |
| Bibliographie (35 entrées citées sur 37 dans `references.bib`) | 76 à 78 |

## Organisation à cinq chapitres

- Le chapitre 3 fusionne l'ancienne partie méthodologique et l'ancien chapitre d'architecture logicielle : les sections d'implémentation y sont devenues des sous-sections.
- Le chapitre 5 fusionne l'ancienne discussion et l'ancienne conclusion en une seule marche argumentative, terminée par une section « Conclusion et perspectives ».

## Illustrations intégrées

27 figures, toutes citées et analysées dans le texte, toutes issues de `latex/images/` dans la même charte (Latin Modern, matplotlib 300 dpi, sans titre interne) : 23 illustrations et schémas du jeu livré, plus les 4 figures de données (perte et configuration du cercle, pertes et planche du curriculum) régénérées depuis les résultats versionnés par `python scripts/generate_data_figures.py`. L'ancien répertoire `latex/figures/` a été supprimé le 18 octobre 2026. Deux fichiers de `images/` restent inutilisés : les planches du jeu de formes (`fig_extra_planche_formes_20pts.png` et `_50pts.png`), retirées du document à la demande de l'auteur.

Chapitre 1 (2 figures) :
- `fig_1_1_chaine_experimentale.png` — chaîne expérimentale de haut niveau.
- `fig_extra_schema_formation.png` — illustration conceptuelle d'une formation (départs aléatoires → formation approchée).

Chapitre 2 (4 figures) :
- `fig_2_1_consensus.png` — convergence d'un consensus sur un graphe linéaire.
- `fig_2_2_flocking.png` — cohésion, séparation et alignement sur un essaim miniature.
- `fig_2_3_agregation.png` — principe de l'agrégation moyenne des messages.
- `fig_2_4_portee_information.png` — portée de l'information selon le nombre de tours.

Chapitre 3 (9 figures) :
- `fig_3_1_essaim_communication.png` — voisinage de communication.
- `fig_3_2_chaine_preparation_cible.png` — image → masque → nuage de points.
- `fig_3_3_controleur_local.png` — bloc de calcul local du contrôleur GNN.
- `fig_3_4_distance_chamfer.png` — lecture géométrique de la distance de Chamfer.
- `fig_3_5_composition_perte.png` — composition des quatre termes de la perte.
- `fig_3_6_curriculum_reduit.png` — progression du curriculum réduit (vraies cibles).
- `fig_3_7_retropropagation.png` — rétropropagation à travers les pas de simulation.
- `fig_3_8_modules_prototype.png` — organisation en modules du prototype.
- `fig_3_9_cycle_episode.png` — cycle de vie d'un épisode.

Chapitre 4 (10 figures) :
- `fig_4_1_perte_cercle.png`, `fig_4_3_configuration_cercle.png` (données) et `fig_4_2_synthese_pilote.png` — entraînement du cercle et synthèse du pilote.
- `fig_4_4_chamfer_vs_pannes.png` — Chamfer selon la fraction de robots retirés (GNN et règle de regroupement).
- `fig_4_5_chronologie_retrait.png` — chronologie d'un retrait à mi-épisode.
- `fig_4_6_passage_echelle.png` — passage à l'échelle du pilote.
- `fig_4_7_schema_convergence.png` — schéma de la mesure de convergence.
- `fig_4_x_dispersion_graines.png` — dispersion des dix graines du pilote, seuil et moyenne.
- `fig_4_x_curriculum_pertes_par_cible.png` — lecture visuelle des pertes par cible et par étape.
- `fig_4_10_pertes_curriculum.png`, `fig_4_11_planche_curriculum.png` (données) — pertes et planche du curriculum.

Chapitre 5 (2 figures) :
- `fig_extra_matrice_comparaison.png` — matrice des contrôleurs comparés ou prévus, avec leur statut.
