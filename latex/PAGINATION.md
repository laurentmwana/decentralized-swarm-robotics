# Pagination et architecture du mémoire

## Contrainte actuelle

L’auteur demande **65 à 70 pages pour le PDF complet**. La version du 29 octobre 2026 compte **70 pages physiques**, page de garde, pages liminaires, sommaire et bibliographie inclus. Elle remplace la reprise condensée de 45 pages ; la structure de quatre chapitres est conservée.

La mise en page reste en 12 points, interligne 1,5 et marges de 2,5 cm. La longueur est obtenue par des développements de fond, sans ajout de pages blanches ni augmentation de la taille du texte.

## Décompte vérifié

| Partie | Numérotation imprimée | Pages physiques | Nombre |
|---|---|---|---:|
| Page de garde | Sans numéro | 1 | 1 |
| Épigraphe à Abstract | i à viii | 2 à 9 | 8 |
| Table des matières | ix à xii | 10 à 13 | 4 |
| Introduction générale | 1 à 5 | 14 à 18 | 5 |
| 1. Concepts de base | 6 à 12 | 19 à 25 | 7 |
| 2. État de l’art | 13 à 21 | 26 à 34 | 9 |
| 3. Méthodologie et solution proposée | 22 à 35 | 35 à 48 | 14 |
| 4. Implémentation, expérimentations et résultats | 36 à 48 | 49 à 61 | 13 |
| Discussion | 49 à 54 | 62 à 67 | 6 |
| Conclusion générale et perspectives | 55 à 56 | 68 à 69 | 2 |
| Bibliographie | 57 | 70 | 1 |
| **Total** | **13 pages avant le texte + 57 pages arabes** | **1 à 70** | **70** |

Le nombre imprimé sur la dernière page, 57, n’est donc pas le nombre de pages du PDF. Le décompte demandé porte sur les 70 pages du fichier.

## Dossiers et plan à maintenir

Les fichiers restent dans `chapters/` et `frontmatter/`, avec les inclusions existantes de `main.tex`. Aucun dossier de chapitres n’est remplacé ou renommé.

L’introduction générale contient Contexte de l’étude, Problématique, Hypothèses, Objectifs poursuivis, Méthodes et techniques utilisées, Contribution de l’étude et Plan sommaire et subdivision du travail.

Le chapitre 2 respecte les six rubriques du professeur : Introduction, Méthodes traditionnelles, Méthodes avancées, Quelques travaux connexes, Positionnement et Conclusion partielle. Les trois travaux connexes comportent chacun Contexte de l’étude et questions de recherche, Méthodologie adoptée, Aperçu sur les résultats et Critique.

La Discussion et la Conclusion générale et perspectives sont distinctes, non numérotées et présentes dans le sommaire. Elles restent successivement dans `chapters/discussion.tex`, comme dans les sources antérieures. Les quatre chapitres numérotés commencent par Introduction et se terminent par une Conclusion partielle en un paragraphe annonçant la suite.

## Figures, tableaux et mesures

La version courante contient huit figures scientifiques et onze tableaux, avec onze références bibliographiques citées et seize groupes d’équations. Les tableaux ont leur légende au-dessus ; les figures en dessous. Les éléments sont annoncés avant leur insertion.

Les figures proviennent du programme et de ses archives : graphe de voisinage, catalogue, courbes d’entraînement, contacts après correction, états de formation, durées, convergence et navigation. Aucun générateur d’images n’est utilisé. Les anciens fichiers restent conservés pour l’historique mais ne sont pas inclus comme résultats de la version actuelle.

## Contrôles

`tools/check_professor_plan.py` vérifie le maintien des rubriques, les annonces, les légendes et l’organisation des dossiers. `tools/check_manuscript.py` vérifie les références et les chiffres. Les preuves de la version actuelle sont dans `../tasks/evidence/022-professor-plan.json`, `022-manuscript-audit.json` et `022-pdf-review.json`.

Le PDF a été compilé avec pdfLaTeX et Biber. Les 70 pages ont été rendues et examinées. Le journal final ne contient ni débordement, ni renvoi indéfini, ni avertissement de lien du sommaire. Toute modification ultérieure doit être recompilée et son nombre total de pages vérifié avant livraison.
