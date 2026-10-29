# Pagination et architecture du mémoire

La cible reste de **65 à 70 pages physiques**, tout compris. Après la révision de lisibilité demandée par l’auteur, le PDF compte **65 pages**, dont 13 avant l’introduction et 52 en numérotation arabe. La mise en page reste en 12 points, interligne 1,5 et marges de 2,5 cm.

| Partie | Numérotation imprimée | Pages physiques | Nombre |
|---|---|---|---:|
| Garde, pages liminaires et sommaire | Sans numéro puis i à xii | 1 à 13 | 13 |
| Introduction générale | 1 à 2 | 14 à 15 | 2 |
| 1. Concepts de base | 3 à 9 | 16 à 22 | 7 |
| 2. État de l’art | 10 à 16 | 23 à 29 | 7 |
| 3. Méthodologie et solution proposée | 17 à 30 | 30 à 43 | 14 |
| 4. Implémentation, expérimentations et résultats | 31 à 43 | 44 à 56 | 13 |
| Discussion | 44 à 49 | 57 à 62 | 6 |
| Conclusion générale et perspectives | 50 à 51 | 63 à 64 | 2 |
| Bibliographie | 52 | 65 | 1 |
| **Total** | | **1 à 65** | **65** |

Les dossiers `chapters/` et `frontmatter/` et les inclusions de `main.tex` sont conservés. Les sept rubriques de l’introduction, les quatre chapitres et la séparation de Discussion et Conclusion générale restent en place.

L’introduction est concise et le plan est énuméré. Chaque notion du titre « Essaim, localité, décentralisation » possède son sous-titre. Les autres titres sont décomposés ou recentrés sur une idée.

À la demande de l’auteur, les trois travaux connexes sont présentés dans trois tableaux. Chacun garde contexte, méthode, résultats et critique. Cette présentation remplace les sous-sous-sections de la version précédente tout en conservant le fond du plan du professeur.

Le document comporte huit figures scientifiques, quatorze tableaux et onze références citées. Les tableaux ont leur légende au-dessus ; les figures en dessous. Les illustrations et les résultats expérimentaux sont inchangés par cette révision de lecture.

La compilation pdfLaTeX/Biber et la relecture des pages sont terminées. Les contrôles et le décompte sont enregistrés dans `../tasks/evidence/023-lecture-plan.json`, `023-manuscript-audit.json` et `023-pdf-review.json`. Les preuves antérieures décrivent leurs versions historiques. Toute modification ultérieure doit être recompilée et contrôlée avant livraison.
