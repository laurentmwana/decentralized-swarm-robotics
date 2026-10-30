# Pagination et architecture du mémoire

La révision 026 compte **71 pages physiques**, dont 13 pages avant l’introduction et 58 pages numérotées en chiffres arabes. Les paramètres restent de 12 points, un interligne de 1,5 et des marges de 2,5 cm.

| Partie | Numérotation arabe | Pages physiques | Nombre |
|---|---|---|---:|
| Garde, pages liminaires et sommaire | — | 1 à 13 | 13 |
| Introduction générale | 1 à 2 | 14 à 15 | 2 |
| Concepts de base | 3 à 10 | 16 à 23 | 8 |
| État de l’art | 11 à 17 | 24 à 30 | 7 |
| Méthodologie et solution proposée | 18 à 34 | 31 à 47 | 17 |
| Implémentation, expérimentations et résultats | 35 à 49 | 48 à 62 | 15 |
| Discussion | 50 à 55 | 63 à 68 | 6 |
| Conclusion générale et perspectives | 56 à 57 | 69 à 70 | 2 |
| Bibliographie | 58 | 71 | 1 |

Les dossiers `chapters/` et `frontmatter/` sont conservés. L’introduction possède sept rubriques numérotées de 1 à 7. Son dernier titre est uniquement « Plan sommaire ». La discussion et la conclusion restent distinctes.

Le texte conserve six blocs d’équations utiles, neuf figures et dix-sept tableaux. Les figures en couleur utilisent aussi des traits et des marqueurs distincts ; les barres sont hachurées. Les aperçus en niveaux de gris ont été examinés. Les exports sont des PDF vectoriels et des PNG à 300 dpi.

La cible actuelle est de **70 à 78 pages physiques**, introduction comprise. L’introduction de deux pages respecte la demande de deux à quatre pages. La liste contient vingt abréviations utilisées dans le texte.

Les preuves sont `../tasks/evidence/026-plan.json`, `026-manuscript-audit.json` et `026-pdf-review.json`. Les preuves antérieures décrivent leurs versions historiques.
