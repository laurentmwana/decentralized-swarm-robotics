# Pagination et architecture du mémoire

La révision 025 compte **66 pages physiques**, dont 13 pages avant l’introduction et 53 pages numérotées en chiffres arabes. Les paramètres restent de 12 points, un interligne de 1,5 et des marges de 2,5 cm.

| Partie | Numérotation arabe | Pages physiques | Nombre |
|---|---|---|---:|
| Garde, pages liminaires et sommaire | — | 1 à 13 | 13 |
| Introduction générale | 1 à 2 | 14 à 15 | 2 |
| Concepts de base | 3 à 9 | 16 à 22 | 7 |
| État de l’art | 10 à 16 | 23 à 29 | 7 |
| Méthodologie et solution proposée | 17 à 30 | 30 à 43 | 14 |
| Implémentation, expérimentations et résultats | 31 à 44 | 44 à 57 | 14 |
| Discussion | 45 à 50 | 58 à 63 | 6 |
| Conclusion générale et perspectives | 51 à 52 | 64 à 65 | 2 |
| Bibliographie | 53 | 66 | 1 |

Les dossiers `chapters/` et `frontmatter/` sont conservés. L’introduction possède sept rubriques numérotées de 1 à 7. Son dernier titre est uniquement « Plan sommaire ». La discussion et la conclusion restent distinctes.

Le texte conserve six blocs d’équations utiles, huit figures et quatorze tableaux. Les figures en couleur utilisent aussi des traits et des marqueurs distincts ; les barres sont hachurées. Les aperçus en niveaux de gris ont été examinés. Les exports sont des PDF vectoriels et des PNG à 300 dpi.

Les preuves de cette étape sont dans `../tasks/evidence/025-plan.json`, `025-manuscript-audit.json` et `025-pdf-review.json`. Les preuves antérieures décrivent leurs versions historiques. La demande suivante de l’auteur fixe une nouvelle cible de 70 à 78 pages, à traiter dans la révision 026.
