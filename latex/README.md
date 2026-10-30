# Mémoire LaTeX

Le point d’entrée est `main.tex`. La version révisée décrit le simulateur à accélérations et l’apprentissage supervisé par imitation utilisés dans l’application actuelle.

## Organisation à conserver

Les dossiers `chapters/` et `frontmatter/` sont conservés. Le plan suit [RULES.md](RULES.md) :

1. Introduction générale avec ses sept rubriques.
2. Concepts de base.
3. État de l’art : méthodes traditionnelles, méthodes avancées, trois travaux connexes en tableaux, positionnement.
4. Méthodologie et solution proposée.
5. Implémentation, expérimentations et résultats.
6. Discussion, puis Conclusion générale et perspectives distinctes.
7. Bibliographie.

Il y a quatre chapitres numérotés. Chaque chapitre s’ouvre par Introduction et se ferme par une Conclusion partielle qui annonce la suite. La discussion et la conclusion générale sont contenues successivement dans `chapters/discussion.tex`, comme dans l’architecture initiale.

La version courante compte **74 pages au total**, dont 13 avant l’introduction et 61 numérotées en chiffres arabes. La cible demandée est de 70 à 78 pages physiques. La répartition est donnée dans [PAGINATION.md](PAGINATION.md). Police 12 points, interligne 1,5 et marges de 2,5 cm conservés.

Les développements restent fondés sur les résultats de la version à accélérations. La rédaction privilégie des phrases simples. Les neuf figures scientifiques sont calculées à partir du programme ou des données enregistrées ; aucune image générée par IA n’est utilisée.

Voir [REDACTION.md](REDACTION.md) pour la provenance des chiffres. Les anciens résultats à commande de vitesse ne sont pas réintroduits.

## Compilation

Avec une installation LaTeX disposant de pdfLaTeX et Biber, depuis ce dossier :

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Le dossier peut également être ouvert dans l’éditeur LaTeX de Codex pour consulter les sources et l’aperçu. La réussite de la compilation et les contrôles visuels de cette reprise sont consignés dans la tâche 026.

## Vérification des données

Depuis la racine du dépôt :

```powershell
python latex/tools/prepare_current_results.py
python latex/tools/check_manuscript.py
python latex/tools/check_professor_plan.py
```

La première commande nécessite NumPy et Matplotlib. Les trois commandes utilisent les archives existantes sans entraîner de modèle. Elles ne remplacent pas la lecture du document ni les recommandations du directeur.

Préférence de lecture de l’auteur : introduction courte, plan énuméré, notions distinguées sous les titres composés. Les quatre rubriques de chaque travail connexe sont conservées comme lignes de tableau, selon la demande de l’auteur.

Révision 026 : introduction de deux pages, rubriques numérotées de 1 à 7 et titre « Plan sommaire ». La liste contient vingt abréviations utilisées dans le texte. Le document conserve six blocs d’équations, des figures en couleur lisibles en niveaux de gris et les titres liminaires centrés. Les développements portent sur l’architecture, les observations, la provenance des fichiers et un cas réservé déjà enregistré ; aucun résultat expérimental supplémentaire n’est annoncé.

Révision 027 : trois annexes sont ajoutées après la bibliographie et inscrites au sommaire. Elles donnent le guide de démarrage, la fiche des sept expériences et les fichiers à conserver pour vérifier les résultats. Elles occupent les pages arabes 59 à 61. Le PDF complet compte 74 pages ; les preuves courantes sont les preuves 027.

Révision 030 : reformulation ciblée des extraits fournis par l’auteur et titres des annexes simplifiés. Le relevé avant/après est dans les preuves 030. Les résultats et le volume de 74 pages sont conservés.
