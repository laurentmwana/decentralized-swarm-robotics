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

La version courante compte **65 pages au total**, dont 13 avant l’introduction et 52 numérotées en chiffres arabes. La cible demandée est de 65 à 70 pages physiques. La répartition est donnée dans [PAGINATION.md](PAGINATION.md). Police 12 points, interligne 1,5 et marges de 2,5 cm conservés.

Les développements restent fondés sur les résultats de la version à accélérations. La rédaction privilégie des phrases simples. Les huit figures scientifiques sont calculées à partir du programme ou des données enregistrées ; aucune image générée par IA n’est utilisée.

Voir [REDACTION.md](REDACTION.md) pour la provenance des chiffres. Les anciens résultats à commande de vitesse ne sont pas réintroduits.

## Compilation

Avec une installation LaTeX disposant de pdfLaTeX et Biber, depuis ce dossier :

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Le dossier peut également être ouvert dans l’éditeur LaTeX de Codex pour consulter les sources et l’aperçu. La réussite de la compilation et les contrôles visuels de cette reprise sont consignés dans la tâche 023.

## Vérification des données

Depuis la racine du dépôt :

```powershell
python latex/tools/prepare_current_results.py
python latex/tools/check_manuscript.py
python latex/tools/check_professor_plan.py
```

La première commande nécessite NumPy et Matplotlib. Les deux commandes utilisent les archives existantes sans entraîner de modèle. Elles ne remplacent pas la lecture du document ni les recommandations du directeur.

Préférence de lecture de l’auteur : introduction courte, plan énuméré, notions distinguées sous les titres composés. Les quatre rubriques de chaque travail connexe sont conservées comme lignes de tableau, selon la demande de l’auteur.

La révision 024 privilégie les phrases directes sans deux-points explicatifs au milieu des phrases. Les titres des pages liminaires sont centrés ; les paragraphes restent justifiés et le style des chapitres reprend dès l’introduction générale. Le PDF conserve 65 pages. Les preuves 024 décrivent cette version courante.
