# Mémoire LaTeX

Le point d'entrée est `main.tex`. Il inclut les chapitres depuis `chapters/` et les références depuis `references.bib`.

Structure à cinq chapitres :

1. `chapters/introduction.tex`
2. `chapters/etat-art.tex`
3. `chapters/methode.tex` — méthodologie **et** architecture logicielle
4. `chapters/experiences.tex`
5. `chapters/discussion.tex` — discussion **et** conclusion

La compilation nécessite pdflatex et Biber :

```bash
cd latex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Illustrations

- `images/` — jeu de figures unique du mémoire (29 PNG, rendu matplotlib 300 dpi, police Latin Modern, nommés par chapitre, `fig_extra_*` pour les figures transversales), dont 27 référencés dans les chapitres. Les 4 figures de données (perte et configuration du cercle, pertes et planche du curriculum) sont régénérées depuis les résultats versionnés par `python scripts/generate_data_figures.py`.
- L'ancien répertoire `figures/` a été supprimé (18 octobre 2026) ; `python scripts/generate_thesis_figures.py` n'est plus rattaché à aucune figure du PDF (sources historiques conservées pour traçabilité).

## État actuel

Le document compile sans erreur à **78 pages** (18 octobre 2026) : 5 chapitres, 27 figures, 12 tableaux, 35 références citées sur 37 (`references.bib`), aucune figure manquante ni référence indéfinie. La page de titre reste à compléter (nom de l'auteur, directeur de mémoire, établissement) : le LaTeX signale « No \author given ».

Les chiffres pilote sont clairement signalés comme tels ; ils devront être actualisés après l'étude complète à 50 graines.
