# Mémoire LaTeX

Le point d'entrée est `main.tex`. Il inclut les chapitres depuis `chapters/` et les références depuis `references.bib`.

La compilation nécessite LaTeX et Biber. Les figures du protocole pilote sont référencées directement depuis `../results/`; elles doivent donc avoir été générées avec `python -m src.eval.run` avant la compilation.

## État actuel

Le document contient un squelette compilable, une bibliographie initiale et une rédaction de travail fondée sur les résultats pilotes. Les chiffres pilotes sont clairement signalés comme tels ; ils devront être actualisés après l'étude complète à 50 graines.
