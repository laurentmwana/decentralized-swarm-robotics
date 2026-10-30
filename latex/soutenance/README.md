# Présentation de soutenance

Vingt diapositives en 16:9 : dix-huit pour la présentation et deux compléments pour les questions. Même contenu scientifique que le PowerPoint, avec une mise en page Beamer adaptée.

Compiler deux fois depuis ce dossier :

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Les figures PDF sont dans `images/`. Les explications orales et les sources sont dans `NOTES.md` et les notes des diapositives. `slides.json` conserve le contenu commun. Le support s’appuie sur les résultats du mémoire ; aucune nouvelle expérience n’a été exécutée.
