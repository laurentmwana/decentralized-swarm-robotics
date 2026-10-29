# Mémoire LaTeX

Le point d’entrée est `main.tex`. La version révisée décrit le simulateur à accélérations et l’apprentissage supervisé par imitation utilisés dans l’application actuelle.

## Organisation

L’introduction générale est suivie de quatre chapitres : notions de base, travaux liés, méthode et implémentation, expériences et résultats. La discussion et la conclusion générale précèdent la bibliographie. Les pages personnelles et les réglages de présentation existants sont conservés.

Chaque chapitre numéroté comporte une introduction et une conclusion partielle. La rédaction privilégie des phrases simples, avec les termes techniques expliqués. Les figures ajoutées sont des tracés scientifiques des données du programme, lisibles en noir et blanc.

Le détail de la reprise, les sources des chiffres et les commandes de reconstruction sont dans [REDACTION.md](REDACTION.md). Les anciens fichiers d’images restent conservés, mais les anciens résultats à commande de vitesse ne sont plus utilisés comme résultats de la version actuelle.

## Compilation

Avec une installation LaTeX disposant de pdfLaTeX et Biber, depuis ce dossier :

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
biber main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Le dossier peut également être ouvert dans l’éditeur LaTeX de Codex pour consulter les sources et l’aperçu. La réussite de la compilation et les contrôles visuels de cette reprise sont consignés dans la tâche 021.

## Vérification des données

Depuis la racine du dépôt :

```powershell
python latex/tools/prepare_current_results.py
python latex/tools/check_manuscript.py
```

La première commande nécessite NumPy et Matplotlib. Les deux commandes utilisent les archives existantes sans entraîner de modèle. Elles ne remplacent pas la lecture du document ni les recommandations du directeur.
