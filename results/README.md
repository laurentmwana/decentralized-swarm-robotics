# Résultats

Ce dossier reçoit les animations, tableaux et sorties générés par les expériences. Seuls les courts résumés utiles au suivi sont versionnés ; les sorties volumineuses sont ignorées par Git.

## Validation de l'étape 2

La commande `python -m src.sim.demo` génère `simulator-gathering.gif` à partir de `configs/simulator.yaml`. La démonstration comporte 20 robots, 100 pas de temps et une règle locale de cohésion-répulsion. Elle permet de visualiser le comportement de base avant l'introduction du modèle GNN.

## Premier entraînement du cercle

La commande `python -m src.train.run`, avec `configs/training-circle.yaml`, entraîne le GNN pendant 80 itérations et sauvegarde le modèle, la configuration et la courbe dans `results/training-circle/`. Pour la graine 41, la perte est passée de 9,84 à 0,28. La commande `python -m src.train.visualize` génère l'animation et l'image finale de la formation du cercle.

## Curriculum réduit

La commande `python -m src.train.curriculum` entraîne successivement les niveaux géométrie, chiffres, lettres et nombres. Avec la configuration actuelle, les pertes finales par niveau sont respectivement 0,318, 0,248, 0,205 et 0,156. La planche `curriculum/numbers/contact-sheet.png` sert de contrôle visuel : elle atteste une couverture approximative des cibles, sans constituer une évaluation finale de lisibilité.
