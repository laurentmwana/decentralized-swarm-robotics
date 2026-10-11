# Résultats

Ce dossier reçoit les animations, tableaux et sorties générés par les expériences. Seuls les courts résumés utiles au suivi sont versionnés ; les sorties volumineuses sont ignorées par Git.

## Validation de l'étape 2

La commande `python -m src.sim.demo` génère `simulator-gathering.gif` à partir de `configs/simulator.yaml`. La démonstration comporte 20 robots, 100 pas de temps et une règle locale de cohésion-répulsion. Elle permet de visualiser le comportement de base avant l'introduction du modèle GNN.
