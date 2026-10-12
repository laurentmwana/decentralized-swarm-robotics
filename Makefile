# Raccourcis d'exécution du projet. Sous Windows, installer GNU Make puis lancer
# `make help` depuis PowerShell à la racine du dépôt.

PYTHON ?= python

.PHONY: help test shapes shapes-complex simulator model circle circle-viz curriculum curriculum-viz complex-smoke complex complex-viz evaluation clean-results

help:
	@echo Commandes disponibles :
	@echo   make test             - execute tous les tests unitaires
	@echo   make shapes           - genere le jeu de formes standard (20 points)
	@echo   make shapes-complex   - genere le jeu de formes complexe (50 points)
	@echo   make simulator        - exporte l'animation de la regle de regroupement
	@echo   make model            - verifie le GNN non entraine dans le simulateur
	@echo   make circle           - entraine le GNN sur le cercle
	@echo   make circle-viz       - exporte le GIF et l'image du cercle
	@echo   make curriculum       - execute le curriculum reduit
	@echo   make curriculum-viz   - exporte sa planche de visualisation
	@echo   make complex-smoke    - verifie rapidement le pipeline complexe
	@echo   make complex          - lance la campagne complexe longue (CPU : plusieurs dizaines de minutes)
	@echo   make complex-viz      - exporte la planche de la campagne complexe
	@echo   make evaluation       - lance pannes et passage a l-echelle du pilote

test:
	$(PYTHON) -m pytest tests -q -p no:cacheprovider

shapes:
	$(PYTHON) -m src.shapes.build --config configs/shapes.yaml --output data/shapes

shapes-complex:
	$(PYTHON) -m src.shapes.build --config configs/shapes-complex.yaml --output data/shapes-complex

simulator:
	$(PYTHON) -m src.sim.demo --config configs/simulator.yaml --output results/simulator-gathering.gif

model:
	$(PYTHON) -m src.models.demo --config configs/simulator.yaml

circle: shapes
	$(PYTHON) -m src.train.run --config configs/training-circle.yaml --output results/training-circle

circle-viz: circle
	$(PYTHON) -m src.train.visualize --config configs/training-circle.yaml --model results/training-circle/circle-best.pt --target data/shapes/geometry/geometry_circle.npy --output results/training-circle

curriculum: shapes circle
	$(PYTHON) -m src.train.curriculum --config configs/curriculum.yaml --initial-model results/training-circle/circle-best.pt --output results/curriculum

curriculum-viz: curriculum
	$(PYTHON) -m src.train.visualize_curriculum --config configs/curriculum.yaml --model results/curriculum/numbers/model.pt --output results/curriculum/numbers/contact-sheet.png

complex-smoke: shapes-complex
	$(PYTHON) -m src.train.curriculum --config configs/curriculum-complex-smoke.yaml --initial-model none --output results/curriculum-complex-smoke

complex: shapes-complex
	$(PYTHON) -m src.train.curriculum --config configs/curriculum-complex.yaml --initial-model none --output results/curriculum-complex

complex-viz:
	$(PYTHON) -m src.train.visualize_curriculum --config configs/curriculum-complex.yaml --model results/curriculum-complex/numbers/model.pt --output results/curriculum-complex/numbers/contact-sheet.png

evaluation: shapes
	$(PYTHON) -m src.eval.run --config configs/evaluation-pilot.yaml --output results/evaluation-pilot
