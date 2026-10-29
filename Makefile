# Parcours actuel à accélérations. Les entraînements restent explicites.
PYTHON ?= python
DATASET ?= outputs/imitation/multiform-expanded-task008d
CIRCLE_DATASET ?= outputs/imitation/circle-reviewed-task007
TRAINING_OUTPUT ?= outputs/imitation/training-new
OUTPUT ?= outputs/comparison-final

ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
endif

.DEFAULT_GOAL := help
.PHONY: help install test web web-reference comparison-reference circle train-imitation train-multiform train-circle

help:
	@echo "make install / test : dependances et tests"
	@echo "make web : demonstration locale sur http://127.0.0.1:8765"
	@echo "make comparison-reference OUTPUT=nouveau-dossier : rapport des campagnes conservees"
	@echo "make circle : professeur circulaire a accelerations"
	@echo "make train-multiform / train-circle : apprentissage reel explicite"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -q -p no:cacheprovider

web:
	$(PYTHON) -m demo.reference_server

web-reference: web

comparison-reference:
	$(PYTHON) -m demo.comparison_report --output "$(OUTPUT)"

circle:
	$(PYTHON) -m src.reference.demo

train-imitation: train-multiform

train-circle:
	$(PYTHON) -m src.imitation.train --dataset "$(CIRCLE_DATASET)" --output "$(TRAINING_OUTPUT)"

train-multiform:
	$(PYTHON) -m src.imitation.local_target_train --dataset "$(DATASET)" --training config/local-target-training-expanded.yaml --output "$(TRAINING_OUTPUT)"
