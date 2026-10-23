# Les commandes d'entraînement sont explicites et ne font pas partie des tests.
PYTHON ?= python
PROFILE ?= standard
VARIANT ?= gnn_2_rounds
SEED ?= 11
TRAINING_OUTPUT ?= outputs/training
EVALUATION_OUTPUT ?= outputs/evaluation
CURRICULUM_VARIANT ?= gnn_3_rounds_geometry
CURRICULUM_CONFIG ?= curriculum

ifeq ($(OS),Windows_NT)
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -Command
endif

.PHONY: help install install-dev test test-all test-training shapes demo tuning baselines metrics train train-all train-complete train-smoke training-plan training-summary train-size-20 train-size-40 train-size-100 train-sizes campaign comparison size-transfer curriculum-plan train-curriculum train-curriculum-all curriculum-summary complex-shapes curriculum-comparison shape-transfer readability readability-score robustness web clean
.DEFAULT_GOAL := help

help:
	@echo "=== swarm-gnn : commandes ==="
	@echo "make install / test / shapes : dependances, tests sans entrainement, cibles"
	@echo "make train VARIANT=gnn_2_rounds SEED=11 PROFILE=standard : un entrainement"
	@echo "make train-all : toutes les variantes et les cinq graines du profil"
	@echo "make train-complete : complete les variantes et graines manquantes"
	@echo "make train-smoke : optimisation courte REELLE, a lancer vous-meme"
	@echo "make training-plan / curriculum-plan : budgets sans optimisation"
	@echo "make training-summary : courbes et synthese des entrainements complets"
	@echo "make train-sizes / size-transfer : entrainer par taille, puis evaluer"
	@echo "make campaign / comparison : evaluer les checkpoints, puis comparer"
	@echo "make train-curriculum / train-curriculum-all : un curriculum ou toute la campagne"
	@echo "make curriculum-summary / complex-shapes / curriculum-comparison / shape-transfer : rapports et etudes"
	@echo "make readability / readability-score : questionnaire aveugle, puis reponses humaines"
	@echo "make robustness / web : perturbations, demonstration locale"
	@echo "PYTHON=python est surchargeable ; aucun dossier de run existant n'est ecrase"

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install pytest-cov ruff httpx

test:
	$(PYTHON) -m pytest tests/ --tb=short -q

test-all:
	$(PYTHON) -m pytest tests/ -v

test-training:
	$(PYTHON) -m pytest tests/ -m training -v

shapes:
	$(PYTHON) -m src.shapes.generation

demo:
	$(PYTHON) -m src.reports.demo_phase1

tuning:
	$(PYTHON) -m src.reports.tune_baselines

baselines:
	$(PYTHON) -m src.reports.baselines_demo

metrics:
	$(PYTHON) -m src.reports.metrics_demo

train:
	$(PYTHON) -m src.training.train --variant $(VARIANT) --seed $(SEED) --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"

train-all:
	$(PYTHON) -m src.training.train --all-variants --all-seeds --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"

train-complete:
	$(PYTHON) -m src.training.train --variant gnn_3_rounds --seed 11 --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"
	$(PYTHON) -m src.training.train --variant no_comm --seed 11 --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"
	$(PYTHON) -m src.training.train --variant no_comm --seed 12 --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"
	$(PYTHON) -m src.training.train --variant no_comm --seed 13 --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"
	$(PYTHON) -m src.training.train --variant no_comm --seed 14 --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"
	$(PYTHON) -m src.training.train --variant no_comm --seed 15 --profile $(PROFILE) --output "$(TRAINING_OUTPUT)"

train-smoke:
	$(PYTHON) -m src.training.train --variant $(VARIANT) --seed 1 --profile smoke --output outputs/training_smoke

training-plan:
	$(PYTHON) -m src.training.train --all-variants --all-seeds --profile $(PROFILE) --output "$(TRAINING_OUTPUT)" --dry-run

training-summary:
	$(PYTHON) -m src.reports.training_summary --input "$(TRAINING_OUTPUT)"

train-size-20:
	$(PYTHON) -m src.training.train --variant gnn_2_rounds --all-seeds --profile size_20 --output outputs/training/size_20

train-size-40:
	$(PYTHON) -m src.training.train --variant gnn_2_rounds --all-seeds --profile size_40 --output outputs/training/size_40

train-size-100:
	$(PYTHON) -m src.training.train --variant gnn_2_rounds --all-seeds --profile size_100 --output outputs/training/size_100

train-sizes:
	$(PYTHON) -m src.training.train --variant gnn_2_rounds --all-seeds --profile size_20 --output outputs/training/size_20
	$(PYTHON) -m src.training.train --variant gnn_2_rounds --all-seeds --profile size_40 --output outputs/training/size_40
	$(PYTHON) -m src.training.train --variant gnn_2_rounds --all-seeds --profile size_100 --output outputs/training/size_100

campaign:
	$(PYTHON) -m src.evaluation.campaign --output "$(EVALUATION_OUTPUT)"

comparison:
	$(PYTHON) -m src.reports.communication --config comparison

size-transfer:
	$(PYTHON) -m src.evaluation.size_transfer --config size-transfer

curriculum-plan:
	$(PYTHON) -m src.training.curriculum --config $(CURRICULUM_CONFIG) --all --profile $(PROFILE) --dry-run

train-curriculum:
	$(PYTHON) -m src.training.curriculum --config $(CURRICULUM_CONFIG) --variant $(CURRICULUM_VARIANT) --seed $(SEED) --profile $(PROFILE)

train-curriculum-all:
	$(PYTHON) -m src.training.curriculum --config $(CURRICULUM_CONFIG) --all --profile $(PROFILE)

curriculum-summary:
	$(PYTHON) -m src.reports.curriculum --config $(CURRICULUM_CONFIG) --profile $(PROFILE)

complex-shapes:
	$(PYTHON) -m src.evaluation.shape_studies --config complex-shapes --mode complex

curriculum-comparison:
	$(PYTHON) -m src.evaluation.shape_studies --config complex-shapes --mode compare

shape-transfer:
	$(PYTHON) -m src.evaluation.shape_studies --config shape-transfer --mode heldout

readability:
	$(PYTHON) -m src.reports.readability --config readability --mode prepare

readability-score:
	$(PYTHON) -m src.reports.readability --config readability --mode score

robustness:
	$(PYTHON) -m src.evaluation.robustness --config robustness

web:
	$(PYTHON) -m demo.server --config web-demo

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(path) for path in pathlib.Path('.').rglob('__pycache__') if '.venv' not in path.parts]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
