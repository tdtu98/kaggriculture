PYTHON ?= python3.12
VENV ?= .venv
PYTHON_BIN := $(VENV)/bin/python
CONFIG ?= configs/v0.json
DATA_ROOT ?= data
RUNS_ROOT ?= runs
RUN_ID ?= ryo-v0

.PHONY: setup prepare train evaluate reproduce test

setup:
	@$(PYTHON) -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
	$(PYTHON) -m venv $(VENV)
	$(PYTHON_BIN) -m pip install -e '.[dev]'

prepare:
	$(PYTHON_BIN) scripts/prepare_data.py --config $(CONFIG) --output-root $(DATA_ROOT)

train:
	$(PYTHON_BIN) scripts/train_v0.py --config $(CONFIG) --data-root $(DATA_ROOT) --runs-root $(RUNS_ROOT) --run-id $(RUN_ID)

evaluate:
	$(PYTHON_BIN) scripts/evaluate_v0.py --data-root $(DATA_ROOT) --runs-root $(RUNS_ROOT) --run-id $(RUN_ID) --split test

reproduce:
	$(PYTHON_BIN) -m scripts.reproduce --config $(CONFIG) --data-root $(DATA_ROOT) --runs-root $(RUNS_ROOT) --run-id $(RUN_ID)

test:
	$(PYTHON_BIN) -m unittest discover -s tests
