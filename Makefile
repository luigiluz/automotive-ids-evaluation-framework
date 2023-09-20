PYTHONPATH := .
VENV := venv
PYMODULE := automotive_ids_evaluation_framework
REQUIREMENTS := -r requirements.txt

BIN := $(VENV)/bin
PIP := $(BIN)/pip

PYTHON := env PYTHONPATH=$(PYTHONPATH) $(BIN)/python

bootstrap: venv \
	requirements \

venv:
	python3 -m venv $(VENV)

requirements:
	$(PIP) install $(REQUIREMENTS)

clean:
	@find . -type d -name '__pycache__' -exec rm -rf {} +

clean-all: clean
	@rm -r $(VENV)
