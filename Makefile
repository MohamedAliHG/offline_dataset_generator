PYTHON ?= python

.PHONY: index run run-agent1 run-agent2 test lint format typecheck

index:
	$(PYTHON) scripts/index.py --dir data/raw

run:
	$(PYTHON) scripts/run_pipeline.py

run-agent1:
	$(PYTHON) scripts/run_pipeline.py --agent1-only

run-agent2:
	$(PYTHON) scripts/run_pipeline.py --agent2-only

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts

format:
	black src tests scripts

typecheck:
	mypy src
