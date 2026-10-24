PYTHON ?= python

.PHONY: test lint

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
