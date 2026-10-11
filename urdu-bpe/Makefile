PYTHON ?= .venv/bin/python

.PHONY: test check

# Full test suite.
test:
	$(PYTHON) -m pytest -q

# Forbidden-imports check only (CLAUDE.md, hard rules 1-3).
check:
	$(PYTHON) -m pytest -q tests/test_forbidden_imports.py
