PYTHON ?= .venv/bin/python

.PHONY: test check analysis

# Full test suite.
test:
	$(PYTHON) -m pytest -q

# Forbidden-imports check only (CLAUDE.md, hard rules 1-3).
check:
	$(PYTHON) -m pytest -q tests/test_forbidden_imports.py

# Part C (Step 11): every table, figure and draft, in order (failure_cases reads compare_tokenizers' counts).
analysis:
	$(PYTHON) analysis/merges_report.py
	$(PYTHON) analysis/compare_tokenizers.py
	$(PYTHON) analysis/failure_cases.py
