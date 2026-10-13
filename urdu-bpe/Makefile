PYTHON ?= .venv/bin/python

.PHONY: test check analysis numbers report report-tools submission

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

# Report (Step 12). numbers: report/numbers.tex and the report's own tables from the JSON stats (needs data/,
# so run it where the corpus is, and commit the result). report: LuaLaTeX from the committed numbers.tex and
# tables (no data needed), then fail if report.pdf is over 4 pages.
numbers:
	$(PYTHON) report/make_numbers.py

report-tools:
	$(PYTHON) report/checks.py tools

report: report-tools
	cd report && latexmk -lualatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex
	$(PYTHON) report/checks.py pages report/report.pdf --max 4

# Pre-flight checks, submission/urdu_bpe_456094.zip and data/final/SHA256SUMS (make_submission.py).
submission:
	$(PYTHON) make_submission.py
