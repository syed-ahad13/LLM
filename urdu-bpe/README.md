# Urdu corpus + byte-level BPE

CS 417 Assignment 1 (NUST, Spring 2026). This repo builds a cleaned Urdu corpus of at least 500 MB, implements a byte-level BPE tokenizer from scratch, trains it to exactly 50,000 tokens, and compares it with GPT-2 and o200k_base.

The handout is in [docs/assignment.md](docs/assignment.md). Project rules and specs are in [CLAUDE.md](CLAUDE.md), and decisions are logged in [docs/decisions.md](docs/decisions.md).

## Setup

You need Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-analysis.txt -r requirements-dev.txt
```

Without uv:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-analysis.txt -r requirements-dev.txt
```

| File | For | Packages |
|---|---|---|
| `requirements.txt` | implementation | regex, numpy, tqdm, requests, beautifulsoup4, lxml, pyarrow |
| `requirements-analysis.txt` | Part C only | tiktoken, tokenizers, matplotlib |
| `requirements-dev.txt` | tests | pytest |

All versions are pinned exactly.

## Checks

```bash
make test    # full test suite (pytest -q)
make check   # forbidden-imports check only
```

## Layout

| Path | Contents |
|---|---|
| `normalize.py`, `bpe.py`, `train_tokenizer.py`, `encode_corpus.py` | names required by the handout |
| `common.py` | JSONL I/O, hashing, machine info, stats, timing, logging |
| `collect/` | downloaders, scraper, `sources.json` (licence/terms per source) |
| `clean/` | extractors, audit, cleaning pipeline, split, data-card generator |
| `analysis/` | Part C scripts (`analysis/samples/` holds the hand-picked inputs) |
| `report/` | LaTeX report, built to `report/report.pdf` |
| `evidence/` | scrape logs and robots.txt/terms snapshots |
| `tokenizer/` | saved tokenizer |
| `data/{raw,interim,clean,final,stats,samples}` | corpus at each stage (gitignored) |
| `logs/` | run logs (gitignored) |

## Rebuild the tokenizer from train.txt

Not written yet. The commands go here once Part B is done.
