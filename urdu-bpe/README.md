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
| `collect/` | `sources.json` (licence/terms per source), `download.py`, `inspect_raw.py`, `polite.py` (HTTP etiquette), `robots.py`; scraper in Step 2 |
| `clean/` | extractors, audit, cleaning pipeline, split, data-card generator |
| `analysis/` | Part C scripts (`analysis/samples/` holds the hand-picked inputs) |
| `report/` | LaTeX report, built to `report/report.pdf` |
| `evidence/` | scrape logs and robots.txt/terms snapshots |
| `tokenizer/` | saved tokenizer |
| `data/{raw,interim,clean,final,stats,samples}` | corpus at each stage (gitignored) |
| `logs/` | run logs (gitignored) |

## Collect (Part A, Step 1)

Every source, with its licence and terms, is listed in [collect/sources.json](collect/sources.json).

```bash
python collect/download.py --source all --dry-run        # every URL and target path; downloads nothing
python collect/download.py --source wiki_ur              # resumable; rerun the same command after an interruption
python collect/download.py --source fineweb2_urd --files 001_00000.parquet
python collect/inspect_raw.py                            # files, first records, source-specific checks
```

Downloads go to `data/raw/<source>/` and are verified against each source's published checksum. They are recorded in `data/raw/<source>/MANIFEST.json` and in `sources.json`. Before the first download, put your contact email on one line in `collect/contact.local` (gitignored), or set `$URDU_BOT_CONTACT`. It goes into the User-Agent, and the scripts refuse to run without it. A source that needs a browser (robots.txt or HTTP 401/403) gets printed manual steps, and `--register-manual` records the file once you have placed it.

## Rebuild the tokenizer from train.txt

Not written yet. The commands go here once Part B is done.
