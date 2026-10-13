# Urdu corpus + byte-level BPE tokenizer

CS 417 Assignment 1 (NUST SEECS). We build a cleaned Urdu corpus of 1,299.86 MB, split by document into train, val and test (98/1/1).
We train a byte-level BPE tokenizer, written from scratch in plain Python, on train.txt to exactly 50,000 tokens, and encode train and val into `train.bin` and `val.bin` for the next assignment.
We then analyse what it learned against GPT-2 and o200k_base: [report/report.pdf](report/report.pdf) (4 pages) and [DATA_CARD.md](DATA_CARD.md).

The handout is [docs/assignment.md](docs/assignment.md). Our rules are in [CLAUDE.md](CLAUDE.md), every decision with its evidence is in [docs/decisions.md](docs/decisions.md), and the commands for each step are in [docs/pipeline.md](docs/pipeline.md).

## Repository layout

| Path | Contents |
|---|---|
| `normalize.py`, `bpe.py`, `train_tokenizer.py`, `encode_corpus.py` | the names the handout requires: Urdu normalization, `BPETokenizer` (pre-tokenizer, reference and fast trainers, encode/decode, save/load), the training CLI, the corpus encoder |
| `common.py` | JSONL I/O, hashing, machine info, stats, timers, logging |
| `collect/` | `sources.json` (licence and terms of every source), downloaders, and the polite scraper in `scrape/` (Steps 1-2) |
| `clean/` | extractors, character audit, normalize + language filter, boilerplate and dedup, budget, split, corpus stats, data card (Steps 3-6) |
| `analysis/` | Part C scripts and the hand-picked inputs in `analysis/samples/` (sources in `SOURCES.md`); the fast-vs-reference and scaling reports |
| `report/` | `report.tex`, `make_numbers.py` (every number in the report, from the JSON stats), generated `tables/` and `figures/`, `report.pdf` |
| `tokenizer/` | `urdu_bpe_50k.json` (all 49,743 merges in order), `urdu_bpe_50k.merges.txt` (readable), `train_log.json`; the 1 MB reference run |
| `evidence/` | our own collection: robots.txt and terms snapshots, the probe and scrape reports, every request we made |
| `docs/` | handout, decisions, pipeline, pre-tokenizer, equivalence, scaling, training run, analysis notes, cleaning detail |
| `tests/` | `pytest -q`: every module, plus integrity checks that run against the real split and tokenizer when present |
| `make_submission.py` | pre-flight checks, the submission archive and the shared-drive checksums |
| `data/`, `logs/` | the corpus at each stage, caches and run logs (gitignored, never committed) |

## Setup

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -r requirements.txt -r requirements-analysis.txt -r requirements-dev.txt
```

Without uv:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-analysis.txt -r requirements-dev.txt
```

| File | For | Packages (versions pinned) |
|---|---|---|
| `requirements.txt` | the implementation | regex, numpy, tqdm, requests, beautifulsoup4, lxml, pyarrow |
| `requirements-analysis.txt` | Part C only | tiktoken, tokenizers, matplotlib |
| `requirements-dev.txt` | tests | pytest |

**Library policy.** The implementation (repo-root scripts, `collect/`, `clean/`) uses only the standard library, regex, numpy, tqdm and the collection tools (requests, beautifulsoup4, lxml; pyarrow only to read the FineWeb-2 parquet files). `bpe.py` itself imports only the standard library and regex. No tokenizer library or published BPE code is used. tiktoken and tokenizers appear only in `analysis/compare_tokenizers.py`, and matplotlib only in `analysis/`. `tests/test_forbidden_imports.py` enforces all of this (`make check`).

## Rebuild the tokenizer from train.txt

Put `train.txt` and `val.txt` from the shared drive (see [Data access](#data-access)) in `data/final/`, then run, in order:

```bash
# 1. environment
uv venv --python 3.12 && uv pip install -r requirements.txt -r requirements-dev.txt

# 2. train the tokenizer: fast trainer, 50,000 tokens, then the post-training checks on val.txt
.venv/bin/python train_tokenizer.py --input data/final/train.txt --vocab-size 50000 \
    --out tokenizer/urdu_bpe_50k.json --workers 7 --checkpoint-every 5000

# 3. encode train and val with it
.venv/bin/python encode_corpus.py --input data/final/train.txt --out data/final/train.bin --workers 7
.venv/bin/python encode_corpus.py --input data/final/val.txt --out data/final/val.bin --workers 7

# 4. verify: each .bin must decode to exactly the bytes of its .txt
.venv/bin/python encode_corpus.py --verify --input data/final/train.txt --out data/final/train.bin
.venv/bin/python encode_corpus.py --verify --input data/final/val.txt --out data/final/val.bin

# 5. tests, including the checks on the trained tokenizer
.venv/bin/python -m pytest -q
```

If training stops, rerun step 2 with `--resume` added. It continues from `tokenizer/urdu_bpe_50k.checkpoint.json`, written every 5,000 merges, and reuses the cached word counts.

Expected outputs:

| Output | What to expect |
|---|---|
| `tokenizer/urdu_bpe_50k.json` | 49,743 merges with IDs 257-49,999 in learned order; vocabulary 50,000; `<|endoftext|>` = 256 |
| `tokenizer/urdu_bpe_50k.merges.txt` | one readable line per merge; sha256 `6f07e0f13c0eabee4ee81b2aa09541830f75e2e72a4e9fffeb7ce8ea9b667c86`, identical on a rebuild |
| the delivered `urdu_bpe_50k.json` | sha256 `4994584016fd841f873599b22e83b084e526092f3c14740fa56314bcc48e6ece`. A rebuild learns identical merges, but the file's `meta` records the new run's times and machine, so its sha256 differs; compare `merges.txt` |
| `tokenizer/train_log.json`, `docs/training_run.md` | phase times, peak memory, machine, and "All post-training checks passed" |
| `data/final/train.bin`, `val.bin`, `meta.json` | 179,309,360 and 1,888,474 tokens, raw little-endian uint16; `--verify` prints VERIFIED for both |
| `pytest -q` | all tests pass, none skipped (with data/final present) |

Measured on an Apple M2 (8 cores, 8 GB RAM, macOS 14.4.1, Python 3.12.15) with 7 workers:
- **Training:** 76.5 s of wall clock: word counting 13.2 s, initialization 3.4 s, 49,743 merges 59.9 s, save 0.1 s. Peak memory 1.44 GB. The post-training checks add a few seconds.
- **Encoding:** train.txt 23.8 s plus 12.1 s to verify; val.txt 1.6 s plus 0.2 s.

## Rebuild the corpus (Part A, Steps 1-6)

From an empty `data/`, in order. Downloads need a contact email in `collect/contact.local` (gitignored) for the User-Agent. The crawls take hours and the cleaning about an hour, so run them in tmux, as [docs/pipeline.md](docs/pipeline.md) shows.

```bash
.venv/bin/python collect/download.py --source all                     # Step 1: published corpora, checksums verified
.venv/bin/python collect/scrape/crawl.py --site voa_urdu --target-mb 70   # Step 2: our own crawl
.venv/bin/python collect/scrape/crawl.py --site humsub --target-mb 20 --max-hours 7
.venv/bin/python collect/scrape/report.py                             #   evidence/scrape_report.md
.venv/bin/python clean/extract_all.py --wiki-jobs 7                   # Step 3: plain text
.venv/bin/python clean/audit_chars.py && .venv/bin/python clean/audit_chars.py --normalized   # Step 4
.venv/bin/python clean/normalize_filter.py --jobs 7                   # Step 5: normalize + language filter
.venv/bin/python clean/dedup.py                                       # Step 5b: boilerplate + exact dedup
.venv/bin/python clean/budget.py --fill fineweb2_urd ufal_umc urdu_news_1m   # Step 6: byte budget
.venv/bin/python clean/split.py && .venv/bin/python clean/corpus_stats.py
.venv/bin/python -m pytest -q -s tests/test_split.py                  #   integrity checks on the split
.venv/bin/python clean/data_card.py                                   #   DATA_CARD.md
```

**Evidence for our own collection (76.01 MB in the corpus).** It is in `evidence/`:
- `probe_report.md`: the site choice;
- `scrape_report.md`: what was collected;
- `crawl_log_<site>.jsonl.gz`: every request, with status and robots decision;
- `robots/`, `terms/`: snapshots of each site's robots.txt and terms of use.

## Loading the data

```python
import numpy as np
from bpe import BPETokenizer

ids = np.memmap("data/final/train.bin", dtype=np.uint16, mode="r")    # little-endian; "<u2" on any machine
tok = BPETokenizer.load("tokenizer/urdu_bpe_50k.json")
print(len(ids), tok.decode(ids[:200].tolist()))
```

- **Format.** Each `.bin` is a flat array of token IDs with no header. Every document is followed by `<|endoftext|>` = ID 256, and the vocabulary size is 50,000, so IDs fit in uint16.
- **`data/final/meta.json`** records the vocabulary size, EOT ID, dtype and byte order. It also records the tokenizer with its sha256, the token counts, and each encode's report: documents, bytes per token, tokens per word, cache hit rate, time and verification.
- **`data/final/manifest.json`** records the split: seed, budget, documents and bytes per split and source, and sha256 of each file.

## Report, analysis and submission

```bash
make analysis     # Part C tables, figures and drafts (needs data/final and the cached word counts)
make numbers      # report/numbers.tex and the report's tables from the JSON stats in data/ (commit the result)
make report       # LuaLaTeX from the committed numbers and tables; fails on a missing glyph or over 4 pages
make submission   # pre-flight checks, submission/urdu_bpe_456094.zip, data/final/SHA256SUMS
```

`make report` needs LuaLaTeX, latexmk and the Noto Naskh Arabic font. `python report/checks.py tools` lists what is missing and how to install it; pdfinfo (poppler) is optional.

## Data access

**Download.** The corpus files go by shared-drive link: train.txt, val.txt, test.txt, train.bin, val.bin, meta.json, manifest.json and SHA256SUMS.

<!-- DRIVE LINK -->
Link: in the submitted copy of this README. This repository is public and the link must stay private to course staff, so the link is not committed: `make_submission.py` reads it from `drive_link.local` (gitignored) and writes it here, in the archive only.
<!-- /DRIVE LINK -->

Check the download with `cd data/final && shasum -a 256 -c SHA256SUMS`.

**Licences.** The link is private to course staff, and the corpus must not be redistributed outside the course:
- **Makhzan:** copyright, research use only, no republication. The instructor approved including it on condition the link stays private.
- **UFAL Urdu Monolingual Corpus:** CC BY-NC-SA 3.0.
- **Urdu News 1M:** CC BY 4.0, non-commercial research (left out of the final corpus).
- **FineWeb-2:** ODC-By 1.0 plus the Common Crawl terms.
- **Urdu Wikipedia:** CC BY-SA 4.0.
- **Our crawl:** VOA's own text is public domain; Hum Sub posts are their authors' copyright.

Every source and its licence is in [collect/sources.json](collect/sources.json) and [DATA_CARD.md](DATA_CARD.md).

## Tests

```bash
make test     # pytest -q
make check    # forbidden-imports check only
```
