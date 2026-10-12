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
| `collect/` | `sources.json` (licence/terms per source), `download.py`, `inspect_raw.py`, `polite.py` (HTTP etiquette), `robots.py`; `scrape/` (probe, crawler, report, one adapter per site in `scrape/sites/`) |
| `clean/` | `extract_<source>.py` + `extract_all.py` (Step 3), `wikitext.py`, `peek.py`; `audit_chars.py` (Step 4), `linefilter.py` + `normalize_filter.py` (Step 5); boilerplate, dedup, split, data-card generator to come |
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

## Collect our own text (Part A, Step 2)

At least 50 MB must be collected by us, with evidence. `collect/scrape/` does it under the same etiquette as the downloader. Both robots.txt parsers (ours and `urllib.robotparser`) must allow every request. Requests are spaced by max(Crawl-delay, 1 s) plus jitter, with no cookies, no login, and no comments, bylines or user pages.

```bash
python collect/scrape/probe.py                                    # Phase A: robots, sitemaps, sample pages, terms per candidate
python collect/scrape/crawl.py --site voa_urdu --limit 50 --max-sitemaps 1   # smoke test
python collect/scrape/crawl.py --site voa_urdu --target-mb 70     # full crawl; resumable, rerun after an interruption
python collect/scrape/crawl.py --site humsub --target-mb 20 --max-hours 7   # stops at whichever comes first
python collect/scrape/crawl.py --site voa_urdu --reextract        # rebuild articles.jsonl from the HTML cache, no requests
python collect/scrape/report.py                                   # Phase C: evidence/scrape_report.md and sources.json
```

The crawl writes `data/raw/own_scrape/<site>/articles.jsonl` and logs every request to `evidence/crawl_log_<site>.jsonl.gz`. The site choice and its evidence are in [evidence/probe_report.md](evidence/probe_report.md) and [docs/decisions.md](docs/decisions.md).

## Extract plain text (Part A, Step 3)

One extractor per source turns its raw container into `data/interim/<source>.jsonl.gz`, one document per line: `doc_id, source, genre, url, title, date, text, meta`. Nothing is normalized or filtered for language yet. Each extractor also writes a deterministic ~2 MB sample to `data/samples/<source>.sample.jsonl`, and its counts (records read, docs written, bytes, drops by reason) to `data/stats/extract.json`.

| Source | Script | Notes |
|---|---|---|
| wiki_ur | `clean/extract_wiki.py` | our own wikitext stripper (`clean/wikitext.py`); ns 0 only; no redirects or disambiguation pages |
| ufal_umc | `clean/extract_ufal.py` | order test decides the document unit (`--group auto`) |
| urdu_news_1m | `clean/extract_news1m.py` | headline + story; `meta.punct_stripped` |
| fineweb2_urd | `clean/extract_fineweb2.py` | `--keep-per-mille K` deterministic subsample |
| makhzan | `clean/extract_makhzan.py` | `<body>` only; `--show-one` prints one raw file and its extraction |
| own_scrape | `clean/extract_scrape.py` | repeated canonical URLs dropped |
| cc100_ur | `clean/extract_cc100.py` | fallback, disabled |

```bash
python clean/extract_all.py --limit 2000                 # smoke run of every enabled source with raw data
python clean/peek.py --source wiki_ur --n 1 --raw        # one article: raw wikitext, then the extracted text
python clean/extract_all.py --wiki-jobs 7                # full run (see the tmux command below)
python clean/extract_all.py --table                      # the per-source table from data/stats/extract.json
```

Full run, in tmux:

```bash
tmux new -s extract 'cd ~/Desktop/LLM/urdu-bpe && mkdir -p logs && .venv/bin/python clean/extract_all.py --wiki-jobs 7 2>&1 | tee logs/extract.log'
```

## Normalize Urdu (Part A, Step 4)

`normalize.normalize(text)` maps Arabic yeh, kaf and heh (and ە, ة, ۀ) to their Urdu forms, and turns presentation forms into plain letters. It composes decomposed letters (NFC), removes tatweel and invisible characters, keeps ZWNJ only inside words, writes all digits as ASCII, and tidies whitespace. The rules, their order and the reasons are in the module docstring and in [docs/decisions.md](docs/decisions.md). The cleaning pipeline applies it, and the tokenizer never does.

```bash
python clean/audit_chars.py                  # code points that normalization acts on, per source (data/samples)
python clean/audit_chars.py --normalized     # what is left after normalize(); should be only deliberate keeps
python clean/audit_chars.py --interim --limit 20000 --source wiki_ur --contexts 20
```

## Normalize and filter by language (Part A, Step 5)

`clean/normalize_filter.py` reads `data/interim/<source>.jsonl.gz` and works through each document in three steps:
- runs `normalize()`;
- drops the lines that are not Urdu (`clean/linefilter.py`): no letters, mostly not Urdu script, Sindhi, or Arabic/Persian with positive evidence;
- drops the documents that end up mostly Sindhi or too short (10 words; 3 for UFAL's one-sentence documents).

Code-mixed lines stay: an Urdu sentence with English names keeps its Urdu function words. Output goes to `data/clean/filtered/<source>.jsonl.gz`. Counts per step go to `data/stats/normalize.json` and `data/stats/langfilter.json`, a 2 MB sample to `data/samples/filtered/`, and up to 200 distinct dropped lines per reason to `data/samples/dropped_lines/<source>.jsonl`.

```bash
python clean/normalize_filter.py --limit 2000            # smoke run, every source
python clean/normalize_filter.py --table                 # per-source MB in / after normalize / out, biggest drops
```

Full run (about 20 minutes on 7 workers for 9.6 GB), in tmux:

```bash
tmux new -s filter 'cd ~/Desktop/LLM/urdu-bpe && mkdir -p logs && caffeinate -i .venv/bin/python clean/normalize_filter.py --jobs 7 2>&1 | tee logs/normalize_filter_run.log'
```

## Rebuild the tokenizer from train.txt

Not written yet. The commands go here once Part B is done.
