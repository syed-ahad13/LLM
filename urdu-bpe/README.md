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
| `clean/` | `extract_<source>.py` + `extract_all.py` (Step 3), `wikitext.py`, `peek.py`; `audit_chars.py` (Step 4), `linefilter.py` + `normalize_filter.py` (Step 5), `dedup.py` (Step 5b); `budget.py`, `split.py`, `corpus_stats.py`, `data_card.py` (Step 6) |
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

## Boilerplate lines and exact duplicates (Part A, Step 5b)

`clean/dedup.py` reads `data/clean/filtered/` and writes `data/clean/<source>.jsonl.gz`. For each document it:
- removes any literal `<|endoftext|>`;
- drops every line found in at least the source's cutoff number of documents (wire-agency credits, bylines, "مزید پڑھیں", section labels);
- re-checks the minimum word count;
- drops exact duplicate documents.

Sources are processed in priority order (own_scrape, makhzan, wiki_ur, urdu_news_1m, ufal_umc, fineweb2_urd), so a document found in two sources stays in the earlier one. Stats go to `data/stats/boilerplate.json` and `dedup.json`. The most frequent removed lines go to `data/samples/boilerplate/<source>.jsonl`, with their document counts.

```bash
python clean/dedup.py --report               # per source: lines and MB removed at each cutoff, with examples
python clean/dedup.py --limit 2000           # smoke run
python clean/dedup.py --table
```

Full run (about 25 minutes, one process; it needs all sources in one run for cross-source dedup), in tmux:

```bash
tmux new -s dedup 'cd ~/Desktop/LLM/urdu-bpe && mkdir -p logs && caffeinate -i .venv/bin/python clean/dedup.py 2>&1 | tee logs/dedup_run.log'
```

## Budget, split and data card (Part A, Step 6)

Step 5b leaves 9.19 GB of cleaned text, and the corpus targets 1.0-1.3 GB. Four scripts finish Part A:

- `clean/budget.py` prints the cleaned MB per source and genre, then decides how many bytes each source contributes. A source is kept whole or subsampled deterministically: a document is kept when `int(blake2b(doc_id).hexdigest(), 16) % 10**6 < rate * 10**6`. By default FineWeb-2 is capped at 45% of the corpus, and Urdu News 1M at 10% because its punctuation was stripped at source. Sizes include the 13-byte `<|endoftext|>` after each document. The scan is cached in `data/stats/budget_scan.json`. Writes `data/final/budget.json`.
- `clean/split.py` works per source: it sorts the kept doc_ids, shuffles them with `random.Random("1337:<source>")` and cuts 98/1/1 by count, with at least one val and one test document. It then shuffles each split across sources and writes every document followed by `<|endoftext|>`. A three-pass external shuffle (64 bucket files per split) keeps the text out of memory. Writes `data/final/{train,val,test}.txt`, the doc_ids in file order (`.ids`) and `manifest.json` (counts, bytes and sha256 per split and source).
- `clean/corpus_stats.py` counts MB, MiB, documents, words and unique words per split, and writes `data/stats/corpus_totals.json`.
- `clean/data_card.py` writes `DATA_CARD.md` and `docs/cleaning_detail.md` from the stats. The "Known gaps and biases" prose comes from `docs/data_card_gaps.md`, so edit it there.

```bash
python clean/budget.py                       # tables; says why no budget fits if the policy cannot be met
python clean/budget.py --fill fineweb2_urd ufal_umc urdu_news_1m   # the budget in docs/decisions.md (Step 6)
python clean/split.py                        # a few minutes
python clean/corpus_stats.py
python -m pytest -q -s tests/test_split.py   # integrity checks on the real files; -m "not slow" skips the rerun
python clean/audit_chars.py && python clean/audit_chars.py --normalized   # both audits on the same samples
python clean/data_card.py
```

## Pre-tokenizer and reference BPE (Part B, Step 7)

`bpe.py` holds `BPETokenizer` and the pre-tokenizer pattern `URDU_PATTERN`. The pattern and the alternatives we considered are explained in [docs/pretokenizer.md](docs/pretokenizer.md). `bpe.py` imports only the standard library and `regex`.
- `train_reference` is the reference trainer. It keeps every word occurrence and recounts all pairs after every merge. Ties go to the greater `(vocab[a], vocab[b])`.
- `encode` / `decode` round-trip any Python string.
- `save` writes `tokenizer/<name>.json` and a readable `tokenizer/<name>.merges.txt`.

```bash
python clean/sample_1mb.py        # data/samples/train_1mb.txt: whole documents from the start of train.txt, at most 1,000,000 bytes
python train_tokenizer.py --trainer reference --input data/samples/train_1mb.txt --vocab-size 3257 --name reference_1mb_3000
python -m pytest -q tests/test_bpe_reference.py -s   # -s prints what GPT-2's pattern does to اُردُو
```

The 3,000-merge reference run takes a few minutes. It prints the first 30 merges and the wall-clock time, which is also stored in the tokenizer's `meta`. Step 8's fast trainer must learn identical merges on the same sample.

## Rebuild the tokenizer from train.txt

Not written yet. The commands go here once Part B is done.
