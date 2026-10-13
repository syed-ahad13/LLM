# Pipeline, step by step

The commands and notes for each step of Parts A-C, moved here from the README (Step 12). The README keeps the overview, the required "Rebuild the tokenizer from train.txt" commands and the full rebuild in order. Decisions and their evidence: [decisions.md](decisions.md).

## Collect (Part A, Step 1)

Every source, with its licence and terms, is listed in [collect/sources.json](../collect/sources.json).

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

The crawl writes `data/raw/own_scrape/<site>/articles.jsonl` and logs every request to `evidence/crawl_log_<site>.jsonl.gz`. The site choice and its evidence are in [evidence/probe_report.md](../evidence/probe_report.md) and [docs/decisions.md](decisions.md).

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

`normalize.normalize(text)` maps Arabic yeh, kaf and heh (and ە, ة, ۀ) to their Urdu forms, and turns presentation forms into plain letters. It composes decomposed letters (NFC), removes tatweel and invisible characters, keeps ZWNJ only inside words, writes all digits as ASCII, and tidies whitespace. The rules, their order and the reasons are in the module docstring and in [docs/decisions.md](decisions.md). The cleaning pipeline applies it, and the tokenizer never does.

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

`bpe.py` holds `BPETokenizer` and the pre-tokenizer pattern `URDU_PATTERN`. The pattern and the alternatives we considered are explained in [docs/pretokenizer.md](pretokenizer.md). `bpe.py` imports only the standard library and `regex`.
- `train_reference` is the reference trainer. It keeps every word occurrence and recounts all pairs after every merge. Ties go to the greater `(vocab[a], vocab[b])`.
- `encode` / `decode` round-trip any Python string.
- `save` writes `tokenizer/<name>.json` and a readable `tokenizer/<name>.merges.txt`.

```bash
python train_tokenizer.py --trainer reference --input data/final/train.txt --sample-mb 1 --vocab-size 3257 \
    --out tokenizer/reference_1mb_3000.json --no-checks
python -m pytest -q tests/test_bpe_reference.py -s   # -s prints what GPT-2's pattern does to اُردُو
```

`--sample-mb 1` takes whole documents from the start of train.txt, at most 1,000,000 bytes, and saves them as `data/samples/train_1mb.txt`. `clean/sample_1mb.py` writes the same file on its own. The 3,000-merge reference run takes a few minutes. It prints the first merges and the wall-clock time, which is also stored in the tokenizer's `meta`. Step 8's fast trainer must learn identical merges on the same sample.

## Fast trainer, equivalence and scaling (Part B, Step 8)

`BPETokenizer.train` is the fast trainer and learns exactly the reference trainer's merges. It uses the handout's three optimizations:
1. **`count_words`** counts each distinct word once. It reads 16 MB blocks cut at document boundaries in a process pool, and caches the counts in `data/cache/`.
2. **An index from each pair to the words that contain it,** so a merge rewrites only those words. Integer pair keys and a lazily repaired heap keep the spec's tie-break.
3. **A per-word cache in `encode`.**

The state and its invariants are described in `bpe.py`'s docstring. `train` can check them with `debug=N`, write checkpoints and resume from one.

```bash
python analysis/equivalence_1mb.py      # fast vs tokenizer/reference_1mb_3000.json on train_1mb.txt -> docs/equivalence_1mb.md
python analysis/scaling.py              # 10 MB and 100 MB prefixes of train.txt, 49,743 merges each -> docs/scaling.md
python train_tokenizer.py --input data/final/train.txt --vocab-size 50000 --out tokenizer/urdu_bpe_50k.json   # Step 9
```

## Train the tokenizer (Part B, Step 9)

`train_tokenizer.py` trains on the training split only. It refuses any input not named `train.txt` unless `--sample-mb` is given, and always refuses validation or test text. It times four phases with `common.Timer`:
1. word counting (process pool, cached in `data/cache/`);
2. initialization;
3. merges;
4. save.

It then runs the post-training checks on `data/final/val.txt`:
- exactly 49,743 merges with IDs 257-49,999, none touching `<|endoftext|>`;
- `load(save(tok))` identical on 5 MB of val.txt;
- 1,000 random val documents round-trip;
- merged tokens that repeat an earlier token's bytes are counted and reported;
- tokens per word, bytes per token, and the share of single-token words.

| Output | Contents |
|---|---|
| `tokenizer/urdu_bpe_50k.json`, `tokenizer/urdu_bpe_50k.merges.txt` | the tokenizer: every merge in order, readable form alongside |
| `tokenizer/train_log.json` | phase times, merges per second (overall and per 5,000 merges), peak RSS, distinct words, pre-token count, train.txt bytes and sha256, command, UTC times, git commit, machine |
| `docs/training_run.md` | one paragraph: phases, total, machine and check results |

A `--sample-mb` run logs to `tokenizer/<name>.train_log.json` instead, so it never overwrites the full run's log. Checkpoints (`tokenizer/<name>.checkpoint.json`, every 5,000 merges) are not committed.

```bash
python train_tokenizer.py --input data/final/train.txt --sample-mb 50 --vocab-size 50000 \
    --out tokenizer/smoke_50mb.json --workers 7       # smoke run on the first 50 MB
python train_tokenizer.py --check-only               # rerun the checks on tokenizer/urdu_bpe_50k.json
python -m pytest -q -s tests/test_trained_tokenizer.py
```

## Encode the corpus (Part B, Step 10)

`encode_corpus.py` turns `train.txt` and `val.txt` into `train.bin` and `val.bin`: every token ID in file order as raw little-endian uint16 (2 bytes per ID, no header), with `<|endoftext|>` = 256 after each document. The next assignment reads them with `np.memmap(path, dtype="<u2", mode="r")`. `test.txt` is not encoded.
- The input is read in 16 MB blocks cut after their last `<|endoftext|>` (the reader of `count_words`) and encoded in a process pool, in file order.
- Each worker loads the tokenizer once and caches the token IDs of every word it has encoded (optimization 3).
- Verification runs after every encode, and on its own with `--verify`. The `.bin` is decoded to bytes in 10M-ID chunks, and the sha256 of those bytes must equal the `.txt`'s sha256. The count of ID 256 must equal the `<|endoftext|>` count in the `.txt` and the document count in `data/final/manifest.json`.

| Output | Contents |
|---|---|
| `data/final/train.bin`, `data/final/val.bin` | the token IDs (gitignored with the rest of `data/`) |
| `data/final/meta.json` | vocab size, EOT ID, dtype, byte order, tokenizer and its sha256, `train_tokens`, `val_tokens`, the pattern; under `splits`, each encode's report (documents, bytes per token, tokens per word, cache hit rate per worker, time, MB/s, verification, sha256 of both files) |

```bash
python encode_corpus.py --input data/final/train.txt --sample-mb 20 --workers 7    # smoke run on the first 20 MB
python encode_corpus.py --verify --input data/final/train.txt --out data/final/train.bin   # verification only
python -m pytest -q tests/test_encode_corpus.py
```

Full runs, in tmux:

```bash
mkdir -p logs
tmux new -s encode 'cd ~/Desktop/LLM/urdu-bpe && caffeinate -dimsu .venv/bin/python encode_corpus.py --tokenizer tokenizer/urdu_bpe_50k.json --input data/final/train.txt --out data/final/train.bin --workers 7 2>&1 | tee logs/encode_train.log; .venv/bin/python encode_corpus.py --tokenizer tokenizer/urdu_bpe_50k.json --input data/final/val.txt --out data/final/val.bin --workers 7 2>&1 | tee logs/encode_val.log'
```

## Analyse the tokenizer (Part C, Step 11)

`make analysis` regenerates every Part C table, figure and draft in about 30 seconds. It needs the tokenizer, `data/final/{train,test}.txt`, `train.ids`, the word counts that Step 9 cached in `data/cache/`, and the hand-picked inputs in `analysis/samples/` (their sources are in [analysis/samples/SOURCES.md](../analysis/samples/SOURCES.md)). Plots carry English labels only. Urdu goes into the LaTeX tables, wrapped in `\ur{...}`, which the report must define (e.g. `\newcommand{\ur}[1]{\texturdu{#1}}` with polyglossia).

| Script | Question | Outputs in `report/` |
|---|---|---|
| `analysis/merges_report.py` | Q1: what did the merges learn? Merges 1-50 and every 5,000th, a category for each merged token (letter, space + lead byte, invalid UTF-8, word, ...), the first whole word, the 20 longest tokens traced to their sources in train.txt | `tables/merges.tex`, `tables/merges_longest.tex`, `figures/merge_categories.pdf` |
| `analysis/compare_tokenizers.py` | Q2: tokens per word, bytes and characters per token, and words split into 3+ tokens, for ours, GPT-2, o200k_base and (downloaded without a login) XLM-RoBERTa, on test.txt and the seven samples | `tables/compare.tex`, `tables/compare_metrics.tex`, `figures/tokens_per_word.pdf` |
| `analysis/failure_cases.py` | Q3: inputs and words ranked by our tokens per word and our-to-o200k ratio, three cases with token boundaries and their causes, and raw vs normalized text | `tables/failure_inputs.tex`, `tables/failure_words.tex`, `tables/failure_cases.tex`, `tables/normalization.tex` |

The raw numbers go to `data/stats/analysis.json` (per-word counts to `analysis_words.json`). The drafts go to [docs/analysis_notes.md](analysis_notes.md), between BEGIN/END markers that are rewritten on every run, so write your own text outside them. `tests/test_analysis.py` checks that each draft is exactly what the scripts produce from `analysis.json`. Only `compare_tokenizers.py` may import tiktoken and tokenizers, so it also stores the counts that `failure_cases.py` needs.

```bash
make analysis
python analysis/failure_cases.py --cases roman_urdu diacritics names   # choose the three cases by hand
python -m pytest -q tests/test_analysis.py
```
