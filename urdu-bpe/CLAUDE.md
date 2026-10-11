# Urdu corpus + byte-level BPE (course assignment)

## What we are building
Part A: at least 500 MB (target about 1 GB) of cleaned Urdu text, split by document into data/final/train.txt, val.txt and test.txt (98/1/1).
Part B: a byte-level BPE tokenizer in plain Python (bpe.py, class BPETokenizer), trained on train.txt to exactly 50,000 tokens; train.txt and val.txt encoded to train.bin and val.bin.
Part C: an analysis of what the tokenizer learned and how it compares with GPT-2 and o200k_base, written up in report/report.pdf (max 4 pages).
These outputs feed the next assignment (training a GPT on train.bin and val.bin), so every data or tokenizer flaw carries forward. Prefer correctness and data quality over speed.
The authoritative spec is docs/assignment.md. When a requirement is unclear, read it there. Do not guess.

## Hard rules
1. Implementation code (repo-root *.py, collect/, clean/) may import only: the standard library, regex, numpy, tqdm, requests, bs4, lxml, pyarrow. pyarrow is used only to read downloaded FineWeb-2 parquet files.
2. Never use or copy from tokenizers, sentencepiece, tiktoken, minbpe, transformers tokenizers, or any published BPE code. Urdu NLP libraries must not do our normalization.
3. tiktoken and tokenizers appear only in analysis/compare_tokenizers.py. matplotlib only in analysis/. pytest only in tests/. tests/test_forbidden_imports.py enforces this; keep it green.
4. normalize.py is applied while cleaning the corpus. The tokenizer never normalizes.
5. Collection: obey robots.txt and terms of use; skip anything behind a login or paywall; collect no personal data (no comments, private messages, social profiles or author emails); never redistribute corpus text outside the course.

## Tokenizer spec
- IDs 0-255 are the 256 byte values. <|endoftext|> is ID 256: never split, never part of a merge.
- Merges get IDs 257-49,999 in the order learned: exactly 49,743 merges.
- Pre-tokenize with a regex so merges never cross word boundaries. A leading space belongs to the following word (GPT-2 style). Combining marks (\p{M}) and ZWNJ (U+200C) stay inside words.
- Pair counts include overlaps ("aaa" holds (a,a) twice). A merge replaces non-overlapping occurrences left to right.
- Ties: merge the GREATER pair, comparing (vocab[left], vocab[right]) as a tuple of Python bytes. Never compare by ID.
- decode(encode(s)) == s for every string: emoji, English, newlines, the empty string.
- The reference trainer (recount all pairs after every merge) and the fast trainer must learn identical merges on the 1 MB sample.

## Corpus spec
- At least 3 sources from at least 2 genres; at least 50 MB collected by us via scraping or an API, with evidence in evidence/; licence or terms recorded for every source in collect/sources.json.
- Order: extract -> normalize -> line language filter (keep natural code-mixing) -> boilerplate-line removal -> exact document dedup (+ optional near-dedup) -> split.
- Split by document, 98/1/1, seed 1337, stratified by source. Write each document followed by <|endoftext|>. Remove any literal <|endoftext|> inside document text first.
- Every stage writes per-source stats (docs, lines, bytes in/out) to data/stats/. DATA_CARD.md tables are generated from them.

## Definitions
- MB = 10^6 bytes of UTF-8 (also report MiB).
- word = whitespace-separated token (str.split()); used for corpus totals and tokens-per-word.
- unique words = distinct maximal runs of [\p{L}\p{M}‌].

## Layout
Root: normalize.py, bpe.py, train_tokenizer.py, encode_corpus.py (required names), common.py helpers.
collect/ downloaders, scraper, sources.json | clean/ extractors, audit, pipeline, split, data card | analysis/ Part C scripts | report/ LaTeX report | evidence/ scrape logs and robots/terms snapshots | tokenizer/ saved tokenizer | data/{raw,interim,clean,final,stats,samples}.

## How we work
- Python 3.12, type hints, argparse CLIs with --help, logging via common.setup_logging.
- Stream big files. Never read a 1 GB file into memory unless a step truly needs it.
- Long jobs (downloads, scraping, full cleaning, training, encoding) are not run inside Claude Code. Write the script, smoke-test it with --limit or --sample-mb, then print the exact tmux command for me, logging to logs/.
- Every module gets tests in tests/. Run pytest -q before calling a step done.
- Log each decision with its evidence in docs/decisions.md.
- data/ and logs/ are gitignored. Never commit data.
