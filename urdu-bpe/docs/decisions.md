# Decision log

Add one entry for each decision that affects the data, the tokenizer or the report. Put the newest entry last. Point to evidence files or stats rather than pasting large outputs.

## Template

### YYYY-MM-DD: short title
- **Decision:** what we will do, in one or two sentences.
- **Options considered:** each alternative and why it lost.
- **Evidence:** numbers, samples or files (`data/stats/…`, `evidence/…`, `analysis/samples/…`), or the handout section.
- **Consequence:** what this changes downstream (corpus, tokenizer, report, next assignment), and what would make us revisit it.

## Entries

### 2026-10-11: docs/assignment.md is a verbatim transcription of the course PDF
- **Decision:** We transcribed the handout PDF into `docs/assignment.md`, which CLAUDE.md names as the authoritative spec. The wording is unchanged. We added only formatting and a clearly marked "Transcription notes" section.
- **Options considered:**
  - Keep only the PDF. Rejected: later sessions cannot search or quote it reliably, and the Urdu examples would need re-extracting each time.
  - Write a summary. Rejected: summaries drift from the source.
  - Verbatim transcription with provenance. Chosen.
- **Evidence:**
  - Source: `Assignment 1 - Urdu BPE Tokenizer from Scratch.pdf`, 5 pages, sha256 `c82bc6fb7d5738402c7ad717c4fac6fe6b9d71bcbc48f2e09d311afe2cfa10a2`.
  - The Urdu examples were checked code point by code point, and the PDF contains no presentation forms.
  - Extraction shows a gap in the GPT-2 regex (`| ? \p{L}+`). It is a text-run break before each `\`, and the same artifact appears in `[^ \s`.
  - Link targets come from the PDF's annotations.
- **Consequence:** Two gaps in the original are recorded, not filled:
  - Part C says "Answer five questions" but lists three.
  - The reference trainer must pass "the sanity check below", but the handout contains none.

  Ask the instructor about both before writing the report or finalizing the reference-trainer tests.

### 2026-10-11: robots.txt is checked for every URL the downloader requests
- **Decision:** `collect/polite.py` fetches robots.txt once per origin and checks every requested URL against it, including API endpoints, using our own RFC 9309 matcher (`collect/robots.py`).
  - 4xx on robots.txt means no restrictions.
  - 5xx or no connection means disallow everything.
  - Redirect targets (signed CDN URLs) are not re-checked.
  - A disallowed URL produces manual download steps, never a fetch.
- **Options considered:**
  - `urllib.robotparser`. Rejected: it does plain prefix matching and ignores the `*` and `$` that LINDAT, Mendeley and GitHub use.
  - Skipping robots.txt for dataset downloads. Rejected: CLAUDE.md hard rule 5 makes no exception.
- **Evidence:** robots.txt files fetched 2026-10-11:
  - data.statmt.org: `User-agent: * / Disallow: /`.
  - lindat.mff.cuni.cz: disallows `/repository/handle`, which is where hdl.handle.net redirects.
  - data.mendeley.com: disallows `/api/` but not `/public-api/`.
  - github.com: disallows `/*/archive/`.
  - dumps.wikimedia.org, api.github.com and codeload.github.com: 404 (no restrictions).
  - huggingface.co: `Allow: /`.
- **Consequence:**
  - CC-100 can only be downloaded by hand.
  - UFAL's handle is resolved through LINDAT's `server/api/pid/find`.
  - Makhzan comes from `api.github.com/.../tarball/<sha>`.
  - The Step 2 scraper reuses the same client.

### 2026-10-11: each published source is pinned to an exact file and verified against its own checksum
- **Decision:** Resolvers in `collect/download.py` turn each source into exact URLs plus the checksum the source publishes:
  - Wikipedia: the dated dump named in `latest/urwiki-latest-sha1sums.txt` (SHA-1, size from `dumpstatus.json`).
  - UFAL: the bitstream's MD5 from the LINDAT API.
  - Urdu News: the file's SHA-256 from Mendeley's public API.
  - FineWeb-2: LFS SHA-256 at main's commit, recorded in the manifest.
  - Makhzan: tarball of a pinned commit (no published checksum).

  Resumes use Range plus If-Range with the saved ETag, so a file that changed upstream is fetched whole again.
- **Options considered:** Downloading `latest`/`main` URLs directly. Rejected: they can move between a resume and the next run, and the data card must name exact versions.
- **Evidence:** Dry run on 2026-10-11 resolved urwiki-20261001, LINDAT bitstream f0fc564f…, Mendeley file 60d1e75f…, and FineWeb-2 revision af9c1333…. A smoke download was killed at 6.29 MB and resumed with HTTP 206 to exactly 20,000,000 bytes, and the joined bz2 prefix decompresses cleanly (108 MB of XML).
- **Consequence:** `MANIFEST.json` and `sources.json` identify the exact bytes used. Rerunning after the next monthly Wikipedia dump starts a new dated file rather than mixing dumps.

### 2026-10-11: FineWeb-2 — download only 001_00000.parquet
- **Decision:** Download only `001_00000.parquet` (2,024,610,997 B) with `--files 001_00000.parquet`. The cleaning step must cap how much of it enters the corpus (about 1.5 GB of raw text).
- **Options considered:**
  - `--max-files 1`. Rejected: by name order that is `000_00000.parquet`, which is 4.84 GB.
  - Both files (6.87 GB). Rejected: more than 3 times the text we need.
  - Fetching individual row groups by Range. Rejected: complex, and writing a subset parquet would mean using pyarrow beyond reading.
- **Evidence:** HF tree API at af9c1333 lists only 2 train files. Parquet footers, read with Range requests:
  - `000_00000.parquet`: 3,262,000 rows, 15.15 GB of uncompressed text, dumps CC-MAIN-2013-20 to 2021-10.
  - `001_00000.parquet`: 1,547,542 rows, 6.29 GB of text, CC-MAIN-2021-10 to 2024-18.

  The card lists 4,809,542 docs and 19.93 GB of text. The Step 1 brief said 4,164,316 docs and 5.80 GB, which looks like an older revision.
- **Consequence:**
  - The web portion comes only from 2021–2024 crawls. DATA_CARD.md must list this as a known bias.
  - Need a sampling rule in Step 3 (for example, a seeded random subset of row groups) rather than "the first N rows".

### 2026-10-11: CC-100 and Makhzan are recorded but disabled
- **Decision:** Both have `enabled: false`, so `--source all` skips them. They run only when named explicitly.
- **Evidence:**
  - CC-100: data.statmt.org/robots.txt disallows everything. It was a fallback anyway.
  - Makhzan: the README says the `/text` files are "covered under standard copyright… you are not allowed to redistribute or republish this text". Our train/val/test.txt go to the course by shared-drive link.
- **Consequence:** Literature as a genre must come from elsewhere unless the instructor confirms that submitting Makhzan-derived text by link is allowed. UFAL also includes literature, and Step 2 can target public-domain books.
- **Update (same day):** The Makhzan part is superseded; see "Makhzan enabled with instructor approval" below. CC-100 stays disabled.

### 2026-10-11: findings from inspecting the smoke-test prefixes (to act on in Step 3)
- **Urdu News 1M (first 8 MB, 1,617 rows):**
  - 0% of rows contain ۔ ، ؟ or ASCII . , ? in News Text or Headline.
  - 0% contain آ (U+0622), أ, ۂ or ۓ, and no diacritics survive. ئ (U+0626) is present.
  - Only 51 distinct characters appear.
  - Example: "اسلام آباد" is stored as "اسلام باد", and sentences are glued together ("گااس").
  - So the "Urdu text and numbers only" filter also deleted letters.
  - Recheck on the full file with `collect/inspect_raw.py`. Then decide whether to drop the source, repair آ only where unambiguous, or cap its share. As it stands it would teach the tokenizer broken words.
- **UFAL (first 4 MB, 20,000 lines):**
  - Mean word-set Jaccard is 0.0390 for consecutive pairs and 0.0390 for random pairs (ratio 1.00), so sentences are shuffled and documents cannot be rebuilt.
  - The text is also pre-tokenized: there is a space before ۔ and ،.
  - Step 3 must choose a "document" unit for this source (single sentences, or fixed-size chunks) for dedup and the split. It also needs to decide whether to undo the space before punctuation.
- **Wikipedia:** The first pages are redirects and `MediaWiki:` namespace pages, so the extractor must keep ns=0 pages that are not redirects.

### 2026-10-11: Makhzan enabled with instructor approval
- **Decision:** Set `enabled: true` for `makhzan`, so `--source all` downloads it and Makhzan text may go into train/val/test.txt.
- **Options considered:** Keeping it disabled. Rejected once the instructor agreed: Makhzan is carefully edited text with intact punctuation and diacritics, unlike Urdu News 1M. (An earlier version of this entry called it "the only source of edited literary prose". Wrong: see the download findings below.)
- **Evidence:** Makhzan's README says the `/text` files are "free to use for analysis, research and development, but you are not allowed to redistribute or republish this text". The instructor was asked and approved including it, provided the shared-drive link for the corpus files stays private to course staff.
- **Consequence:**
  - The shared-drive link must stay private.
  - `data/raw/makhzan/` and any file containing Makhzan text must never be committed or published. `data/` is already gitignored.
  - Makhzan text must not be quoted in the report or in `analysis/samples/`. Pick the Part C hand-picked inputs from other sources.
  - DATA_CARD.md must list Makhzan with its licence restriction and this condition.

### 2026-10-11: Makhzan download findings (to act on in Step 3)
- **Download:** `makhzan-2c7ed4f78d98.tar.gz`, 27,043,691 bytes, SHA-256 `a8b0c47e…` (commit 2c7ed4f). No published checksum exists, so integrity was checked by streaming the whole archive: no gzip or tar errors, and all 6,315 `text/*.xml` files decode as UTF-8 and parse as XML.
- **Genre:** mostly journalism, not literature. By `<copyright-holder>`:
  - IBC Urdu (news/opinion site): 5,474 files (87%).
  - Al-Mawrid (religious essays): 630.
  - Gurmani Centre, LUMS (*Bunyad*, a literary research journal): 206.
  - Unicode, Inc.: 4.

  `sources.json` now says "edited journalism and essays". The data card's genre table must use this breakdown.
- **Extraction rules for Step 3:**
  - Take text from `<body>` only. Never take `<meta>`: it holds titles, author names and links.
  - Treat each `<p>` and `<blockquote>` paragraph as one line.
  - 1,277 files are flagged as containing non-Urdu text, marked with `<annotation>`. Decide whether to keep those spans (Arabic quotations, English terms) or drop them.
  - The download is not resumable: codeload ignores Range, so a dropped connection restarts it. That happened once, costing about 11 minutes.

### 2026-10-11: full downloads complete; smoke-test findings confirmed on the full files
- **Downloads:** 5 sources, 4,846,213,387 bytes in total (formats as published). Sizes on disk match MANIFEST.json for every file.
  - Published checksums verified: Wikipedia urwiki-20261001 (SHA-1), UFAL (MD5), Urdu News 1M (SHA-256), FineWeb-2 `001_00000.parquet` (SHA-256).
  - Makhzan has no published checksum; the archive streamed end to end without errors.
  - Resume worked on real drops: UFAL lost its connection 6 times and Wikipedia twice. Each resumed with Range, and both still match their published checksums.
- **Urdu News 1M:** 1,038,340 rows, read in full (the last row's Index is 1038339). In a seeded random sample of 1,000 rows:
  - 0% contain ۔ ، ؟ or ASCII . , ? in News Text or Headline.
  - 0% contain آ, أ, ۂ or ۓ, and none have diacritics. ئ appears in 99.6% of News Text samples.
  - Only 51 distinct characters appear.

  The smoke-test finding holds for the whole file: sentence boundaries and alef madda were stripped at source. The Step 3 decision (drop, repair or cap) is still open.
- **UFAL:** 5,464,575 lines, matching the repository metadata. The order test at five offsets (0, 1.0M, 2.5M, 4.0M, 5.3M; 20,000 lines each) gives ratios of 0.99–1.00, so the sentences are shuffled throughout and there are no documents to recover.
- **FineWeb-2 `001_00000.parquet`:** 1,547,542 rows in 1,548 row groups, with about 6.29 GB of uncompressed text, as the footers predicted.
- **Evidence:** `logs/inspect_raw.txt` (local, gitignored; regenerate with `python collect/inspect_raw.py`).
