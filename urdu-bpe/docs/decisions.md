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

### 2026-10-11: the scraper fetches a URL only if both robots.txt parsers allow it
- **Decision:** `collect/scrape/fetch.py` (ScrapeClient) checks every URL, each redirect hop included, with our RFC 9309 matcher (`collect/robots.py`) **and** `urllib.robotparser.can_fetch`. A URL is requested only if both allow it for our product token. The probe reports each parser's answer separately, for User-agent `*` and for our token.
- **Options considered:**
  - `urllib.robotparser` alone, as the Step 2 brief says. Rejected as the only check: it treats `*` and `$` literally, so `Disallow: /*?print=` blocks nothing.
  - Our matcher alone (the Step 1 decision). Rejected as the only check: the brief asks for robotparser, and robotparser's first-match rule is sometimes stricter (`Disallow: /` before `Allow: /news/`).
  - Both must allow. Chosen (approved 2026-10-11): it is never less strict than either parser.
- **Evidence:** `tests/test_scrape_fetch.py`: `test_wildcard_rule_blocks_although_stdlib_parser_would_allow` and `test_stdlib_first_match_blocks_although_rfc_would_allow`.
- **Consequence:** A site whose robots.txt the two parsers read differently loses the disputed URLs. The probe report lists any disagreements.

### 2026-10-11: scraper etiquette beyond robots.txt
- **Decision:** User-Agent `UrduCorpusCourseBot/1.0 (NUST SEECS coursework; contact: …)`. The contact comes from `collect/contact.local`, as for the downloader.
  - Delay: max(Crawl-delay, 1 s) plus 0–0.5 s of jitter.
  - One connection per host, no concurrency, no cookies (an empty cookie-policy allow-list), no login.
  - Redirects are followed by hand, at most 5 hops. Each hop is robots-checked and must stay on the site.
  - 429/503 back off exponentially and honour Retry-After. A Retry-After above the cap stops the site rather than retrying early.
  - A bot challenge (Cloudflare, Sucuri, Imperva, DataDome, Akamai, AWS WAF) stops the site at once, and so do 20 consecutive 403s.
- **Options considered:** Letting requests follow redirects. Rejected: the hops would skip the robots check, and redirect responses' cookies would be sent on the next hop.
- **Evidence:** `tests/test_scrape_fetch.py` (cookies, pacing, Retry-After, challenge and 403 tests).
- **Consequence:** Every request, robots-blocked URLs included, appears in an `evidence/*log*.jsonl.gz` file with url, UTC time, status, bytes, robots_allowed and elapsed_ms.

### 2026-10-11: own collection from VOA Urdu (70 MB target) and Hum Sub (20 MB target)
- **Decision:** Crawl VOA Urdu (`--target-mb 70`) and Hum Sub (`--target-mb 20`) with `collect/scrape/crawl.py`. Chosen by the user on 2026-10-11 from the Phase A probe.
  - VOA Urdu: news. Its own text is US-government work in the public domain.
  - Hum Sub: opinion and blogs, a genre our other sources barely cover.
- **Options considered** (evidence/probe_report.md; each site got at most 30 requests per probe run, plus targeted re-runs):
  - BBC Urdu. Rejected: robots.txt comments forbid scraping, crawling, AI training on its content, creating datasets and text and data mining.
  - DW Urdu. Rejected: robots.txt comments prohibit "any device, tool, or process designed to data mine or scrape the content using automated means… without prior written permission".
  - Express (~490k articles, no Crawl-delay). Usable, but copyrighted news ("all rights reserved"), with no terms page and no genre gain. It is the fallback if VOA falls short.
  - Independent Urdu and Urdu News (Crawl-delay 10). Their sitemaps do not list articles, so crawling would need category-page pagination. Independent Urdu's terms page is empty; Urdu News's terms link points to `#`.
- **Evidence:** `evidence/probe_report.md`, `evidence/probe_20261011.json`, `evidence/robots/*_20261011.txt`, `evidence/terms/*_20261011.txt`, `evidence/probe_log_20261011.jsonl.gz`.
- **Consequence:** About 90 MB of extracted text, above the 50 MB requirement after cleaning and dedup.
  - VOA's archive ends at the 2025 shutdown.
  - Hum Sub's 20 s Crawl-delay makes its crawl take about a day.
  - DATA_CARD.md lists both sites with their terms summaries (sources.json `own_scrape`).

### 2026-10-11: Hum Sub reprints from BBC and DW are skipped
- **Decision:** The Hum Sub adapter skips posts in the `bbc`, `dw` or `rsstest` categories (feeds imported from other outlets) and posts credited to BBC or DW as author. `skips.jsonl` and the scrape report count them.
- **Options considered:**
  - Keeping them. Rejected: BBC's and DW's stated terms forbid dataset creation, AI training and scraping of their content. Taking that content through a third site would defeat the reason we excluded them.
  - Skipping all newsdesk posts. Rejected: Hum Sub's own newsdesk items carry no such restriction.
- **Evidence:** 2 of the 10 Hum Sub probe samples were BBC reprints (`category-bbc category-rsstest`, meta author بی بی سی, "BBC"). Tests: `tests/test_scrape_sites.py`.
- **Consequence:** Hum Sub yields roughly 20% fewer usable posts, still far more than the 20 MB target. Columns reprinted from newspapers with a بشکریہ ("courtesy of") credit are kept: those outlets publish no such restriction.

### 2026-10-11: crawl order, resume and what counts toward the target
- **Decision:**
  - Crawl order: URLs sorted by sha1("1337:<url>"). This is a seeded pseudo-random order over the whole sitemap archive, and a URL's position does not depend on which other URLs were found.
  - Target: `--target-mb` counts the UTF-8 bytes of `title + "\n" + body` in articles.jsonl, before cleaning.
  - Skipped pages: media pages (VOA video and audio), reprints, pages with too little text and duplicate canonical URLs are skipped and logged.
  - Resume: discovery resumes per sitemap, and articles resume per URL. Cached HTML is re-extracted rather than fetched again, and 404s and robots-blocked URLs are never retried.
- **Options considered:**
  - Newest-first order. Rejected: 70 MB would cover only the last year or two.
  - Python `random.shuffle` over the list. Rejected: the order changes whenever discovery finds one more URL.
- **Evidence:** `tests/test_scrape_crawl.py`. VOA smoke run, 50 articles: 58 fetched, 8 media pages skipped, 0.35 MB (mean 7.0 KB per article, against a probe median of 2.7 KB), 61 requests in 71 s.
- **Consequence:** The full VOA crawl needs about 10,000 articles, roughly 4 h at ~1.2 s per request. The probe's 11 h estimate came from its median article size.

### 2026-10-11: extractor v2 reads text outside <p> (VOA stories from about 2012-2014)
- **Decision:** `text_blocks()` (`collect/scrape/extract.py`) now also collects loose text in the article container: runs of text nodes and inline elements, broken at `<br>` and at the edges of block-level elements, alongside `<p>`, `<h2>`, `<h3>` and `<li>`.
  - VOA also drops its "SEE ALSO" cards (`.also-read` in `div[data-owner-ct]`) and stand-alone datelines.
  - Extractors are now `voa_urdu/2` and `humsub/2`.
  - Both crawls were stopped, re-extracted from the HTML cache with `--reextract` (no requests) and restarted.
- **Options considered:** Adding `span` to the block tags. Rejected: it would split sentences that use `<span>` for emphasis inside a `<p>`.
- **Evidence:** In the first 20 minutes of the full VOA crawl, 41 of 333 pages were skipped as "too little text". Their `.wsw` held 400 to 4,900 Arabic letters, written as `<span>`s separated by `<br><br>` or as a bare `<div>`, with no `<p>`. On all 452 cached VOA pages, v2 keeps 366 against v1's 317 (49 recovered). Of the pages both versions keep, only 2 changed, each gaining an `<h4>` subheading or a lead line. Hum Sub's 10 pages are unchanged. Test: `tests/test_scrape_sites.py::test_voa_old_layout_without_paragraphs`.
- **Consequence:** Older VOA years are no longer silently dropped, which would have skewed the date range toward 2015 and later. `<h4>`–`<h6>` subheadings are now part of the body.

### 2026-10-11: Hum Sub crawl capped at 7 hours
- **Decision:** Hum Sub runs with `--target-mb 20 --max-hours 7` (new crawler option: stop after 7 h of wall-clock time, discovery included), so it ends about when VOA does. User's request, 2026-10-11.
- **Options considered:**
  - Keeping the 20 MB target. Rejected by the user: at Hum Sub's robots.txt Crawl-delay of 20 s it would take about 21–31 h.
  - Ignoring or shortening the Crawl-delay. Not an option: robots.txt is binding (CLAUDE.md, hard rule 5).
- **Evidence:** Measured spacing between Hum Sub requests: minimum 20.07 s, mean 20.28 s (`evidence/crawl_log_humsub.jsonl.gz`). Discovering all 136 post sitemaps takes about 46 min of the budget.
- **Consequence:**
  - Hum Sub should yield roughly 900–1,100 posts, about 4–7 MB of opinion and blog text, a minority genre in the corpus.
  - VOA's 70 MB alone covers the 50 MB requirement.
  - DATA_CARD.md must state Hum Sub's small share.
  - Rerunning the same command later adds more posts, since the crawl resumes.

### 2026-10-12: own collection finished: 77.16 MB (VOA Urdu 70.01 MB, Hum Sub 7.16 MB)
- **Decision:** Accept the crawls as our own collection. Hum Sub's extractor goes to `humsub/3`, which drops the page links of multi-page posts (`.pagination` and the "مزید پڑھنے کے لیے اگلا صفحہ کا بٹن دبائیں" line, "press the next-page button to read more"). `pagination` and `page-links` join the generic junk list. Only page 1 of a multi-page post is kept.
- **Options considered:** Fetching pages 2+ of multi-page posts. Rejected for now: it affects about 1% of posts (7 of 843) and would need crawler support for continuation pages.
- **Evidence** (`evidence/scrape_report.md`):
  - VOA Urdu: 16,325 articles from 21,849 fetched article URLs. 5,448 were media pages; 9 were connection errors after retries.
  - Hum Sub: 843 posts from 1,145 fetched. 296 were BBC/DW/RSS reprints, 26% of fetches.
  - No bot challenges, and no request refused by robots.txt.
  - VOA publication dates 2010–2025-03, about 1,000 a year. Hum Sub 2016–2025.
  - No VOA body line repeats in 0.5% or more of documents.
  - Re-extracting VOA with the new generic list gives a byte-identical articles.jsonl (SHA-256 `23a640e4…`).
- **Consequence:** Step 3 reads `data/raw/own_scrape/{voa_urdu,humsub}/articles.jsonl`. The remaining repeated lines are newspaper credits (بشکریہ … "courtesy of …") and "(جاری ہے)" ("to be continued"), left for the boilerplate-line step. The DATA_CARD.md genre table must show Hum Sub as about 9% of our own collection.

### 2026-10-12: Step 3 extraction format and per-source document units
- **Decision:** Every source is extracted to `data/interim/<source>.jsonl.gz` with one schema (`doc_id, source, genre, url, title, date, text, meta`), with no normalization and no filtering beyond what each container needs. Strings that are not valid UTF-8 are dropped and counted as `utf8_error`. A deterministic ~2 MB sample per source goes to `data/samples/`: the documents with the smallest blake2b-64 of `doc_id`, written in corpus order, so the same documents come out whatever the read order. Document units:
  - **wiki_ur:** one article (title + body).
  - **ufal_umc:** one sentence per document (`--group auto` picks 1, because the order test ratio is below 1.5). Step 5 needs a lower `--min-words` for this source.
  - **urdu_news_1m:** headline + story. `meta.punct_stripped` is set per story: true when it holds none of ۔ ، ؟ . , ? !
  - **fineweb2_urd:** one row. `--keep-per-mille K` keeps `int(sha1(id), 16) % 1000 < K`, to be tuned in Step 6.
  - **makhzan:** one XML file, `<body>` only; each `<p>`, `<heading>` or `<blockquote>` becomes one line.
  - **own_scrape:** one article (title + body). Repeated canonical URLs are dropped.
- **Options considered:**
  - UFAL as 32-line pseudo-documents. Rejected: consecutive lines are unrelated, so the chunks would be random sentence bags. Dedup and the split would also treat 32 unrelated sentences as one unit.
  - Makhzan titles in the text. Rejected: the 2026-10-11 Makhzan rule says never to take `<meta>`. Only the publication name, year, copyright holder and non-Urdu flag go into `meta`, for the data card's genre breakdown. Author names and links are never read.
  - Makhzan `<annotation>` spans (non-Urdu passages) dropped at extraction. Rejected: they sit inside Urdu sentences, so they are code-mixing, and the Step 5 line filter judges them like any other line.
- **Evidence:**
  - UFAL: Step 1 order test, mean word-set Jaccard 0.0390 for consecutive pairs vs 0.0390 for random pairs (ratio 1.00), at five offsets (ratios 0.99–1.00). `extract_ufal.py` reruns it and writes both numbers to `data/stats/extract.json` (`order_jaccard_consecutive`, `order_jaccard_random`).
  - Urdu News 1M: Step 1 found 0% of 1,000 sampled rows with any sentence punctuation, so `punct_stripped` will be true for nearly all of them.
  - Tests: `tests/test_extract.py`.
- **Consequence:**
  - Step 5's minimum-words filter must be set per source (UFAL sentences are short).
  - Step 6 decides the fate of Urdu News 1M from `punct_stripped`.
  - UFAL's space before ۔ and ، (Step 1) is left as is here and comes up in Step 4.

### 2026-10-12: Wikipedia markup is stripped by our own scanner, and disambiguation pages are detected on the raw wikitext
- **Decision:** `clean/wikitext.py` strips markup with the standard library and `regex` only, in the order given in its docstring:
  - comments, then refs, then dropped tags, then the back-matter cut, then templates and tables, then external links, then internal links, then emphasis and headings, then tags and entities, then lines.
  - Templates and tables share one depth scanner. A `{{` or `{|` that is never closed is treated as text (MediaWiki shows it as text), so one broken template cannot swallow the rest of an article.
  - Internal links are resolved innermost first, so a file caption's own links are resolved before the file link is dropped whole.
  - Lines starting with `|` (table or infobox residue) are dropped along with empty and punctuation-only lines.
  - A page is a disambiguation page if, on the raw wikitext, it has a template named ضد ابہام, Disambiguation, disambig*, dab or dmbox, or a category containing ضد ابہام or "disambiguation", or a title containing "(ضد ابہام)". Name comparisons fold ي/ى→ی, ك→ک, ه→ہ, `_`→space and case. Each method is counted separately (`drop_disambiguation_template`, `_category`, `_title`).
- **Options considered:**
  - mwparserfromhell or WikiExtractor. Not allowed by the brief.
  - A regex for `{{…}}` alone. Rejected: templates nest, and tables sit inside templates.
  - `Pool.imap` over page batches. Replaced by `common.ordered_imap`: the same ordering, but at most 4 × jobs batches in flight. `Pool.imap`'s feeder thread would read the whole dump into its task queue.
- **Evidence:**
  - `tests/test_wikitext.py`: 80 hand-written fixtures covering every rule, including nested templates, a table inside a template, a file link whose caption contains links, the back-matter cut, and redirect and disambiguation detection.
  - Stripping runs at about 20 MB/s per core on 141 KB synthetic pages, so XML parsing in the main process is the bottleneck.
- **Consequence:** The detection counts per method go into `data/stats/extract.json` for the data card. After the first full run, spot-check 20 dropped `short_body` pages and 20 kept articles with `clean/peek.py --raw`.

### 2026-10-12: Step 4: Urdu normalization rules
- **Decision:** `normalize.py` applies, in this order:
  1. line breaks to LF;
  2. Arabic presentation forms to the letters they display (from their positional compatibility decompositions), except the phrase ligatures ﷺ (U+FDFA) and ﷻ (U+FDFB); ﷲ becomes اللہ;
  3. removal of tatweel, ZWJ, ZWSP, word joiner, LRM/RLM/ALM, bidi embeddings, overrides and isolates, BOM, soft hyphen and C0/C1 controls;
  4. ZWNJ kept only between two Arabic-script letters, removed elsewhere;
  5. NFC;
  6. ي/ى → ی, ك → ک, ه/ە → ہ, ة → ۃ, ۀ → ۂ, and Arabic-Indic and Urdu digits → ASCII, then NFC again;
  7. ٫ and ٬ between digits → `.` and `,`;
  8. whitespace: Unicode spaces, tab, VT and FF → space; no space before ۔ ، ؟ ؛; one space between words; trimmed lines; no empty lines.

  The module docstring gives the reason for the order. `normalize` is idempotent.
- **Options considered:**
  - **Digits.** Urdu digits ۰–۹ everywhere. Rejected: ASCII digits dominate five of the six sources in the audit (4,064–15,842 per million characters). Only Makhzan uses more ۰–۹ (2,736 per million) than ASCII (2,214), and Arabic-Indic digits are rare everywhere (at most 215 per million, in Wikipedia). ASCII also costs 1 byte per digit instead of 2, and matches the GPT-2/o200k comparison in Part C.
  - **ZWNJ.**
    - Remove it everywhere. Rejected: Makhzan uses it on purpose inside compounds. Of 47 ZWNJs in its sample, 34 are ت+ZWNJ+ح/گ/خ (صورت + ZWNJ + حال, …).
    - Keep it everywhere. Rejected: the stray ones (next to a space, a newline or punctuation in Wikipedia) are invisible noise.

    The CLAUDE.md pre-tokenizer spec already keeps ZWNJ inside words.
  - **Full NFKC.** Rejected: it would also rewrite full-width Latin, superscripts and ligatures that are not Urdu typing variants, and it expands ﷺ into a four-word phrase that does not look like the symbol. The positional table changes only glyph-shape variants of the same letters.
  - **Swash kaf ڪ → ک.** Rejected: ڪ is a Sindhi letter, and the Sindhi letters are what the Step 5 filter needs to find Sindhi text (see the next entry).
  - **ASCII `.` and `,` after Urdu → ۔ and ،.** Rejected for now: they look different, so they are not invisible typing variants, and abbreviations, decimals and URLs make the mapping a guess. FineWeb has 497 per million characters.
  - **Gol heh ہ vs do-chashmi ھ.** Kept apart, as the handout says. Makhzan's older IBC texts write ھ for ہ (ھو گئے, ھیں); that is spelling, not a code-point variant.
  - **NFC through `unicodedata`.** Kept: it is the standard library's Unicode table, not an Urdu NLP library, and every Urdu-specific rule is our own code.
- **Evidence:**
  - `clean/audit_chars.py` on the 2 MB samples gives per-source counts and contexts in `data/stats/char_audit.json`. Highlights:
    - Arabic yeh: 876–2,234 per million characters in UFAL, own_scrape and Wikipedia.
    - Presentation forms: 3,039 per million in Makhzan (whole documents typed in them).
    - Decomposed ا + ٓ: 1,374 in the Makhzan sample.
    - Space before ۔ or ،: 11,752 per million in UFAL and 1,979 in FineWeb.
  - `python clean/audit_chars.py --normalized` (`data/stats/char_audit_normalized.json`): every category the rules target is 0 per million in every source, and no document changes under NFC. What remains is deliberate:
    - ڪ in Sindhi text;
    - ۃ in Arabic loanwords;
    - 44 in-word ZWNJs;
    - ﷺ and the ornate parentheses ﴾﴿;
    - a few malformed hamza sequences with no precomposed form (یٔ, a lone ٕ after a space).
  - On the full interim files, normalization removes 0.10% of own_scrape bytes (8,837 of 17,168 docs change) and 0.42% of Makhzan (4,962 of 6,301 docs).
  - Speed: 11–19 MB/s on one core.
  - Tests: `tests/test_normalize.py`, covering every rule and the ordering traps (ي + ٔ → ئ, ه + ٔ → ۂ, an LRM between ا and ٓ), plus idempotence and output invariants on 5,000 seeded random strings.
- **Consequence:**
  - The tokenizer sees one code point per letter and digit.
  - Harakat and ۃ stay, so the vocabulary will hold some Arabic-quotation tokens.
  - The data card's cleaning table gets a normalization row from the stage stats.
  - Revisit the ASCII-punctuation decision if Part C shows `.` and `۔` splitting the same sentence-final merges.

### 2026-10-12: VOA Urdu contains Sindhi articles (to act on in Step 5)
- **Finding:** 217 of the 16,325 VOA Urdu articles are written in Sindhi, not Urdu. They hold 0.99 MB of text, all under `www.urduvoa.com/a/`, and more than 1% of their characters are Sindhi-only letters (ڪ ٽ ڏ ڀ ٻ ڄ ڃ ڇ ڻ ٿ ڦ). They read as Sindhi throughout, for example "پاڪستان جي پرڏيهي وزير". One Hum Sub post contains some Sindhi letters. The FineWeb sample also has Sindhi passages (ڪ: 14 in the normalized sample).
- **Consequence:**
  - Step 5's line language filter must reject Sindhi. Sindhi-only letters are the signal, which is why `normalize` keeps ڪ.
  - Our own collection loses about 1 MB, leaving about 76 MB, still above the 50 MB requirement.
  - DATA_CARD.md should say that the scrape contained Sindhi pages and that they were removed.

### 2026-10-12: Step 3 full run results; Wikipedia short_body cutoff kept at 200 characters
- **Decision:** Keep `--min-body-chars 200` for Wikipedia.
- **Full run** (`clean/extract_all.py --wiki-jobs 7`, about 17 minutes on an 8-core M-series Mac with 8 GB RAM; `data/stats/extract.json`):

  | Source | Docs | MB |
  |---|---|---|
  | wiki_ur | 302,658 | 539.24 |
  | ufal_umc | 5,464,575 | 711.94 |
  | urdu_news_1m | 1,038,340 | 1,962.36 |
  | fineweb2_urd | 1,547,542 | 6,277.12 |
  | makhzan | 6,301 | 52.93 |
  | own_scrape | 17,168 | 77.16 |
  | **Total** | **8,376,584** | **9,620.76** |

  Wikipedia drops: 1,098,627 redirects, 926,072 non-article pages, 399,290 short bodies, and 2,585 disambiguation pages (template 2,063, category 507, title 15).
- **Options considered:** Lower the cutoff to about 100 characters. Rejected:
  - The extra text is small: dropped bodies average about 110 characters, so about 85 MB in total.
  - Much of it is bot output, which the boilerplate step would partly remove anyway.
  - We have 9.6 GB extracted against a 0.5–1 GB target, so size is not the constraint.
- **Evidence:** Spot-check of `short_body` drops among the first 60,000 candidate pages: 30,283 dropped and 28,677 kept. Body lengths of the drops: under 50 chars 2,402; 50–99 11,261; 100–149 9,296; 150–199 7,324. Of 11 random drops:
  - 4 are year pages whose body is only the twelve month names (97ء, 538ء, 934ء, 1597ء);
  - 2 are infobox-only articles with an empty body (یوروپیئم: 2,756 raw chars, 0 after stripping);
  - 1 is a list of meanings (عسکری);
  - 4 are clean one- or two-sentence definitions (تبرا, طلوع, رب العالمین, آمریت).
- **Consequence:**
  - FineWeb is 65% of the extracted text, so Step 6 must set `--keep-per-mille` (or cap FineWeb another way) to keep the genre balance.
  - Spot-check of 20 kept articles (`clean/peek.py` reservoir, seed 1337): no markup left, but 9 of 20 are bot-generated template stubs whose sentences differ only in names and numbers:
    - 4 asteroids ("… نظام شمسی کا ایک سیارچہ ہے۔ اس کا عارضی نام … تھا");
    - 2 railway stations;
    - 2 footballers;
    - 1 US census place.

    Exact line dedup will not catch these. The dedup step (Step 6/7) should measure them, e.g. by counting articles that share the first 40 characters after the title with the title masked, and decide whether to cap them.
  - Removed templates sometimes leave a hole: "کیمیائی فارمولا  ہے" (formula template) and "لین فونٹین  6 دسمبر 18 - جولائی 30، 1983)" (date template). The text is still readable, and normalization collapses the double space.

### 2026-10-12: Step 5: line language filter and document rules
- **Decision:** `clean/normalize_filter.py` normalizes each document and then classifies each line with `clean/linefilter.py`.
  - **Line rules:**
    - drop lines with no letters;
    - drop lines where fewer than half the letter-bearing words are in Arabic script, unless they hold at least 2 Urdu function words (this keeps code-mixing);
    - drop Sindhi lines: at least 2 Sindhi-only characters, making up at least 1% of the line's Arabic-script letters;
    - drop Arabic or Persian lines only on positive evidence: at least 12 Arabic-script letters, no Urdu-only letter (ٹ ڈ ڑ ں ے ۓ), no Urdu function word, and either at least 2 Arabic/Persian function words or harakat on at least 25% of the letters.
  - **Document rules:**
    - drop a document when Sindhi lines are at least 72% of its normalized bytes;
    - drop a document when no line is left;
    - drop a document with fewer than 10 words (3 for UFAL, whose documents are single sentences).
- **Options considered:**
  - Drop Arabic-script lines with no Urdu function word. Rejected: 323 of 7,770 Wikipedia sample lines and 81 FineWeb lines match, and most are Urdu headlines, name lines and list items ("محسنِ پاکستان ڈاکٹر عبد القدیر خان").
  - Drop lines lacking ٹ ڈ ڑ ں ے and function words. Rejected: it removed 1.4% of Wikipedia bytes, including Urdu headlines ("طلبہ و طالبات پریشان، شدید احتجاج") and poetry.
  - Letter share instead of word share for the script test. Rejected: Urdu sentences full of English club names fall under 30% Arabic-script letters but are Urdu.
  - Counting distinct foreign words. Rejected: Arabic repeats فی and من, so real quotes were kept, while this rule only fixed "سی این این" (CNN), which removing این from the list fixes too. علی and آن are not on the list because they are Urdu words.
  - A 20% Sindhi share for whole documents. Rejected: it dropped the Urdu Wikipedia article on G. M. Syed (673 Urdu words) for its bibliography of Sindhi titles.
  - A minimum of 20 words. Rejected: it dropped clean 14–18-word Makhzan video blurbs. Below 10 words only fragments remain ("بشکریہ : روزنامہ جنگ").
- **Evidence:**
  - Sindhi share per document, over all of own_scrape and the first 60,000 Wikipedia articles, is bimodal: the 217 VOA Sindhi articles are at 0.785 or more (216 at 0.9 or more), and Urdu articles about Sindh and Saraiki reach at most 0.655 (سرائیکی حروف تہجی).
  - Word counts per document after line filtering (samples): p1 is 37 words for Wikipedia, 52 for FineWeb, 64 for Makhzan and 51 for news. UFAL's median is 12, and 1.8% of its sentences have fewer than 3 words.
  - Smoke run, 20,000 docs per source (all of Makhzan and own_scrape): 377.33 MB in, 376.78 MB after normalization, 373.16 MB out (1.10% removed). Biggest drops:
    - own_scrape: sindhi 0.98 MB, and 217 documents dropped (exactly the Sindhi articles);
    - Makhzan: arabic_or_persian 0.79 MB (Quran and hadith in Al-Mawrid essays);
    - Wikipedia: arabic_or_persian 0.53 MB and mostly_not_urdu_script 0.40 MB;
    - Urdu News 1M: nothing removed.
  - Dropped-line samples (`data/samples/dropped_lines/`) are almost all Quranic or hadith Arabic, Persian verse, English boilerplate ("EPAPER", "Last Updated: …") and number-only lines. Known false positives: Urdu citation lines with an Arabic book title and the Persian از ("اقتباس : معالم فی الطریق از سید قطب ( اردو ترجمہ … )"), and short vowelled Urdu lines.
  - Tests: `tests/test_linefilter.py`, `tests/test_normalize_filter.py`.
- **Consequence:**
  - Our own collection ends at about 76 MB (above 50 MB).
  - The data card's cleaning table gets the normalize and langfilter rows from `data/stats/`.
  - Short Arabic quotes without harakat inside Urdu religious texts stay; they are code-mixing.
  - Revisit the thresholds if the full-run dropped-line samples show a new false-positive class.

### 2026-10-12: Step 5b: boilerplate cutoff 5 documents (3 for own_scrape); exact dedup across sources in priority order
- **Decision:**
  - A line is boilerplate, and every copy of it is removed, when it appears in at least 5 documents of its source; for own_scrape the cutoff is 3.
  - Exact duplicate documents (same text after boilerplate removal) are dropped within a source and across sources. Sources are processed in the order own_scrape, makhzan, wiki_ur, urdu_news_1m, ufal_umc, fineweb2_urd, so the copy in the earlier source is kept: our own collection first, the web crawl last.
  - Literal `<|endoftext|>` is removed from the text here.
- **Options considered:**
  - **Cutoff 2–3 everywhere.** Rejected. Lines in 2–4 documents are mostly real content: duplicated paragraphs, quoted verses, recurring sentences in series. In Makhzan, cutoff 2 would remove 7.16% of the text against 0.11% at cutoff 5.
  - **Cutoff 10–100 for FineWeb, since it has 1.5M documents.** Rejected. Lines in 5–49 FineWeb documents are still navigation:
    - related-headline lists ("- امریکا کے مختلف شہروں میں فائرنگ سے 6 افراد ہلاک، 17 زخمی", 39 docs);
    - WordPress and app spam;
    - newsroom datelines.

    FineWeb will be subsampled to its budget anyway, so precision matters more than size.
  - **Near-duplicate detection** (MinHash or similar). Not done now. The handout calls it optional. What it would catch: about 300 MB of FineWeb lines shared by 2–4 documents (syndicated news), Urdu News 1M stories repeated under different headlines, and Wikipedia bot stubs.
- **Evidence:** `logs/dedup_report.log` (`clean/dedup.py --report` on the full Step 5 output). Distinct lines and the share of each source's line text removed:

  | Source | Cutoff 2 | Cutoff 3 | Cutoff 5 | Cutoff 10 | Cutoff 100 |
  |---|---|---|---|---|---|
  | fineweb2_urd | 9.43% | 5.91% | 4.63% (286.65 MB) | 3.75% | 1.25% |
  | wiki_ur | 5.27% | 4.08% | 3.68% (19.56 MB) | 3.40% | 2.92% |
  | urdu_news_1m | 1.45% | 0.34% | 0.08% | 0.03% | 0.01% |
  | ufal_umc | 0.68% | 0.27% | 0.03% | 0.01% | 0.00% |
  | makhzan | 7.16% | 0.95% | 0.11% | 0.06% | 0.00% |
  | own_scrape | 0.45% | 0.09% (0.07 MB) | 0.04% | 0.02% | 0.00% |

  The most frequent removed lines:
  - **Wikipedia:** "مداری خصوصیات" (asteroid stubs, 26,587 docs), a chemical-compound safety sentence (11,939), and section headings ("انتقال" 5,490, "تصانیف" 2,412).
  - **FineWeb:** "خلیج اردو" 1,873, "متعلقہ خبریں" 1,739, "پرائیویسی پولیسی" 265.
  - **Urdu News 1M:** market headlines ("امریکا میں خام تیل کے نرخوں میں اضافہ" 164) and "سپورٹس ڈیسک" 277.
  - **UFAL:** bibliography lines ("بیروت، لبنان : دار الکتب العلمیہ۔" 123).
  - **own_scrape:** wire credits ("آڈیو رپورٹ سنیئے:" 69; "اس رپورٹ کا مواد اے ایف پی سے لیا گیا ہے۔" 27) and reporter bylines.
- **Consequence:**
  - Expected removal is about 308 MB of lines (3.2%), plus documents that fall under the word minimum and exact duplicates; the full run gives the final numbers.
  - Removing VOA bylines also serves the no-personal-data rule.
  - Wikipedia keeps about 26,600 asteroid stubs minus their heading, so bot text is still over-represented; this goes in the data card's known gaps.
- **Full run result** (`logs/dedup_run.log`, `data/stats/boilerplate.json`, `dedup.json`; about 23 minutes in one process): 9,546.90 MB in, 9,194.65 MB out (3.69% removed); 8,276,053 documents in, 8,219,948 out.
  - Boilerplate lines removed: 309.99 MB (FineWeb 288.41, Wikipedia 19.66, news 1.59, UFAL 0.20, own_scrape 0.07, Makhzan 0.06).
  - Exact duplicates dropped within a source: FineWeb 18,848 (33.05 MB), UFAL 28,559 (2.59 MB), news 1,867 (2.74 MB), Makhzan 35 (1.00 MB), Wikipedia 5, own_scrape 1.
  - Exact duplicates across sources: only 105 FineWeb documents (0.10 MB). Copies of the same article in FineWeb usually differ by a header line or spacing, so exact matching misses them; that is the near-duplicate gap noted above.
  - Documents under the word minimum after boilerplate removal: UFAL 3,361, FineWeb 3,173, Wikipedia 146, news 5.
  - No literal `<|endoftext|>` was found in any source.
  - Cleaned sizes for Step 6 (MB): fineweb2_urd 5,893.90; urdu_news_1m 1,958.01; ufal_umc 702.21; wiki_ur 513.78; own_scrape 76.01; makhzan 50.75.

### 2026-10-13: Step 6: byte budget, split and data card
- **Decision:**
  - **Sizes.** Budgets and shares count file bytes: the text plus the 13-byte `<|endoftext|>` written after each document. UFAL's 5.33M one-sentence documents carry 69 MB of separators, about 10% on top of their text.
  - **Budget (proposed, waiting for approval).** The brief's rules cannot all hold. The four sources it keeps whole (UFAL 702.21, Wikipedia 513.78, own_scrape 76.01 and Makhzan 50.75 MB of text) already hold 1,342.75 MB, or 1,416 MB with separators, above the 1.3 GB ceiling. Proposed: keep the 1.3 GB ceiling; keep Wikipedia, own_scrape and Makhzan whole; fill FineWeb-2 up to its 45% cap; subsample UFAL into the rest (about 70 MB, rate about 0.09). Urdu News 1M gets nothing: `punct_stripped` holds for all its documents, so its cap is 10%, and no space is left. Command: `python clean/budget.py --fill fineweb2_urd ufal_umc urdu_news_1m`. The full-run numbers go here after the run.
  - **Split.** Per source: sorted doc_ids, shuffled with `random.Random("1337:<source>")`, then train = n - 2k, val = k, test = k in that order, k = max(1, round(n/100)) rounded half up. A source with 0 kept documents is left out, and budget.py never leaves a source with 1 or 2. Each split is then shuffled across sources with `random.Random("1337:order:<split>")`.
  - **External shuffle.** Pass 1 holds only doc_ids. Pass 2 writes each document to one of 64 bucket files per split by position. Pass 3 sorts one bucket at a time. `<split>.ids` lists the doc_ids in file order, so the integrity tests can check that no document is in two splits.
- **Options considered** (projected from the Step 5b sizes, file MB):
  - **A, proposed:** 1.3 GB with UFAL subsampled. FineWeb 585 (45%), Wikipedia 518 (39.8%), own 76 (5.9%), Makhzan 51 (3.9%), UFAL 70 (5.4%). UFAL is the source that adds least per byte: shuffled sentences with no document context, from 2014, with the largest separator overhead.
  - **A2:** as A with `--cap fineweb2_urd=0.40`. FineWeb 520 (40%), UFAL 135 (10.4%).
  - **B:** keep all four whole and allow 2.6 GB (`--total-mb 2600`). UFAL becomes 29.7% of the corpus. Rejected: twice the size, dominated by sentence-level 2014 text.
  - **Keeping UFAL whole at about 1.6 GB.** Rejected: UFAL 48%, FineWeb 11%.
  - **Urdu News 1M at its 10% cap.** Not proposed: its text lacks sentence punctuation and lost letters (اسلام آباد stored as اسلام باد); `--mb urdu_news_1m=X` adds it if wanted.
- **Evidence:**
  - Dry run on the four real sources present on the second machine (1.0 GB budget, `--fill ufal_umc`; 8-core M-series Mac, 16 GB RAM):
    - budget scan 26 s, and its documents and MB match `dedup.json` exactly;
    - split 53 s, peak RSS 0.8 GB;
    - corpus_stats 11 s;
    - every check in `tests/test_split.py` passed, including the rerun with identical sha256 and `normalize(doc) == doc` on 2,000 random documents.
  - Smoke run of all six sources: 3,000 documents each.
  - The copied `char_audit.json` comes from the smoke-run samples (179 Wikipedia documents) and `char_audit_normalized.json` from the full-run samples (870), so they cannot be subtracted. `data_card.py` refuses audits of different samples; rerun both before generating the card.
  - Tests: `tests/test_budget.py`, `tests/test_split.py`, `tests/test_corpus_stats.py`, `tests/test_data_card.py`.
- **Consequence:**
  - Wikipedia is about 40% of any 1.3 GB budget that keeps it whole. That is the main genre bias in DATA_CARD.md.
  - The FineWeb-2 text covers CC-MAIN-2021-10 to 2024-18 only, not 2013-2024.
  - Revisit the budget if the tokenizer or the next assignment needs more data. B is one command away.

### 2026-10-13: Step 7: pre-tokenizer pattern and reference trainer
- **Decision:**
  - **Pre-tokenizer.** `bpe.URDU_PATTERN` as given in the Step 7 brief. A word is letters + marks + ZWNJ with an optional leading space; digits come in groups of up to 3; contractions match case-insensitively. Rationale and alternatives: [docs/pretokenizer.md](pretokenizer.md).
  - **Reference trainer.** `BPETokenizer.train_reference` follows the brief: every occurrence is its own byte list, all pairs are recounted after every merge, and ties go to the greater `(vocab[a], vocab[b])` compared as bytes.
  - **Ties on identical bytes.** The spec cannot decide one case. Two different merges can build the same string (`ab + c` and `a + bc`), and then two tied pairs can have identical bytes. Only in that case the greater pair of IDs wins, and a warning is logged. The fast trainer (Step 8) must use the same rule.
  - **encode** merges the present pair with the lowest merge ID until none is left, with a per-word cache. **decode** uses UTF-8 with surrogatepass and falls back to replace when an ID sequence cuts a character.
  - **Saved files.** `tokenizer/<name>.json` (format `urdu-bpe-v1`; `meta` also holds `merge_counts` when trained in the same process) and `tokenizer/<name>.merges.txt`. A bare `merges.txt` would be overwritten by the next tokenizer saved in `tokenizer/`.
  - **1 MB sample.** `data/samples/train_1mb.txt` (`clean/sample_1mb.py`) is the longest prefix of train.txt that ends just after an `<|endoftext|>` and holds at most 1,000,000 bytes, separators included. train.txt is already shuffled across sources.
  - **Handout sanity check.** The handout refers to a sanity check that it does not contain (2026-10-11 entry). `tests/test_bpe_reference.py::test_handout_sanity_check` is skipped with that reason, so the gap shows in every run. `test_tie_break_aaabdaaabac` (the Wikipedia BPE example, with a four-way tie at merge 4 that ID order gets wrong) stands in until the instructor answers.
- **Options considered:**
  - **Tie-breaking.** Ties by ID: rejected by the spec. Ties on identical bytes left to Counter order: rejected, because the fast trainer would visit pairs in a different order and could learn different merges.
  - **The pattern.** See docs/pretokenizer.md: GPT-2's pattern, o200k_base's, and splitting by script.
- **Evidence:**
  - Tests: `tests/test_bpe_reference.py` (26 tests) and `tests/test_forbidden_imports.py` (bpe.py: stdlib and regex only).
  - GPT-2's pattern cuts اُردُو into 5 pieces (`ا` `ُ` `رد` `ُ` `و`) and صورت‌حال into 3.
  - Timing on a stand-in 1 MB (993,276 bytes from the four cleaned sources on the second machine; not the official sample): 3,000 merges in 189 s, 63 ms per merge, 62 MB peak RSS (Apple M1 Pro, Python 3.12.14).
  - The first merges rebuild two-byte letters (ا, ی, ک, ر). Merges 3 and 4 are a space plus a lead byte (`␣\xd8`, `␣\xd9`), tokens that are not valid UTF-8 on their own.
  - The official run on train_1mb.txt goes here when it is done.
- **Consequence:** Part C's question on merges that are not valid UTF-8 has early examples. Step 8 compares its merges, and their counts (`meta.merge_counts`), with `tokenizer/reference_1mb_3000.json`.
