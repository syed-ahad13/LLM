"""Step 6d: generate DATA_CARD.md (one page) and docs/cleaning_detail.md from the pipeline's own numbers.

    python clean/data_card.py

Reads:
- collect/sources.json: URL, licence (licence_short), collection date, raw size;
- data/stats/{extract,normalize,langfilter,boilerplate,dedup}.json: what each
  cleaning step removed, per source;
- data/stats/char_audit.json and char_audit_normalized.json: the Step 4 audit
  of the 2 MB samples, before and after normalize(). Both must cover the same
  documents (rerun clean/audit_chars.py with and without --normalized if not);
  a rule's count is the occurrences that normalization changed, per code point;
- data/final/manifest.json and the budget.json it names: what is in the corpus;
- data/stats/corpus_totals.json (clean/corpus_stats.py): MB, words, unique words;
- docs/data_card_gaps.md: the hand-edited "Known gaps and biases" bullets,
  with {name} placeholders filled from the numbers above.

Nothing in the card is typed by hand except the gaps bullets. Writes
DATA_CARD.md and the per-source tables to docs/cleaning_detail.md.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.budget import FINAL_DIR  # noqa: E402
from clean.dedup import ENDOFTEXT, PRIORITY  # noqa: E402
from clean.extract_common import SOURCES_JSON  # noqa: E402
from common import STATS_DIR, setup_logging  # noqa: E402

DATA_CARD = ROOT / "DATA_CARD.md"
DETAIL = ROOT / "docs" / "cleaning_detail.md"
GAPS = ROOT / "docs" / "data_card_gaps.md"
SPLITS = ("train", "val", "test")
STAGES = ("extract", "normalize", "langfilter", "boilerplate", "dedup")
NAMES = {"wiki_ur": "Urdu Wikipedia", "ufal_umc": "UFAL Urdu Monolingual Corpus", "urdu_news_1m": "Urdu News 1M",
         "fineweb2_urd": "FineWeb-2 (urd_Arab)", "makhzan": "Makhzan", "own_scrape": "Own crawl: VOA Urdu + Hum Sub",
         "cc100_ur": "CC-100 Urdu"}
SHORT = {"wiki_ur": "Wikipedia", "ufal_umc": "UFAL", "urdu_news_1m": "Urdu News 1M", "fineweb2_urd": "FineWeb-2",
         "makhzan": "Makhzan", "own_scrape": "our crawl", "cc100_ur": "CC-100"}
# Normalization rules (normalize.py, docs/decisions.md Step 4) and the audit categories each one acts on.
RULES = [
    ("Presentation forms → letters (ﷲ → اللہ; ﷺ ﷻ kept)", ["presentation_form"]),
    ("Arabic yeh ي ى → Urdu yeh ی", ["arabic_yeh"]),
    ("Arabic kaf ك → ک (Sindhi ڪ kept)", ["arabic_kaf"]),
    ("Arabic heh ه ە → ہ; ة → ۃ; ۀ → ۂ", ["arabic_heh", "heh_like_other"]),
    ("Letter + combining madda/hamza → one code point (NFC)", ["combining_madda_hamza"]),
    ("Arabic-Indic and Urdu digits → ASCII", ["digits_arabic_indic", "digits_extended"]),
    ("Tatweel removed", ["tatweel"]),
    ("ZWJ, bidi marks, BOM, soft hyphen, controls removed; ZWNJ kept only inside words",
     ["zwj", "invisible_other", "control", "zwnj"]),
    ("Whitespace: no-break spaces and tabs → space, runs collapsed, lines trimmed, no blank lines",
     ["space_unusual", "tab", "carriage_return", "line_separator_other", "space_run", "space_at_line_edge",
      "blank_line_run"]),
    ("No space before ۔ ، ؟ ؛", ["space_before_urdu_punct"]),
]
PLACEHOLDER = regex.compile(r"\{([a-z0-9_]+)\}")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def mb(n: float) -> str:
    return f"{n / 1e6:,.2f}"


def pct(x: float) -> str:
    return f"{x:.1%}" if x >= 0.001 or x == 0 else f"{x:.2%}"


def num(n: int) -> str:
    return f"{n:,}"


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]], right: Sequence[int] = ()) -> str:
    align = ["---:" if i in right else "---" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def order(sources: Sequence[str]) -> list[str]:
    return sorted(sources, key=lambda s: (PRIORITY.index(s) if s in PRIORITY else len(PRIORITY), s))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load(args.manifest)
    budget_path = Path(manifest["budget"]["path"])
    if not budget_path.exists():
        budget_path = args.manifest.parent / "budget.json"
    inputs = {
        "sources": {s["id"]: s for s in load(args.sources)["sources"]},
        "stats": {stage: load(args.stats_dir / f"{stage}.json") for stage in STAGES},
        "audit": load(args.stats_dir / "char_audit.json"),
        "audit_norm": load(args.stats_dir / "char_audit_normalized.json"),
        "totals": load(args.stats_dir / "corpus_totals.json"),
        "manifest": manifest,
        "budget": load(budget_path),
        "gaps": args.gaps.read_text(encoding="utf-8") if args.gaps.exists() else "",
    }
    raw, norm = inputs["audit"]["sources"], inputs["audit_norm"]["sources"]
    differ = [s for s in raw if s not in norm or raw[s]["docs"] != norm[s]["docs"]]
    if differ:
        raise SystemExit(f"the two Step 4 audits cover different samples for {', '.join(differ)}; rerun "
                         "python clean/audit_chars.py and python clean/audit_chars.py --normalized")
    shas = inputs["totals"].get("manifest_sha256")
    if shas != {s: manifest["splits"][s]["sha256"] for s in SPLITS}:
        raise SystemExit("data/stats/corpus_totals.json does not describe the current split files; "
                         "rerun python clean/corpus_stats.py")
    return inputs


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def step_rows(inp: dict[str, Any]) -> list[dict[str, Any]]:
    """Totals across sources for each cleaning step: bytes and documents in and out."""
    st, manifest = inp["stats"], inp["manifest"]
    sources = list(manifest["sources"])

    def tot(stage: str, key: str) -> int:
        return sum(st[stage].get(s, {}).get(key, 0) for s in sources)

    lf_sindhi_own = st["langfilter"].get("own_scrape", {}).get("drop_docs_sindhi_doc", 0)
    rates = ", ".join(f"{SHORT.get(s, s)} {pct(r['rate'])}"
                      for s, r in manifest["budget"]["sources"].items() if r["rule"] != "full")
    rows = [
        {"step": "1. Extract plain text", "what": "Text out of dumps, CSV, parquet and HTML; Wikipedia: articles "
         "only, no redirects, disambiguation or bodies under 200 characters",
         "bytes_in": tot("extract", "raw_text_bytes"), "bytes_out": tot("extract", "bytes_written"),
         "docs_in": tot("extract", "records_read"), "docs_out": tot("extract", "docs_written")},
        {"step": "2. Normalize", "what": "normalize.py, rules below", "bytes_in": tot("normalize", "bytes_in"), "bytes_out": tot("normalize", "bytes_out"),
         "docs_in": tot("normalize", "docs_in"), "docs_out": tot("normalize", "docs_out")},
        {"step": "3. Line language filter", "what": "Drop lines with no letters, mostly not Arabic script, Sindhi, "
         f"or Arabic/Persian; drop Sindhi documents ({lf_sindhi_own} from our VOA crawl) and documents under 10 "
         "words (3 for UFAL)", "bytes_in": tot("langfilter", "bytes_in"), "bytes_out": tot("langfilter", "bytes_out"),
         "docs_in": tot("langfilter", "docs_in"), "docs_out": tot("langfilter", "docs_out")},
        {"step": "4. Boilerplate lines", "what": "Drop lines found in 5 or more documents of a source (3 for our "
         "crawl)",
         "bytes_in": tot("boilerplate", "bytes_in"), "bytes_out": tot("boilerplate", "bytes_out"),
         "docs_in": tot("boilerplate", "docs_in"), "docs_out": tot("boilerplate", "docs_out")},
        {"step": "5. Exact document dedup", "what": "Drop exact duplicate documents, within and across sources", "bytes_in": tot("dedup", "bytes_in"),
         "bytes_out": tot("dedup", "bytes_out"), "docs_in": tot("dedup", "docs_in"),
         "docs_out": tot("dedup", "docs_out")},
        {"step": "6. Budget subsample", "what": f"Keep documents by blake2b(doc_id) to fit the budget: {rates} "
         "kept, the rest whole", "bytes_in": tot("dedup", "bytes_out"),
         "bytes_out": manifest["totals"]["text_bytes"], "docs_in": tot("dedup", "docs_out"),
         "docs_out": manifest["totals"]["docs"]},
    ]
    return rows


def normalization_rows(inp: dict[str, Any]) -> tuple[list[tuple[str, int, float]], int]:
    """(rule, occurrences changed, per million characters) over the audit samples, and the characters audited."""
    raw, norm = inp["audit"]["sources"], inp["audit_norm"]["sources"]
    chars = sum(r["chars"] for r in raw.values())
    out = []
    for rule, cats in RULES:
        changed = 0
        for s in raw:
            for cat in cats:
                before = raw[s]["categories"].get(cat, {}).get("matches", {})
                after = norm[s]["categories"].get(cat, {}).get("matches", {})
                changed += sum(max(0, n - after.get(k, 0)) for k, n in before.items())
        out.append((rule, changed, changed / chars * 1e6 if chars else 0.0))
    return out, chars


def genre_bytes(manifest: dict[str, Any]) -> Counter[str]:
    g: Counter[str] = Counter()
    for s in SPLITS:
        g.update(manifest["splits"][s]["genres"])
    return g


def facts(inp: dict[str, Any]) -> dict[str, str]:
    """Values for the {name} placeholders in docs/data_card_gaps.md."""
    m, st, totals = inp["manifest"], inp["stats"], inp["totals"]
    f: dict[str, str] = {}
    total = m["totals"]["bytes"]
    for s, row in m["sources"].items():
        f[f"share_{s}"] = pct(row["bytes"] / total)
        f[f"rate_{s}"] = f"{m['budget']['sources'][s]['rate']:.4f}"
        f[f"mb_{s}"] = mb(row["text_bytes"])
    g = genre_bytes(m)
    text_total = sum(g.values())
    ranked = g.most_common()
    if ranked:
        f["top_genre"] = ranked[0][0][:1].upper() + ranked[0][0][1:]
        f["top_genre_share"] = pct(ranked[0][1] / text_total)
        f["genre_shares"] = ", ".join(f"{name} {pct(n / text_total)}" for name, n in ranked)
    fw = m["sources"].get("fineweb2_urd")
    if fw and fw["docs"]:
        for tld in ("pk", "in", "com"):
            f[f"fineweb_{tld}"] = pct(fw["url_domains"].get(tld, 0) / fw["docs"])
    lf = st["langfilter"]
    used = list(m["sources"])
    f["arabic_persian_mb"] = mb(sum(lf.get(s, {}).get("drop_bytes_arabic_or_persian", 0) for s in used))
    f["arabic_persian_lines"] = num(sum(lf.get(s, {}).get("drop_lines_arabic_or_persian", 0) for s in used))
    f["sindhi_mb"] = mb(sum(lf.get(s, {}).get("drop_bytes_sindhi", 0) for s in used))
    f["sindhi_docs"] = num(sum(lf.get(s, {}).get("drop_docs_sindhi_doc", 0) for s in used))
    audit = inp["audit"]["sources"]
    chars = sum(r["chars"] for r in audit.values())
    heh = sum(r["categories"]["arabic_heh"]["count"] for r in audit.values())
    f["heh_per_million"] = f"{heh / chars * 1e6:,.0f}" if chars else "0"
    markers = totals["total"].get("markers", {})
    f["shahmukhi_docs"] = num(markers.get("docs_with_U+0768", 0))
    f["shahmukhi_chars"] = num(markers.get("char_U+0768", 0))
    sites = {site["site_id"]: site for site in inp["sources"].get("own_scrape", {}).get("sites", [])}
    for site_id, key in (("voa_urdu", "voa_range"), ("humsub", "humsub_range")):
        if site_id in sites and sites[site_id].get("published_range"):
            a, b = sites[site_id]["published_range"]
            f[key] = f"{a} to {b}"
    f["final_mb"] = mb(total)
    f["final_docs"] = num(m["totals"]["docs"])
    return f


def fill(template: str, values: dict[str, str], log: Any) -> str:
    missing = sorted({k for k in PLACEHOLDER.findall(template) if k not in values})
    if missing:
        log.warning("no value for %s in %s; left as is", ", ".join(missing), GAPS.name)
    return PLACEHOLDER.sub(lambda mt: values.get(mt.group(1), mt.group(0)), template)


def gap_bullets(template: str) -> str:
    """The bullets of docs/data_card_gaps.md, without its leading comment."""
    body = regex.sub(r"\A\s*<!--(?:(?!-->).)*-->\s*", "", template, flags=regex.S)
    return body.strip()


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def data_card(inp: dict[str, Any], log: Any) -> str:
    m, st, totals, src = inp["manifest"], inp["stats"], inp["totals"], inp["sources"]
    used = order(list(m["sources"]))
    total = m["totals"]["bytes"]
    genres = {s: inp["budget"]["sources"][s]["genres"] for s in used}
    n_genres = len(genre_bytes(m))
    t = totals["total"]
    own = m["sources"].get("own_scrape", {}).get("text_bytes", 0)
    big = sorted(used, key=lambda s: -m["sources"][s]["bytes"])[:2]
    out = [
        "# Data card: Urdu corpus",
        "",
        "<!-- Generated by clean/data_card.py from data/stats/*.json, collect/sources.json and "
        "data/final/manifest.json. Do not edit by hand: the gaps section comes from docs/data_card_gaps.md. -->",
        "",
        f"**Summary.** {mb(total)} MB of cleaned Urdu text ({num(t['docs'])} documents, {num(t['words'])} words) "
        f"from {sum(1 for s in used if m['sources'][s]['docs'])} sources in {n_genres} genres, split by document "
        f"into train, val and test (98/1/1, seed {m['seed']}, stratified by source) with {ENDOFTEXT} after each "
        f"document. The largest parts are {NAMES.get(big[0], big[0])} ({pct(m['sources'][big[0]]['bytes'] / total)}) "
        f"and {NAMES.get(big[1], big[1])} ({pct(m['sources'][big[1]]['bytes'] / total)}), and {mb(own)} MB is our "
        "own crawl of VOA Urdu and Hum Sub.",
        "",
        "MB = 10^6 bytes, MiB = 2^20 bytes. Words are `str.split()` tokens; unique words are distinct runs of "
        "`[\\p{L}\\p{M}\\u200c]`.",
        "",
        "## Sources",
        "",
    ]
    rows = []
    for s in used:
        e = src.get(s, {})
        urls = " ".join(f"<{u.strip()}>" for u in (e.get("landing_url") or "").split(";") if u.strip())
        rows.append((NAMES.get(s, s), ", ".join(genres[s]), urls, e.get("licence_short", e.get("licence", "")),
                     e.get("collection_date") or "", mb(e.get("raw_bytes") or 0),
                     mb(st["dedup"].get(s, {}).get("bytes_out", 0)), pct(m["sources"][s]["bytes"] / total)))
    out.append(md_table(("Source", "Genre", "URL", "Licence / terms", "Collected", "Raw MB", "Cleaned MB",
                         "Share of corpus"), rows, right=(5, 6, 7)))
    out += ["",
            "Raw MB: files as downloaded (compressed except Urdu News; FineWeb-2: one of its two files; our crawl: "
            "JSON lines). Cleaned MB: text after Step 5b. Share: bytes in train+val+test. Makhzan may not be "
            "republished, so the corpus link stays private to course staff.",
            "",
            "## Corpus totals",
            ""]
    rows = []
    for name, r in [*((s, totals["splits"][s]) for s in SPLITS), ("**total**", t)]:
        rows.append((name, f"{r['mb']:,.2f}", f"{r['mib']:,.2f}", num(r["docs"]), num(r["words"]),
                     num(r["unique_words"])))
    out.append(md_table(("Split", "MB", "MiB", "Documents", "Words", "Unique words"), rows, right=(1, 2, 3, 4, 5)))
    out += ["", f"MB includes the 13-byte {ENDOFTEXT} after each document ({mb(t['separator_bytes'])} MB in all).",
            "", "## Cleaning steps", ""]
    rows = []
    first = None
    for r in step_rows(inp):
        removed = r["bytes_in"] - r["bytes_out"]
        first = first if first is not None else r["bytes_in"]
        rows.append((r["step"], r["what"], mb(removed), pct(removed / r["bytes_in"]) if r["bytes_in"] else "-",
                     num(r["docs_in"] - r["docs_out"])))
        last = r
    removed = first - last["bytes_out"]
    rows.append(("**Total**", "From extracted text to the corpus", mb(removed), pct(removed / first),
                 num(step_rows(inp)[0]["docs_in"] - last["docs_out"])))
    out.append(md_table(("Step", "What it does", "MB removed", "% of step input", "Docs removed"), rows,
                        right=(2, 3, 4)))
    norm_in = sum(st["normalize"].get(s, {}).get("bytes_in", 0) for s in used)
    norm_out = sum(st["normalize"].get(s, {}).get("bytes_out", 0) for s in used)
    out += ["", "Step 1 counts the UTF-8 text read from each container, so Wikipedia's removed bytes are mostly "
            "markup and non-article pages. Per-source numbers: [docs/cleaning_detail.md](docs/cleaning_detail.md).",
            "", "## Normalization", ""]
    nrows, chars = normalization_rows(inp)
    out.append(md_table(("Rule (normalize.py, in order of application)", "Changed", "Per million chars"),
                        [(rule, num(n), f"{r:,.0f}") for rule, n, r in nrows], right=(1, 2)))
    out += ["", f"Changed: occurrences that `normalize()` altered in the Step 4 audit samples (about 2 MB per "
            f"source, {num(chars)} characters). On the whole corpus, normalization shrank the text from "
            f"{mb(norm_in)} to {mb(norm_out)} MB ({pct((norm_in - norm_out) / norm_in) if norm_in else '-'}).",
            "", "## Known gaps and biases", ""]
    out.append(fill(gap_bullets(inp["gaps"]), facts(inp), log) if inp["gaps"] else "_No docs/data_card_gaps.md._")
    out += ["", "## Rebuild", "", "From the downloads in `data/raw/` (Steps 1-2 in README.md):", "", "```bash",
            "python clean/extract_all.py --wiki-jobs 7          # Step 3: extract",
            "python clean/audit_chars.py && python clean/audit_chars.py --normalized   # Step 4: audit samples",
            "python clean/normalize_filter.py --jobs 7          # Step 5: normalize + line language filter",
            "python clean/dedup.py                              # Step 5b: boilerplate lines + exact dedup",
            f"{inp['budget'].get('command') or 'python clean/budget.py'}   # Step 6: budget",
            "python clean/split.py && python clean/corpus_stats.py",
            "python -m pytest -q -s tests/test_split.py && python clean/data_card.py",
            "```", ""]
    return "\n".join(out)


def cleaning_detail(inp: dict[str, Any]) -> str:
    m, st = inp["manifest"], inp["stats"]
    used = order(list(m["sources"]))
    total = m["totals"]["bytes"]
    out = ["# Cleaning detail per source", "",
           "Generated by clean/data_card.py from data/stats/*.json and data/final/manifest.json. MB = 10^6 bytes "
           "of UTF-8 text. The totals are in [DATA_CARD.md](../DATA_CARD.md).", ""]

    def removed(a: int, b: int) -> str:
        return pct((a - b) / a) if a else "-"

    ex = st["extract"]
    out += ["## Step 3: extract", ""]
    rows = []
    for s in used:
        r = ex.get(s, {})
        drops = ", ".join(f"{k.removeprefix('drop_')} {num(v)}" for k, v in sorted(r.items())
                          if k.startswith("drop_") and v)
        rows.append((s, num(r.get("records_read", 0)), num(r.get("docs_written", 0)), mb(r.get("raw_text_bytes", 0)),
                     mb(r.get("bytes_written", 0)), drops or "-"))
    out += [md_table(("Source", "Records read", "Docs written", "Text MB read", "MB written", "Dropped records"),
                     rows, right=(1, 2, 3, 4)), ""]

    for stage, title in (("normalize", "Step 5: normalize"), ("langfilter", "Step 5: line language filter"),
                         ("boilerplate", "Step 5b: boilerplate lines"), ("dedup", "Step 5b: exact dedup")):
        rows = []
        for s in used:
            r = st[stage].get(s, {})
            detail = []
            if stage == "langfilter":
                detail = [f"{k.removeprefix('drop_bytes_')} {mb(v)} MB" for k, v in sorted(r.items())
                          if k.startswith("drop_bytes_") and v]
                detail += [f"{k.removeprefix('drop_docs_')} {num(v)} docs" for k, v in sorted(r.items())
                           if k.startswith("drop_docs_") and v]
            elif stage == "boilerplate":
                detail = [f"cutoff {r.get('min_docs', '-')} docs", f"{num(r.get('drop_lines_boilerplate', 0))} lines "
                          f"({mb(r.get('drop_bytes_boilerplate', 0))} MB)",
                          f"{num(r.get('drop_docs_min_words', 0))} docs under the word minimum"]
            elif stage == "dedup":
                detail = [f"in source {num(r.get('drop_docs_duplicate_in_source', 0))} docs "
                          f"({mb(r.get('drop_bytes_duplicate_in_source', 0))} MB)",
                          f"other source {num(r.get('drop_docs_duplicate_other_source', 0))} docs "
                          f"({mb(r.get('drop_bytes_duplicate_other_source', 0))} MB)"]
            rows.append((s, num(r.get("docs_in", 0)), num(r.get("docs_out", 0)), mb(r.get("bytes_in", 0)),
                         mb(r.get("bytes_out", 0)), removed(r.get("bytes_in", 0), r.get("bytes_out", 0)),
                         "; ".join(detail) or "-"))
        out += [f"## {title}", "", md_table(("Source", "Docs in", "Docs out", "MB in", "MB out", "Removed", "Detail"),
                                            rows, right=(1, 2, 3, 4, 5)), ""]

    out += ["## Step 6: budget and split", "", f"Budget command: `{inp['budget'].get('command') or '-'}`", ""]
    rows = []
    for s in used:
        r, b = m["sources"][s], m["budget"]["sources"][s]
        rows.append((s, b["rule"], f"{b['rate']:.4f}", num(r["available_docs"]), num(r["docs"]), mb(r["text_bytes"]),
                     num(r["train"]), num(r["val"]), num(r["test"]), pct(r["bytes"] / total)))
    out += [md_table(("Source", "Rule", "Rate", "Docs available", "Docs kept", "Text MB", "Train", "Val", "Test",
                      "Share"), rows, right=(2, 3, 4, 5, 6, 7, 8, 9)), ""]
    out += ["Most common URL domains of the documents in the corpus (last label of the host; - = no URL):", ""]
    rows = [(s, ", ".join(f".{d} {num(n)}" if d != "-" else f"- {num(n)}"
                          for d, n in list(m["sources"][s]["url_domains"].items())[:8]) or "-") for s in used]
    out += [md_table(("Source", "Domains"), rows), ""]
    return "\n".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    p.add_argument("--sources", type=Path, default=SOURCES_JSON)
    p.add_argument("--manifest", type=Path, default=FINAL_DIR / "manifest.json")
    p.add_argument("--gaps", type=Path, default=GAPS)
    p.add_argument("--out", type=Path, default=DATA_CARD)
    p.add_argument("--detail-out", type=Path, default=DETAIL)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("data_card")
    inp = load_inputs(args)
    card = data_card(inp, log)
    args.out.write_text(card, encoding="utf-8")
    args.detail_out.parent.mkdir(parents=True, exist_ok=True)
    args.detail_out.write_text(cleaning_detail(inp), encoding="utf-8")
    review = card.count("<!-- REVIEW -->")
    print(f"wrote {args.out} ({len(card.splitlines())} lines; {review} gaps bullets marked REVIEW) "
          f"and {args.detail_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
