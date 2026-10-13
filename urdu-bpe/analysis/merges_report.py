"""Part C, question 1: what did the merges learn?

    python analysis/merges_report.py              # -> report/tables/merges*.tex, report/figures/merge_categories.pdf
    python analysis/merges_report.py --no-trace   # skip tracing the longest tokens through train.txt

Reads tokenizer/urdu_bpe_50k.json and the training word counts that Step 9 cached in data/cache/ (found by the
tokenizer's meta.input_sha256, the sha256 of train.txt).

Each merged token gets one category, tested in this order:
- LETTER: exactly one valid 2-byte UTF-8 character (lead byte 0xC2-0xDF plus a continuation byte), e.g.
  D8 A7 = ا;
- SPACE+LEAD: 0x20 followed by an incomplete character (a lead byte and fewer continuation bytes than it
  needs), e.g. ␣\\xd8;
- INVALID_START / INVALID_END: any other token that strict UTF-8 decoding rejects; it starts with a
  continuation byte, or else it ends in the middle of a character. Every token is a substring of valid UTF-8
  text, so these are the only two ways to be invalid;
- WORD: valid UTF-8 matching ^ ?[\\p{L}\\p{M}\\u200c]+$ with at least two letters (\\p{L}), whose string with
  or without its leading space is a whole pre-token of train.txt at least --min-count (100) times;
- SPACE+LETTER: a space and one letter (marks allowed), e.g. ␣ک;
- OTHER: any other valid text (digits, punctuation, whitespace, rare fragments).

The brief's WORD has no two-letter minimum. Without it, ␣ک (merge 10) would be the first "word": single
letters occur standalone in abbreviations, so they pass the frequency test. The one-letter word و ("and")
is counted as SPACE+LETTER for the same reason; analysis.json keeps the brief's literal first word too.

Writes report/tables/merges.tex (merges 1-50, every 5,000th, the last), report/tables/merges_longest.tex (the
20 longest tokens, each traced to the sources of the train.txt documents that contain it),
report/figures/merge_categories.pdf (category shares in log-spaced rank buckets), the section "q1_merges"
of data/stats/analysis.json and the Q1 draft in docs/analysis_notes.md.
"""

import argparse
import pickle
import sys
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report_utils import (ANALYSIS_JSON, CACHE_DIR, NOTES, PALETTE, TABLES_DIR, TOKENIZER,  # noqa: E402
                                   TRAIN_TXT, latex_display, latex_escape, pct, pyplot, save_figure, show,
                                   update_analysis, update_notes, write_table)
from bpe import COUNT_BLOCK_BYTES, ENDOFTEXT, BPETokenizer, _iter_chunks, word_counts_cache_path  # noqa: E402
from common import setup_logging  # noqa: E402

SCRIPT = "analysis/merges_report.py"
CATEGORIES = ("LETTER", "SPACE+LEAD", "INVALID_START", "INVALID_END", "SPACE+LETTER", "WORD", "OTHER")
CATEGORY_LABELS = {"LETTER": "Letter (2 bytes)", "SPACE+LEAD": "Space + lead byte",
                   "INVALID_START": "Invalid: starts mid-character", "INVALID_END": "Invalid: ends mid-character",
                   "SPACE+LETTER": "Space + one letter", "WORD": "Word", "OTHER": "Other"}
WORD_RE = regex.compile(r" ?[\p{L}\p{M}\u200c]+")
LETTER_RE = regex.compile(r"\p{L}")
EARLY = 50
EVERY = 5000
LONGEST = 20
# Traced with the longest tokens: an early merge (rank 363) whose count looked out of place for "station".
TRACE_EXTRA = (" اسٹیشن",)
# Sentences of templated Wikipedia stubs, found by reading the documents that contain the longest tokens
# (asteroids, US census-designated places, railway stations); counted per source as evidence.
TEMPLATES = {"asteroid": "نظام شمسی کا ایک سیارچہ ہے",
             "US census-designated place": "مردم شماری نامزد مقام",
             "railway station": "میں واقع ایک اسٹیشن ہے"}
MIN_COUNT = 100


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

def char_length(lead: int) -> int:
    """Bytes in a UTF-8 character with this first byte; 0 if it cannot start one."""
    if lead < 0x80:
        return 1
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return 0


def is_incomplete_char(data: bytes) -> bool:
    """A lead byte followed by fewer continuation bytes than it needs."""
    n = char_length(data[0]) if data else 0
    return n > 1 and len(data) < n and all(0x80 <= c <= 0xBF for c in data[1:])


def category(token: bytes, counts: Mapping[str, int], min_count: int = MIN_COUNT) -> str:
    """The category of a merged token (module docstring)."""
    try:
        text = token.decode("utf-8")
    except UnicodeDecodeError:
        if len(token) > 1 and token[0] == 0x20 and is_incomplete_char(token[1:]):
            return "SPACE+LEAD"
        return "INVALID_START" if 0x80 <= token[0] <= 0xBF else "INVALID_END"
    if len(token) == 2 and 0xC2 <= token[0] <= 0xDF:
        return "LETTER"
    if WORD_RE.fullmatch(text):
        core = text.lstrip(" ")
        letters = len(LETTER_RE.findall(core))
        if letters >= 2 and max(counts.get(core, 0), counts.get(" " + core, 0)) >= min_count:
            return "WORD"
        if letters == 1 and text.startswith(" "):
            return "SPACE+LETTER"
    return "OTHER"


def literal_word(token: bytes, counts: Mapping[str, int], min_count: int = MIN_COUNT) -> bool:
    """The brief's WORD test, without the two-letter minimum (reported for comparison); LETTER comes first."""
    try:
        text = token.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if len(token) == 2:
        return False
    core = text.lstrip(" ")
    return bool(WORD_RE.fullmatch(text)) and max(counts.get(core, 0), counts.get(" " + core, 0)) >= min_count


def buckets(n: int) -> list[tuple[int, int]]:
    """Log-spaced rank buckets: 1-10, 11-100, ..., the last one ending at n."""
    out, lo, hi = [], 1, 10
    while lo <= n:
        out.append((lo, min(hi, n)))
        lo, hi = hi + 1, hi * 10
    return out


def load_counts(tok: BPETokenizer, cache_dir: Path) -> dict[str, int]:
    sha = tok.meta.get("input_sha256")
    if not sha:
        raise SystemExit("the tokenizer's meta has no input_sha256 (train it with train_tokenizer.py)")
    path = word_counts_cache_path(cache_dir, sha)
    if not path.exists():
        raise SystemExit(f"no word counts at {path}: rerun train_tokenizer.py, which caches them")
    with open(path, "rb") as f:
        return pickle.load(f)["counts"]


# ---------------------------------------------------------------------------
# Tracing tokens to sources
# ---------------------------------------------------------------------------

def documents_with_sources(train: Path, ids: Path) -> Iterator[tuple[str, bytes, bytes]]:
    """(source, document, the block it is in) for every document of train.txt, with train.ids in step."""
    eot = ENDOFTEXT.encode("utf-8")
    with open(ids, encoding="utf-8") as id_lines:
        for block in _iter_chunks(train, COUNT_BLOCK_BYTES, [eot]):
            docs = block.split(eot)
            if docs and docs[-1] == b"":
                docs.pop()
            for doc in docs:
                yield next(id_lines).split(":", 1)[0], doc, block


def trace(needles: Sequence[str], train: Path, ids: Path) -> dict[str, dict[str, int]]:
    """{needle: {source: documents of train.txt containing it}}; a block is searched before its documents."""
    found: dict[str, Counter[str]] = {n: Counter() for n in needles}
    encoded = {n: n.encode("utf-8") for n in needles}
    last_block, present = None, []
    for source, doc, block in documents_with_sources(train, ids):
        if block is not last_block:
            last_block, present = block, [n for n in needles if encoded[n] in block]
        for n in present:
            if encoded[n] in doc:
                found[n][source] += 1
    return {n: dict(c.most_common()) for n, c in found.items()}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def token_row(tok: BPETokenizer, rank: int, counts: Mapping[str, int]) -> dict[str, Any]:
    a, b = tok.merges[rank - 1]
    new_id = tok.first_merge_id + rank - 1
    token = tok.vocab[new_id]
    row = {"rank": rank, "id": new_id, "left": show(tok.vocab[a]), "right": show(tok.vocab[b]),
           "token": show(token), "bytes": len(token), "category": category(token, counts)}
    if rank <= len(tok.merge_counts):
        row["count"] = tok.merge_counts[rank - 1]
    return row


def analyse(tok: BPETokenizer, counts: Mapping[str, int], sources: dict[str, dict[str, int]] | None,
            min_count: int = MIN_COUNT, wiki_docs: int | None = None) -> dict[str, Any]:
    n = len(tok.merges)
    first = tok.first_merge_id
    cats = [category(tok.vocab[first + r - 1], counts, min_count) for r in range(1, n + 1)]
    table_ranks = [*range(1, min(EARLY, n) + 1), *range(EVERY, n, EVERY)] + ([n] if n > EARLY else [])
    table = [token_row(tok, r, counts) for r in sorted(set(table_ranks))]
    early = [row for row in table if row["rank"] <= EARLY]

    def first_rank(test: Any) -> dict[str, Any] | None:
        for r in range(1, n + 1):
            if test(r):
                token = tok.vocab[first + r - 1]
                core = token.decode("utf-8").lstrip(" ")
                return {"rank": r, "token": show(token), "bytes": len(token),
                        "pretoken_count": max(counts.get(core, 0), counts.get(" " + core, 0))}
        return None

    shares = []
    for lo, hi in buckets(n):
        c = Counter(cats[lo - 1:hi])
        total = hi - lo + 1
        shares.append({"ranks": [lo, hi], "merges": total, "counts": {k: c.get(k, 0) for k in CATEGORIES},
                       "shares": {k: round(c.get(k, 0) / total, 4) for k in CATEGORIES}})

    longest_ids = sorted(range(first, first + n), key=lambda i: (-len(tok.vocab[i]), i))[:LONGEST]
    longest = []
    for i in longest_ids:
        token = tok.vocab[i]
        text = token.decode("utf-8", "replace")
        entry = {"rank": i - first + 1, "id": i, "token": show(token), "chars": len(text), "bytes": len(token),
                 "category": cats[i - first]}
        if sources is not None:
            docs = sources.get(text.lstrip(" "), {})
            total = sum(docs.values())
            top = max(docs, key=docs.get) if docs else None
            entry.update(docs=total, sources=docs, top_source=top,
                         top_share=round(docs[top] / total, 4) if top else None)
        longest.append(entry)
    junk = [e for e in longest if e["category"] != "WORD"]
    extra = []
    if sources is not None:
        for needle in TRACE_EXTRA:
            ids = [i for i in range(first, first + n) if tok.vocab[i] == needle.encode("utf-8")]
            docs = sources.get(needle.lstrip(" "), {})
            total = sum(docs.values())
            top = max(docs, key=docs.get) if docs else None
            extra.append({"token": show(needle.encode("utf-8")), "rank": ids[0] - first + 1 if ids else None,
                          "count": tok.merge_counts[ids[0] - first] if ids and tok.merge_counts else None,
                          "docs": total, "sources": docs, "top_source": top,
                          "top_share": round(docs[top] / total, 4) if top else None})

    return {
        "tokenizer": str(TOKENIZER.relative_to(ROOT)), "merges": n, "min_count": min_count,
        "category_totals": dict(Counter(cats)),
        "early": {"letters": [{"rank": r["rank"], "token": r["token"]} for r in early if r["category"] == "LETTER"],
                  "not_valid_utf8": [{"rank": r["rank"], "token": r["token"], "category": r["category"]}
                                     for r in early if r["category"] in ("SPACE+LEAD", "INVALID_START",
                                                                         "INVALID_END")],
                  "categories": dict(Counter(r["category"] for r in early))},
        # A whole word as it occurs in running text: WORD with its leading space. The first WORD of any
        # kind can be a word ending that also stands alone (یں).
        "first_word": first_rank(lambda r: cats[r - 1] == "WORD" and tok.vocab[first + r - 1][:1] == b" "),
        "first_word_any": first_rank(lambda r: cats[r - 1] == "WORD"),
        "first_word_brief_literal": first_rank(lambda r: literal_word(tok.vocab[first + r - 1], counts,
                                                                      min_count)),
        "first_space_letter": first_rank(lambda r: cats[r - 1] == "SPACE+LETTER"),
        "buckets": shares,
        "table": table,
        "longest": longest,
        "longest_junk": [e["token"] for e in junk],
        "traced_extra": extra,
        "templates": ({name: {"phrase": phrase, "docs": sources.get(phrase, {})} for name, phrase in TEMPLATES.items()}
                      if sources is not None else {}),
        "wiki_train_docs": wiki_docs,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_tables(res: dict[str, Any], tables_dir: Path = TABLES_DIR) -> list[Path]:
    cell = latex_display
    rows = [[str(r["rank"]), str(r["id"]), cell(r["left"]), cell(r["right"]), cell(r["token"]), str(r["bytes"]),
             latex_escape(r["category"])] for r in res["table"]]
    paths = [write_table("merges", "rrlllrl", ["Rank", "ID", "Left", "Right", "Merged", "Bytes", "Category"], rows,
                         SCRIPT, tables_dir)]
    traced = "docs" in (res["longest"][0] if res["longest"] else {})
    header = ["Rank", "Token", "Chars", "Bytes"] + (["Docs", "Main source"] if traced else [])
    rows = []
    for e in res["longest"]:
        row = [f"{e['rank']:,}", cell(e["token"]), str(e["chars"]), str(e["bytes"])]
        if traced:
            row += [f"{e['docs']:,}", latex_escape(f"{e['top_source']} ({pct(e['top_share'], 0)})")
                    if e["top_source"] else "--"]
        rows.append(row)
    paths.append(write_table("merges_longest", "rlrr" + ("rl" if traced else ""), header, rows, SCRIPT, tables_dir))
    return paths


def plot(res: dict[str, Any]) -> Path:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    labels = [f"{lo:,}-{hi:,}" for lo, hi in (b["ranks"] for b in res["buckets"])]
    bottom = [0.0] * len(labels)
    for color, cat in zip(PALETTE, CATEGORIES):
        vals = [b["shares"][cat] for b in res["buckets"]]
        if not any(vals):
            continue
        bars = ax.bar(labels, vals, bottom=bottom, color=color, edgecolor="white", linewidth=1.0, width=0.62,
                      label=CATEGORY_LABELS[cat])
        for bar, v, b0 in zip(bars, vals, bottom):
            if v >= 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2, b0 + v / 2, f"{100 * v:.0f}%", ha="center",
                        va="center", fontsize=7, color="white" if cat in ("LETTER", "WORD", "OTHER") else "#1f1f1e")
        bottom = [b0 + v for b0, v in zip(bottom, vals)]
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda y, _: f"{100 * y:.0f}%")
    ax.set_xlabel("Merge rank")
    ax.set_ylabel("Share of merges")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
    return save_figure(fig, "merge_categories")


def draft(analysis: dict[str, Any]) -> str:
    """The Q1 draft for docs/analysis_notes.md, from analysis.json alone."""
    res = analysis["q1_merges"]
    letters = res["early"]["letters"]
    invalid = res["early"]["not_valid_utf8"]
    lead = [e for e in invalid if e["category"] == "SPACE+LEAD"]
    cut = [e for e in invalid if e["category"] != "SPACE+LEAD"]

    def listing(entries: list[dict[str, Any]], k: int) -> str:
        return ", ".join(f"{e['token']} at {e['rank']}" for e in entries[:k]) + (", ..." if len(entries) > k else "")

    fw, any_w, lit = res["first_word"], res["first_word_any"], res["first_word_brief_literal"]
    b0, b_last = res["buckets"][0], res["buckets"][-1]
    s1 = (f"Of the first {EARLY} merges, {len(letters)} rebuild a two-byte letter from its bytes ({listing(letters, 6)}). "
          f"{len(invalid)} produce a token that is not valid UTF-8 on its own: {len(lead)} are a space plus the lead "
          f"byte of the next letter ({listing(lead, 3)}), the byte-level start of a word"
          + (f", and {len(cut)} end in the middle of a character ({listing(cut, 3)})" if cut else "") + ".")
    s2 = (f"The first whole word is {fw['token']} ({fw['bytes']} bytes) at merge {fw['rank']}, a pre-token "
          f"{fw['pretoken_count']:,} times in train.txt")
    notes = []
    if any_w and any_w["rank"] < fw["rank"]:
        notes.append(f"the word ending {any_w['token']} passes the frequency test earlier, at merge {any_w['rank']}, "
                     f"because it stands alone {any_w['pretoken_count']:,} times")
    if lit and lit["rank"] < fw["rank"]:
        notes.append(f"the brief's test without a two-letter minimum would pick {lit['token']} at merge {lit['rank']}, "
                     f"a single letter that stands alone {lit['pretoken_count']:,} times")
    s2 += (" (" + "; ".join(notes) + ")" if notes else "") + "."
    s3 = (f"Across the merge list, letters fall from {pct(b0['shares']['LETTER'], 0)} of merges "
          f"{b0['ranks'][0]}-{b0['ranks'][1]} to {pct(b_last['shares']['LETTER'], 0)} of merges "
          f"{b_last['ranks'][0]:,}-{b_last['ranks'][1]:,}, while words rise from {pct(b0['shares']['WORD'], 0)} to "
          f"{pct(b_last['shares']['WORD'], 0)}: bytes, then letters, then words.")
    lg = res["longest"]
    s4 = (f"The {len(lg)} longest tokens ({lg[-1]['bytes']}-{lg[0]['bytes']} bytes, e.g. {lg[0]['token']}) are "
          + ("all whole words, none junk" if not res["longest_junk"] else
             f"words except {', '.join(res['longest_junk'])}") + ".")
    if lg and "top_source" in lg[0]:
        wiki = [e for e in lg if e["top_source"] == "wiki_ur" and (e["top_share"] or 0) >= 0.9]
        s4 += (f" But {len(wiki)} of them occur almost only (in 90% or more of their documents) in Urdu Wikipedia ("
               + ", ".join(e["token"] for e in wiki[:4]) + "). Reading documents that contain them turns up templated stub "
               "articles that differ only in a name")
        tmpl = [(name, sum(t["docs"].values()), t["docs"].get("wiki_ur", 0)) for name, t in res["templates"].items()]
        if tmpl:
            s4 += (": " + "; ".join(f"{wiki_n:,} train.txt documents are {name} stubs" for name, _, wiki_n in tmpl)
                   + (f", together {pct(sum(w for _, _, w in tmpl) / res['wiki_train_docs'])} of the "
                      f"{res['wiki_train_docs']:,} Wikipedia documents" if res.get("wiki_train_docs") else ""))
        for e in res["traced_extra"]:
            if e["rank"]:
                s4 += (f". This is why {e['token']} is already merge {e['rank']} ({e['count']:,} pair occurrences): "
                       f"{pct(e['top_share'], 0)} of the {e['docs']:,} documents containing it are {e['top_source']}")
        s4 += ". A data-quality finding: templated Wikipedia text over-represents a few phrases and names."
    return (f"{s1} {s2}\n\n{s3}\n\n{s4}\n\nTables: `report/tables/merges.tex`, `report/tables/merges_longest.tex`. "
            "Figure: `report/figures/merge_categories.pdf`.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="where Step 9 cached the word counts")
    p.add_argument("--train", type=Path, default=TRAIN_TXT, help="for tracing the longest tokens")
    p.add_argument("--no-trace", action="store_true", help="do not search train.txt for the longest tokens")
    p.add_argument("--min-count", type=int, default=MIN_COUNT, help="pre-token count for WORD")
    p.add_argument("--analysis-json", type=Path, default=ANALYSIS_JSON)
    p.add_argument("--notes", type=Path, default=NOTES)
    p.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("analysis")
    tok = BPETokenizer.load(args.tokenizer)
    counts = load_counts(tok, args.cache_dir)
    sources, wiki_docs = None, None
    if not args.no_trace:
        ids = args.train.with_suffix(".ids")
        if not (args.train.exists() and ids.exists()):
            raise SystemExit(f"tracing needs {args.train} and {ids} (or --no-trace)")
        first = tok.first_merge_id
        longest = sorted(range(first, tok.vocab_size), key=lambda i: (-len(tok.vocab[i]), i))[:LONGEST]
        needles = sorted({tok.vocab[i].decode("utf-8", "replace").lstrip(" ") for i in longest}
                         | {t.lstrip(" ") for t in TRACE_EXTRA} | set(TEMPLATES.values()))
        with open(ids, encoding="utf-8") as f:
            wiki_docs = sum(1 for line in f if line.startswith("wiki_ur:"))
        start = time.perf_counter()
        sources = trace(needles, args.train, ids)
        log.info("traced %d tokens through %s in %.1f s", len(needles), args.train, time.perf_counter() - start)
    res = analyse(tok, counts, sources, args.min_count, wiki_docs)
    analysis = update_analysis("q1_merges", res, args.analysis_json)
    for path in write_tables(res, args.tables_dir):
        log.info("wrote %s", path)
    log.info("wrote %s", plot(res))
    update_notes("q1", "Q1. What did the merges learn?", draft(analysis), args.notes)
    log.info("first word %s; early letters %d, not valid UTF-8 %d; longest %s (%d bytes); updated %s and %s",
             res["first_word"], len(res["early"]["letters"]), len(res["early"]["not_valid_utf8"]),
             res["longest"][0]["token"], res["longest"][0]["bytes"], args.analysis_json, args.notes)
    print(draft(analysis))
    return 0


if __name__ == "__main__":
    sys.exit(main())
