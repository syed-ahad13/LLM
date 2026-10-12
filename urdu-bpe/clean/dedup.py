"""Step 5b: remove boilerplate lines, then exact duplicate documents (within and across sources).

    python clean/dedup.py --report                 # per source: what each boilerplate cutoff would remove
    python clean/dedup.py --limit 2000             # smoke run
    python clean/dedup.py                          # full run (tmux command in README.md)
    python clean/dedup.py --table

Reads data/clean/filtered/<source>.jsonl.gz (Step 5). Sources are processed in
PRIORITY order, so when two sources hold the same document the copy from the
earlier source is kept: our own scrape first (it must count toward the 50 MB
we collected), then edited sources, then the web crawl. For each source:

Pass 1. For every distinct line of every document, count the documents that
   contain it (a line repeated inside one document counts once). Lines are
   identified by an 8-byte blake2b hash held in a numpy array, so memory is 8
   bytes per (document, line) pair, not the text.
Pass 2. For each document:
   - remove every literal <|endoftext|> (the split step writes it as the
     separator), then normalize again so no stray spaces are left;
   - drop the boilerplate lines: those in at least the source's cutoff
     number of documents (BOILERPLATE_MIN_DOCS, or --min-docs);
   - drop the document if fewer words than Step 5's minimum are left;
   - drop the document if its text is an exact duplicate of a document kept
     earlier, in this source (duplicate_in_source) or an earlier one
     (duplicate_other_source). Documents are compared by an 8-byte blake2b of
     the text; at 8.4 million documents the chance of any false match is
     about 2e-6.

--report runs pass 1 only and prints, for each cutoff in REPORT_CUTOFFS, the
distinct lines and bytes it would remove, with example lines per band.

Writes data/clean/<source>.jsonl.gz, a ~2 MB sample to data/samples/clean/,
the most frequent removed lines with their document counts to
data/samples/boilerplate/<source>.jsonl, and stats to data/stats/boilerplate.json
and data/stats/dedup.json.
"""

import argparse
import hashlib
import heapq
import json
import sys
import time
from array import array
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import SAMPLE_BYTES, SAMPLES_DIR, Sink  # noqa: E402
from clean.normalize_filter import DEFAULT_MIN_WORDS, FILTERED_DIR, MIN_WORDS  # noqa: E402
from common import STATS_DIR, Stats, iter_jsonl, setup_logging, write_jsonl  # noqa: E402
from normalize import normalize  # noqa: E402

CLEAN_DIR = ROOT / "data" / "clean"
BOILERPLATE_STAGE = "boilerplate"
DEDUP_STAGE = "dedup"
ENDOFTEXT = "<|endoftext|>"
PRIORITY = ("own_scrape", "makhzan", "wiki_ur", "urdu_news_1m", "ufal_umc", "fineweb2_urd")
# From the full-data --report (docs/decisions.md, Step 5b): lines in 5+ documents are credits, bylines,
# headings, link lists and bot-template sentences; below 5 they are mostly real shared content. VOA words its
# AP/Reuters/AFP credit a dozen ways, each in 3-4 articles, so own_scrape uses 3.
DEFAULT_MIN_DOCS = 5
BOILERPLATE_MIN_DOCS: dict[str, int] = {"own_scrape": 3}
REPORT_CUTOFFS = (2, 3, 5, 10, 20, 50, 100, 1000)
TOP_LINES = 200


def h64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def iter_docs(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    docs: Iterator[dict[str, Any]] = iter_jsonl(path)
    return islice(docs, limit) if limit is not None else docs


def line_doc_counts(path: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pass 1: (distinct line hashes, documents containing each, UTF-8 bytes of each line)."""
    hashes = array("Q")
    sizes = array("I")
    for doc in iter_docs(path, limit):
        for line in set(doc["text"].split("\n")):
            hashes.append(h64(line))
            sizes.append(len(line.encode("utf-8")))
    all_hashes = np.frombuffer(hashes, dtype=np.uint64)
    unique, first, counts = np.unique(all_hashes, return_index=True, return_counts=True)
    return unique, counts, np.frombuffer(sizes, dtype=np.uint32)[first]


def min_docs_for(source: str, args: argparse.Namespace) -> int:
    return args.min_docs if args.min_docs is not None else BOILERPLATE_MIN_DOCS.get(source, DEFAULT_MIN_DOCS)


def report(source: str, path: Path, args: argparse.Namespace, log: Any) -> None:
    """Print what each cutoff would remove, with example lines from each band of document counts."""
    unique, counts, sizes = line_doc_counts(path, args.limit)
    total = int((counts.astype(np.int64) * sizes).sum())
    docs = sum(1 for _ in iter_docs(path, args.limit))
    print(f"\n== {source}: {docs:,} docs, {len(unique):,} distinct lines, ~{total / 1e6:,.2f} MB of line text")
    print(f"   {'cutoff':>7}  {'lines':>10}  {'MB removed':>10}  {'share':>7}")
    for k in REPORT_CUTOFFS:
        mask = counts >= k
        removed = int((counts[mask].astype(np.int64) * sizes[mask]).sum())
        print(f"   {k:>7,}  {int(mask.sum()):>10,}  {removed / 1e6:>10,.2f}  {removed / max(total, 1):>7.2%}")
    bands = list(zip(REPORT_CUTOFFS, [*REPORT_CUTOFFS[1:], 1 << 62]))
    rng = np.random.default_rng(1337)
    wanted: dict[int, tuple[int, int]] = {}  # hash -> (band index, docs)
    for i, (lo, hi) in enumerate(bands):
        idx = np.flatnonzero((counts >= lo) & (counts < hi))
        for j in rng.choice(idx, size=min(args.report_examples, len(idx)), replace=False):
            wanted[int(unique[j])] = (i, int(counts[j]))
    found: dict[int, str] = {}
    for doc in iter_docs(path, args.limit):
        for line in doc["text"].split("\n"):
            h = h64(line)
            if h in wanted and h not in found:
                found[h] = line
        if len(found) == len(wanted):
            break
    for i, (lo, hi) in enumerate(bands):
        rows = sorted(((n, found.get(h, "?")) for h, (b, n) in wanted.items() if b == i), reverse=True)
        if rows:
            print(f"   -- lines in {lo:,}{'+' if hi == 1 << 62 else f'-{hi - 1:,}'} docs:")
            for n, line in rows:
                print(f"   {n:>7,} | {line[:120]}")
    log.info("%s: reported", source)


class TopLines:
    """The removed lines with the highest document counts (text filled in during pass 2)."""

    def __init__(self, unique: np.ndarray, counts: np.ndarray, boiler_mask: np.ndarray, n: int) -> None:
        idx = np.flatnonzero(boiler_mask)
        top = idx[np.argsort(-counts[idx], kind="stable")[:n]]
        self.docs = {int(unique[j]): int(counts[j]) for j in top}
        self.text: dict[int, str] = {}

    def offer(self, h: int, line: str) -> None:
        if h in self.docs and h not in self.text:
            self.text[h] = line

    def rows(self) -> list[dict[str, Any]]:
        return [{"docs": n, "line": self.text.get(h, "")}
                for h, n in sorted(self.docs.items(), key=lambda kv: (-kv[1], kv[0]))]


def clean_source(source: str, args: argparse.Namespace, earlier: np.ndarray,
                 log: Any) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Passes 1 and 2 for one source. Returns its stats rows and the sorted hashes of all documents kept so far."""
    path = args.filtered_dir / f"{source}.jsonl.gz"
    start = time.perf_counter()
    min_docs = min_docs_for(source, args)
    min_words = MIN_WORDS.get(source, DEFAULT_MIN_WORDS)
    unique, counts, sizes = line_doc_counts(path, args.limit)
    boiler_mask = counts >= min_docs
    boiler = set(unique[boiler_mask].tolist())
    top = TopLines(unique, counts, boiler_mask, TOP_LINES)
    bp: Counter[str] = Counter()
    dd: Counter[str] = Counter()
    seen: set[int] = set()
    sink = Sink(source, sample_bytes=args.sample_bytes)

    def kept_docs() -> Iterator[dict[str, Any]]:
        for doc in iter_docs(path, args.limit):
            text = doc["text"]
            bp.update(docs_in=1, lines_in=text.count("\n") + 1, bytes_in=len(text.encode("utf-8")))
            if ENDOFTEXT in text:
                bp["endoftext_removed"] += text.count(ENDOFTEXT)
                text = normalize(text.replace(ENDOFTEXT, " "))
            kept = []
            for line in text.split("\n"):
                h = h64(line)
                if h in boiler:
                    bp["drop_lines_boilerplate"] += 1
                    bp["drop_bytes_boilerplate"] += len(line.encode("utf-8"))
                    top.offer(h, line)
                else:
                    kept.append(line)
            text = "\n".join(kept)
            size = len(text.encode("utf-8"))
            if not kept or len(text.split()) < min_words:
                bp["drop_docs_min_words"] += 1
                bp["drop_doc_bytes_min_words"] += size if kept else 0
                continue
            bp.update(docs_out=1, lines_out=len(kept), bytes_out=size)
            dd.update(docs_in=1, bytes_in=size)
            dh = h64(text)
            if dh in seen:
                reason = "duplicate_in_source"
            elif earlier.size and earlier[min(np.searchsorted(earlier, dh), earlier.size - 1)] == dh:
                reason = "duplicate_other_source"
            else:
                seen.add(dh)
                dd.update(docs_out=1, bytes_out=size, words_out=len(text.split()))
                yield {**doc, "text": text}
                continue
            dd[f"drop_docs_{reason}"] += 1
            dd[f"drop_bytes_{reason}"] += size

    out_path = args.out_dir / f"{source}.jsonl.gz"
    write_jsonl(out_path, sink.accept(kept_docs()))
    write_jsonl(args.samples_dir / "clean" / f"{source}.sample.jsonl", map(json.loads, sink.sample_lines()))
    write_jsonl(args.samples_dir / "boilerplate" / f"{source}.jsonl", top.rows())
    seconds = round(time.perf_counter() - start, 1)

    bp_row: dict[str, Any] = {k: bp[k] for k in (
        "docs_in", "lines_in", "bytes_in", "docs_out", "lines_out", "bytes_out", "drop_lines_boilerplate",
        "drop_bytes_boilerplate", "drop_docs_min_words", "drop_doc_bytes_min_words", "endoftext_removed")}
    bp_row.update(min_docs=min_docs, boilerplate_distinct_lines=int(boiler_mask.sum()), limit=args.limit or 0,
                  seconds=seconds)
    dd_row: dict[str, Any] = {k: dd[k] for k in (
        "docs_in", "bytes_in", "docs_out", "bytes_out", "words_out", "drop_docs_duplicate_in_source",
        "drop_bytes_duplicate_in_source", "drop_docs_duplicate_other_source", "drop_bytes_duplicate_other_source")}
    dd_row["limit"] = args.limit or 0
    log.info("%s: %d docs in, %d out; %.2f MB in, %.2f MB after boilerplate, %.2f MB out; %.0f s -> %s",
             source, bp_row["docs_in"], dd_row["docs_out"], bp_row["bytes_in"] / 1e6, bp_row["bytes_out"] / 1e6,
             dd_row["bytes_out"] / 1e6, seconds, out_path)
    kept_hashes = np.fromiter(seen, dtype=np.uint64, count=len(seen))
    return bp_row, dd_row, np.union1d(earlier, kept_hashes)


def find_sources(filtered_dir: Path, wanted: Sequence[str] | None) -> list[str]:
    found = {p.name.removesuffix(".jsonl.gz") for p in filtered_dir.glob("*.jsonl.gz")}
    if wanted:
        missing = [s for s in wanted if s not in found]
        if missing:
            raise SystemExit(f"no filtered file in {filtered_dir} for: {', '.join(missing)}")
        found &= set(wanted)
    if not found:
        raise SystemExit(f"no *.jsonl.gz in {filtered_dir}; run clean/normalize_filter.py first")
    return sorted(found, key=lambda s: (PRIORITY.index(s) if s in PRIORITY else len(PRIORITY), s))


def load_stage(stats_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    path = stats_dir / f"{stage}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_table(bp: dict[str, dict[str, Any]], dd: dict[str, dict[str, Any]]) -> str:
    header = ("source", "docs in", "docs out", "MB in", "boilerplate MB", "dup docs", "dup MB", "MB out", "removed")
    rows = [header]
    tot = Counter()
    for source in sorted(dd, key=lambda s: (PRIORITY.index(s) if s in PRIORITY else len(PRIORITY), s)):
        b, d = bp.get(source, {}), dd[source]
        dup_docs = d.get("drop_docs_duplicate_in_source", 0) + d.get("drop_docs_duplicate_other_source", 0)
        dup_mb = (d.get("drop_bytes_duplicate_in_source", 0) + d.get("drop_bytes_duplicate_other_source", 0)) / 1e6
        mb_in, mb_out = b.get("bytes_in", 0) / 1e6, d.get("bytes_out", 0) / 1e6
        limit = f" (limit {d['limit']:,})" if d.get("limit") else ""
        rows.append((source + limit, f"{b.get('docs_in', 0):,}", f"{d.get('docs_out', 0):,}", f"{mb_in:,.2f}",
                     f"{b.get('drop_bytes_boilerplate', 0) / 1e6:,.2f}", f"{dup_docs:,}", f"{dup_mb:,.2f}",
                     f"{mb_out:,.2f}", f"{1 - mb_out / mb_in:.2%}" if mb_in else "-"))
        tot.update(docs_in=b.get("docs_in", 0), docs_out=d.get("docs_out", 0), mb_in=mb_in, mb_out=mb_out,
                   bp=b.get("drop_bytes_boilerplate", 0) / 1e6, dup_docs=dup_docs, dup_mb=dup_mb)
    rows.append(("total", f"{tot['docs_in']:,}", f"{tot['docs_out']:,}", f"{tot['mb_in']:,.2f}", f"{tot['bp']:,.2f}",
                 f"{tot['dup_docs']:,}", f"{tot['dup_mb']:,.2f}", f"{tot['mb_out']:,.2f}",
                 f"{1 - tot['mb_out'] / tot['mb_in']:.2%}" if tot["mb_in"] else "-"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join([r[0].ljust(widths[0])] + [r[j].rjust(widths[j]) for j in range(1, len(header))]))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", help="process only this source (repeatable). Cross-source dedup "
                   "only sees the sources processed in the same run, so the full run uses all of them.")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="read only the first N documents per source")
    p.add_argument("--min-docs", type=int, default=None, metavar="N",
                   help=f"boilerplate cutoff for every source (default {DEFAULT_MIN_DOCS}, or BOILERPLATE_MIN_DOCS)")
    p.add_argument("--report", action="store_true", help="print what each cutoff would remove; write nothing")
    p.add_argument("--report-examples", type=int, default=6, metavar="N", help="example lines per band in --report")
    p.add_argument("--sample-bytes", type=int, default=SAMPLE_BYTES)
    p.add_argument("--filtered-dir", type=Path, default=FILTERED_DIR)
    p.add_argument("--out-dir", type=Path, default=CLEAN_DIR)
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    p.add_argument("--table", action="store_true", help="only print the table from the stats files")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.table:
        log = setup_logging("dedup")
        sources = find_sources(args.filtered_dir, args.source)
        if args.report:
            for source in sources:
                report(source, args.filtered_dir / f"{source}.jsonl.gz", args, log)
            return 0
        earlier = np.empty(0, dtype=np.uint64)
        for source in sources:
            bp_row, dd_row, earlier = clean_source(source, args, earlier, log)
            stats = Stats(args.stats_dir)
            stats.update(BOILERPLATE_STAGE, source, bp_row)
            stats.update(DEDUP_STAGE, source, dd_row)
            stats.dump()
    print(format_table(load_stage(args.stats_dir, BOILERPLATE_STAGE), load_stage(args.stats_dir, DEDUP_STAGE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
