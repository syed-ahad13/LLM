"""Step 5: normalize each document, drop lines that are not Urdu, then drop documents too short to keep.

    python clean/normalize_filter.py --limit 2000                  # smoke run over every source in data/interim
    python clean/normalize_filter.py --source ufal_umc --jobs 7
    python clean/normalize_filter.py --jobs 7                      # full run (tmux command in README.md)
    python clean/normalize_filter.py --table                       # per-source table from data/stats

For each data/interim/<source>.jsonl.gz, in input order:

1. normalize.normalize(text). A document left empty is dropped
   (normalize stage: drop_docs_empty).
2. clean.linefilter.classify_line on every line. Dropped lines are counted
   per reason (langfilter stage: drop_lines_<reason>, drop_bytes_<reason>).
3. Document rules, in this order:
   - sindhi_doc: the lines dropped as Sindhi hold at least SINDHI_DOC_SHARE
     of the normalized bytes. The rest of such a document is Sindhi too
     (titles and credit lines without Sindhi-only letters).
   - no_lines_left: every line was dropped.
   - min_words: fewer than the source's minimum number of words
     (whitespace-separated) is left. The minimum is MIN_WORDS.get(source,
     DEFAULT_MIN_WORDS), or --min-words for every source.

Writes:
- data/clean/filtered/<source>.jsonl.gz: the interim schema, with the text replaced;
- data/samples/filtered/<source>.sample.jsonl: a ~2 MB sample, picked as in Step 3;
- data/samples/dropped_lines/<source>.jsonl: up to --dropped-examples distinct
  dropped lines per reason (the smallest blake2b of the line, so the pick does
  not depend on the read order);
- data/stats/normalize.json and data/stats/langfilter.json: docs, lines and
  bytes in and out per source, the drops by reason, and the run time.
"""

import argparse
import hashlib
import heapq
import json
import multiprocessing
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from itertools import batched, islice
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import INTERIM_DIR, SAMPLE_BYTES, SAMPLES_DIR, Sink  # noqa: E402
from clean.linefilter import REASONS, dropped_bytes, filter_lines  # noqa: E402
from common import STATS_DIR, Stats, iter_jsonl, ordered_imap, setup_logging, write_jsonl  # noqa: E402
from normalize import normalize  # noqa: E402

FILTERED_DIR = ROOT / "data" / "clean" / "filtered"
NORMALIZE_STAGE = "normalize"
FILTER_STAGE = "langfilter"
DEFAULT_MIN_WORDS = 10
MIN_WORDS = {"ufal_umc": 3}  # one sentence per document (Step 3)
# VOA's Sindhi articles have >= 0.785 of their bytes in Sindhi lines; Urdu Wikipedia articles about Sindh or
# Saraiki (with Sindhi titles or letter tables) reach 0.655 (docs/decisions.md, Step 5).
SINDHI_DOC_SHARE = 0.72
DOC_REASONS = ("sindhi_doc", "no_lines_left", "min_words")
BATCH_DOCS = 256
DROPPED_EXAMPLES = 200

Dropped = list[tuple[str, str, str]]  # (reason, line, doc_id)


def _size(text: str) -> tuple[int, int]:
    """(lines, UTF-8 bytes) of a text, counting it as Step 3 does."""
    return text.count("\n") + 1, len(text.encode("utf-8"))


def process_doc(doc: dict[str, Any], min_words: int, norm: Counter[str], filt: Counter[str],
                dropped_out: Dropped) -> dict[str, Any] | None:
    """Run steps 1-3 on one document, adding to the counters. Returns the kept document or None."""
    lines, size = _size(doc["text"])
    norm.update(docs_in=1, lines_in=lines, bytes_in=size)
    text = normalize(doc["text"])
    if not text:
        norm["drop_docs_empty"] += 1
        return None
    lines, size = _size(text)
    norm.update(docs_out=1, lines_out=lines, bytes_out=size)
    filt.update(docs_in=1, lines_in=lines, bytes_in=size)

    kept, dropped = filter_lines(text)
    for reason, n in Counter(r for r, _ in dropped).items():
        filt[f"drop_lines_{reason}"] += n
    for reason, n in dropped_bytes(dropped).items():
        filt[f"drop_bytes_{reason}"] += n
    dropped_out.extend((reason, line, doc["doc_id"]) for reason, line in dropped)

    out = "\n".join(kept)
    sindhi = sum(len(line.encode("utf-8")) for reason, line in dropped if reason == "sindhi")
    if sindhi >= SINDHI_DOC_SHARE * size:
        reason = "sindhi_doc"
    elif not kept:
        reason = "no_lines_left"
    elif len(out.split()) < min_words:
        reason = "min_words"
    else:
        lines, size = _size(out)
        filt.update(docs_out=1, lines_out=lines, bytes_out=size, words_out=len(out.split()))
        return {**doc, "text": out}
    filt[f"drop_docs_{reason}"] += 1
    filt[f"drop_doc_bytes_{reason}"] += len(out.encode("utf-8")) if kept else 0
    return None


def process_batch(docs: Sequence[dict[str, Any]], min_words: int) -> tuple[list[dict[str, Any]], Counter[str],
                                                                           Counter[str], Dropped]:
    norm: Counter[str] = Counter()
    filt: Counter[str] = Counter()
    dropped: Dropped = []
    out = [d for d in (process_doc(doc, min_words, norm, filt, dropped) for doc in docs) if d is not None]
    return out, norm, filt, dropped


class Examples:
    """Up to n distinct dropped lines per reason: those with the smallest blake2b of the line."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._heaps: dict[str, list[tuple[int, str, str]]] = {}  # reason -> max-heap (-hash, line, doc_id)
        self._seen: dict[str, set[str]] = {}

    def offer(self, dropped: Iterable[tuple[str, str, str]]) -> None:
        if self.n <= 0:
            return
        for reason, line, doc_id in dropped:
            heap = self._heaps.setdefault(reason, [])
            seen = self._seen.setdefault(reason, set())
            if line in seen:
                continue
            h = int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")
            if len(heap) < self.n:
                heapq.heappush(heap, (-h, line, doc_id))
                seen.add(line)
            elif h < -heap[0][0]:
                _, old, _ = heapq.heapreplace(heap, (-h, line, doc_id))
                seen.discard(old)
                seen.add(line)

    def rows(self) -> list[dict[str, str]]:
        return [{"reason": reason, "line": line, "doc_id": doc_id}
                for reason in sorted(self._heaps)
                for _, line, doc_id in sorted(self._heaps[reason], key=lambda e: -e[0])]


def run_source(source: str, args: argparse.Namespace, log: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter one source; returns its (normalize, langfilter) stats rows."""
    path = args.interim_dir / f"{source}.jsonl.gz"
    min_words = args.min_words if args.min_words is not None else MIN_WORDS.get(source, DEFAULT_MIN_WORDS)
    norm: Counter[str] = Counter()
    filt: Counter[str] = Counter()
    examples = Examples(args.dropped_examples)
    sink = Sink(source, sample_bytes=args.sample_bytes)
    docs: Iterator[dict[str, Any]] = iter_jsonl(path)
    if args.limit is not None:
        docs = islice(docs, args.limit)
    start = time.perf_counter()
    pool = multiprocessing.Pool(args.jobs) if args.jobs > 1 else None

    def kept_docs() -> Iterator[dict[str, Any]]:
        work = partial(process_batch, min_words=min_words)
        try:
            for out, n, f, dropped in ordered_imap(pool, work, batched(docs, BATCH_DOCS), window=4 * args.jobs):
                norm.update(n)
                filt.update(f)
                examples.offer(dropped)
                yield from out
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()

    out_path = args.out_dir / f"{source}.jsonl.gz"
    write_jsonl(out_path, sink.accept(kept_docs()))
    seconds = round(time.perf_counter() - start, 1)
    write_jsonl(args.samples_dir / "filtered" / f"{source}.sample.jsonl", map(json.loads, sink.sample_lines()))
    write_jsonl(args.samples_dir / "dropped_lines" / f"{source}.jsonl", examples.rows())

    norm_row: dict[str, Any] = {k: norm[k] for k in ("docs_in", "lines_in", "bytes_in", "docs_out", "lines_out",
                                                     "bytes_out", "drop_docs_empty")}
    filt_row: dict[str, Any] = {k: filt[k] for k in ("docs_in", "lines_in", "bytes_in", "docs_out", "lines_out",
                                                     "bytes_out", "words_out")}
    for reason in REASONS:
        filt_row[f"drop_lines_{reason}"] = filt[f"drop_lines_{reason}"]
        filt_row[f"drop_bytes_{reason}"] = filt[f"drop_bytes_{reason}"]
    for reason in DOC_REASONS:
        filt_row[f"drop_docs_{reason}"] = filt[f"drop_docs_{reason}"]
        filt_row[f"drop_doc_bytes_{reason}"] = filt[f"drop_doc_bytes_{reason}"]
    filt_row.update(min_words=min_words, seconds=seconds, jobs=args.jobs)
    for row in (norm_row, filt_row):
        row["limit"] = args.limit or 0
    log.info("%s: %d docs in, %d out; %.2f MB in, %.2f MB after normalize, %.2f MB out; %.0f s -> %s",
             source, norm_row["docs_in"], filt_row["docs_out"], norm_row["bytes_in"] / 1e6,
             norm_row["bytes_out"] / 1e6, filt_row["bytes_out"] / 1e6, seconds, out_path)
    return norm_row, filt_row


def find_sources(interim_dir: Path, wanted: Sequence[str] | None) -> list[str]:
    found = sorted(p.name.removesuffix(".jsonl.gz") for p in interim_dir.glob("*.jsonl.gz"))
    if wanted:
        missing = [s for s in wanted if s not in found]
        if missing:
            raise SystemExit(f"no interim file in {interim_dir} for: {', '.join(missing)}")
        return list(wanted)
    if not found:
        raise SystemExit(f"no *.jsonl.gz in {interim_dir}; run clean/extract_all.py first")
    return found


def format_table(norm: dict[str, dict[str, Any]], filt: dict[str, dict[str, Any]]) -> str:
    """One row per source: MB in, after normalize and out, docs, and the drops that cost the most."""
    header = ("source", "docs in", "docs out", "MB in", "MB norm", "MB out", "removed", "biggest drops (MB)")
    rows = [header]
    totals = [0, 0, 0.0, 0.0, 0.0]
    for source in sorted(filt):
        n, f = norm.get(source, {}), filt[source]
        mb_in, mb_norm, mb_out = n.get("bytes_in", 0) / 1e6, n.get("bytes_out", 0) / 1e6, f.get("bytes_out", 0) / 1e6
        drops = {k.removeprefix("drop_bytes_"): v for k, v in f.items() if k.startswith("drop_bytes_")}
        drops.update({k.removeprefix("drop_doc_bytes_") + " (docs)": v for k, v in f.items()
                      if k.startswith("drop_doc_bytes_")})
        top = ", ".join(f"{k}={v / 1e6:,.2f}" for k, v in sorted(drops.items(), key=lambda kv: -kv[1]) if v)
        limit = f" (limit {f['limit']:,})" if f.get("limit") else ""
        removed = f"{1 - mb_out / mb_in:.2%}" if mb_in else "-"
        rows.append((source + limit, f"{n.get('docs_in', 0):,}", f"{f.get('docs_out', 0):,}", f"{mb_in:,.2f}",
                     f"{mb_norm:,.2f}", f"{mb_out:,.2f}", removed, top or "none"))
        for i, v in enumerate((n.get("docs_in", 0), f.get("docs_out", 0), mb_in, mb_norm, mb_out)):
            totals[i] += v
    removed = f"{1 - totals[4] / totals[2]:.2%}" if totals[2] else "-"
    rows.append(("total", f"{totals[0]:,}", f"{totals[1]:,}", f"{totals[2]:,.2f}", f"{totals[3]:,.2f}",
                 f"{totals[4]:,.2f}", removed, ""))
    widths = [max(len(r[i]) for r in rows) for i in range(7)]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join([r[0].ljust(widths[0])] + [r[j].rjust(widths[j]) for j in range(1, 7)] + [r[7]]))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths) + "  " + "-" * 18)
    return "\n".join(lines)


def load_stage(stats_dir: Path, stage: str) -> dict[str, dict[str, Any]]:
    path = stats_dir / f"{stage}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", help="process only this source (repeatable; default: all in interim)")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="read only the first N documents (smoke runs)")
    p.add_argument("--jobs", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1),
                   help="worker processes (default: CPUs - 1; 1 = no pool)")
    p.add_argument("--min-words", type=int, default=None, metavar="N",
                   help=f"minimum words per document for every source (default {DEFAULT_MIN_WORDS}, "
                        f"except {', '.join(f'{k} {v}' for k, v in MIN_WORDS.items())})")
    p.add_argument("--dropped-examples", type=int, default=DROPPED_EXAMPLES, metavar="N",
                   help=f"distinct dropped lines kept per reason (default {DROPPED_EXAMPLES})")
    p.add_argument("--sample-bytes", type=int, default=SAMPLE_BYTES)
    p.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    p.add_argument("--out-dir", type=Path, default=FILTERED_DIR)
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    p.add_argument("--table", action="store_true", help="only print the table from the stats files")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.table:
        log = setup_logging("normalize_filter")
        for source in find_sources(args.interim_dir, args.source):
            norm_row, filt_row = run_source(source, args, log)
            stats = Stats(args.stats_dir)
            stats.update(NORMALIZE_STAGE, source, norm_row)
            stats.update(FILTER_STAGE, source, filt_row)
            stats.dump()
    print(format_table(load_stage(args.stats_dir, NORMALIZE_STAGE), load_stage(args.stats_dir, FILTER_STAGE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
