"""Step 6c: corpus totals per split and overall: MB, MiB, documents, words and unique words.

    python clean/corpus_stats.py                   # data/final/{train,val,test}.txt -> data/stats/corpus_totals.json
    python clean/corpus_stats.py --jobs 4

Definitions (CLAUDE.md):
- MB = 10^6 bytes, MiB = 2^20 bytes, of the split file. The file holds each
  document followed by <|endoftext|>; text_bytes leaves the separators out.
- words = str.split() tokens of each document (the separator is not a word);
- unique words = distinct maximal runs of [\\p{L}\\p{M}\\u200c] (letters,
  combining marks, ZWNJ). The total counts each distinct run once over all
  three splits.

Each file is cut into ranges at document boundaries, one worker counts each
range, and the per-range Counters are merged. The script also counts a few
marker characters that the data card's gaps section uses: U+0768 ݨ (Punjabi
and Saraiki noon, a sign of Shahmukhi), U+06AA ڪ (Sindhi kaf) and U+0647 ه
(should be 0 after normalization).

If data/final/manifest.json exists, the document counts and file sizes must
match it.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.budget import FINAL_DIR  # noqa: E402
from clean.split import EOT, READ_BLOCK, SPLITS, read_docs  # noqa: E402
from common import STATS_DIR, setup_logging, write_json  # noqa: E402

WORD_RUN = regex.compile(r"[\p{L}\p{M}‌]+")
MARKERS = {"U+0768": "ݨ", "U+06AA": "ڪ", "U+0647": "ه"}
OUT_JSON = STATS_DIR / "corpus_totals.json"


def next_boundary(f: Any, offset: int, size: int) -> int:
    """The first document start at or after offset: 0, the end of the file, or just after a separator."""
    if offset <= 0:
        return 0
    if offset >= size:
        return size
    base = max(0, offset - len(EOT))  # a separator that ends exactly at offset counts
    f.seek(base)
    carry = b""
    while True:
        block = f.read(READ_BLOCK)
        if not block:
            return size
        data = carry + block
        i = data.find(EOT)
        while i != -1 and base + i + len(EOT) < offset:
            i = data.find(EOT, i + 1)
        if i != -1:
            return base + i + len(EOT)
        carry = data[-(len(EOT) - 1):]
        base += len(data) - len(carry)


def ranges(path: Path, n: int) -> list[tuple[int, int]]:
    """Up to n byte ranges of about equal size, each starting and ending at a document boundary."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        cuts = sorted({next_boundary(f, size * i // n, size) for i in range(n)} | {size})
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


def count_range(item: tuple[str, str, int, int]) -> tuple[str, dict[str, int], Counter[str]]:
    """Counts and the word-run Counter for one byte range of one split file."""
    split, path, start, end = item
    counts: Counter[str] = Counter()
    runs: Counter[str] = Counter()
    for raw in read_docs(Path(path), start, end):
        text = raw.decode("utf-8")
        counts["docs"] += 1
        counts["text_bytes"] += len(raw)
        counts["words"] += len(text.split())
        runs.update(WORD_RUN.findall(text))
        for name, ch in MARKERS.items():
            if (c := text.count(ch)):
                counts[f"char_{name}"] += c
                counts[f"docs_with_{name}"] += 1
    return split, dict(counts), runs


def summary(counts: Counter[str], size: int, unique: int) -> dict[str, Any]:
    row: dict[str, Any] = {"bytes": size, "mb": round(size / 1e6, 2), "mib": round(size / 2**20, 2),
                           "text_bytes": counts["text_bytes"], "separator_bytes": size - counts["text_bytes"],
                           "docs": counts["docs"], "words": counts["words"], "unique_words": unique}
    row["markers"] = {k: counts[k] for k in sorted(counts) if k.startswith(("char_", "docs_with_"))}
    return row


def run(final_dir: Path, jobs: int, log: Any) -> dict[str, Any]:
    start = time.perf_counter()
    files = {split: final_dir / f"{split}.txt" for split in SPLITS}
    for path in files.values():
        if not path.exists():
            raise SystemExit(f"no {path}; run clean/split.py first")
    out: dict[str, Any] = {"definitions": {
        "MB": "10^6 bytes of the split file (MiB: 2^20), separators included; text_bytes leaves them out",
        "words": "str.split() tokens per document",
        "unique_words": "distinct maximal runs of [\\p{L}\\p{M}\\u200c]; the total counts each run once",
        "markers": "U+0768 Shahmukhi/Saraiki noon, U+06AA Sindhi kaf, U+0647 Arabic heh (0 after normalization)",
    }, "splits": {}}
    total_counts: Counter[str] = Counter()
    total_runs: Counter[str] = Counter()
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        # train first, so its Counter becomes the running total and only one big Counter is held.
        for split in ("train", "val", "test"):
            path = files[split]
            items = [(split, str(path), a, b) for a, b in ranges(path, jobs * 2 if split == "train" else 1)]
            counts: Counter[str] = Counter()
            runs: Counter[str] = Counter() if total_runs else total_runs
            for _, c, r in pool.imap_unordered(count_range, items):
                counts.update(c)
                runs.update(r)
            out["splits"][split] = summary(counts, os.path.getsize(path), len(runs))
            if runs is not total_runs:
                total_runs.update(runs)
            total_counts.update(counts)
            log.info("%s: %d docs, %d words, %d unique words", split, counts["docs"], counts["words"], len(runs))
    out["splits"] = {s: out["splits"][s] for s in SPLITS}
    out["total"] = summary(total_counts, sum(os.path.getsize(p) for p in files.values()), len(total_runs))
    out["seconds"] = round(time.perf_counter() - start, 1)
    out["jobs"] = jobs

    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        for split in SPLITS:
            want, got = manifest["splits"][split], out["splits"][split]
            if (want["docs"], want["bytes"]) != (got["docs"], got["bytes"]):
                raise ValueError(f"{split}: {got['docs']} docs / {got['bytes']} bytes, but the manifest says "
                                 f"{want['docs']} / {want['bytes']}")
        out["manifest_sha256"] = {s: manifest["splits"][s]["sha256"] for s in SPLITS}
    return out


def table(stats: dict[str, Any]) -> str:
    rows = [("split", "MB", "MiB", "docs", "words", "unique words")]
    for name, r in [*stats["splits"].items(), ("total", stats["total"])]:
        rows.append((name, f"{r['mb']:,.2f}", f"{r['mib']:,.2f}", f"{r['docs']:,}", f"{r['words']:,}",
                     f"{r['unique_words']:,}"))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join([r[0].ljust(widths[0])] + [r[j].rjust(widths[j]) for j in range(1, len(r))]))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final-dir", type=Path, default=FINAL_DIR)
    p.add_argument("--out", type=Path, default=OUT_JSON, help="default data/stats/corpus_totals.json")
    p.add_argument("--jobs", type=int, default=max(1, min(6, (multiprocessing.cpu_count() or 2) - 1)))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("corpus_stats")
    stats = run(args.final_dir, args.jobs, log)
    write_json(args.out, stats)
    print(table(stats))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
