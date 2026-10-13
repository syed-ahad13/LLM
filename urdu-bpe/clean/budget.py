"""Step 6a: cleaned size per source and genre, and the byte budget for the final corpus.

    python clean/budget.py                                     # tables + budget under the default policy
    python clean/budget.py --fill fineweb2_urd ufal_umc        # ufal_umc is subsampled into the space left
    python clean/budget.py --total-mb 2600                     # allow a larger corpus
    python clean/budget.py --rescan                            # read data/clean again

Reads every data/clean/<source>.jsonl.gz (Step 5b) once and caches, per source,
the documents and bytes per genre and per hash bucket in
data/stats/budget_scan.json. Only files that changed since are scanned again,
so trying another policy takes a second.

Sizes are what the split files will hold: the UTF-8 text plus the 13 bytes of
<|endoftext|> written after each document ("file bytes"). The total and the
shares use file bytes; the tables also show text MB. MB = 10^6 bytes.

Subsampling: a document is kept when keep_key(doc_id) < keep_below, where
keep_key is int(blake2b(doc_id).hexdigest(), 16) mod 10^6 and keep_below =
rate * 10^6. Rates are multiples of 1e-4, so the scan stores 100 hash values
per bucket and still gives the exact size kept at every rate.

Policy (defaults from the Step 6 brief):
- full sources, every source not in --fill and without --mb, are kept whole;
- --fill sources, in order, take the space left under --total-mb, each up to
  its cap: fineweb2_urd 45% of the final bytes, urdu_news_1m 25%, or 10% when
  most of its bytes have meta.punct_stripped; --cap overrides, and a source
  without a cap may take all the space left;
- --mb SOURCE=X keeps at most X MB (file bytes) of a source.
A fill source that would keep only 1 or 2 documents keeps none, because the
split gives every source in the corpus at least one val and one test document.
A source with nothing left after filling is left out of the corpus (rate 0).
The total must land in [--min-total-mb, --total-mb]. When it cannot (the full
sources alone are too big, say), nothing is written: the script prints why and
exits with status 2.

Writes data/final/budget.json, which clean/split.py reads.
"""

import argparse
import hashlib
import json
import multiprocessing
import shlex
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.dedup import CLEAN_DIR, ENDOFTEXT, PRIORITY  # noqa: E402
from common import STATS_DIR, iter_jsonl, setup_logging, write_json  # noqa: E402

FINAL_DIR = ROOT / "data" / "final"
BUDGET_JSON = FINAL_DIR / "budget.json"
SCAN_JSON = STATS_DIR / "budget_scan.json"
SCAN_VERSION = 1

HASH_MOD = 10**6
BUCKETS = 10**4
PER_BUCKET = HASH_MOD // BUCKETS
EOT_BYTES = len(ENDOFTEXT.encode("utf-8"))

DEFAULT_TOTAL_MB = 1300.0
DEFAULT_MIN_TOTAL_MB = 1000.0
DEFAULT_FILL = ("fineweb2_urd", "urdu_news_1m")
DEFAULT_CAPS = {"fineweb2_urd": 0.45, "urdu_news_1m": 0.25}
PUNCT_STRIPPED_CAP = 0.10
MIN_SPLIT_DOCS = 3  # a source in the corpus gives train, val and test at least one document each


def keep_key(doc_id: str) -> int:
    """The document's subsampling key in [0, 10^6): keep it when the key is below rate * 10^6."""
    return int(hashlib.blake2b(doc_id.encode("utf-8")).hexdigest(), 16) % HASH_MOD


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

def scan_source(path: Path) -> dict[str, Any]:
    """Documents and text bytes per genre, per hash bucket, and with meta.punct_stripped."""
    bucket_docs = [0] * BUCKETS
    bucket_bytes = [0] * BUCKETS
    genres: dict[str, Counter[str]] = defaultdict(Counter)
    punct: Counter[str] = Counter()
    for doc in iter_jsonl(path):
        size = len(doc["text"].encode("utf-8"))
        b = keep_key(doc["doc_id"]) // PER_BUCKET
        bucket_docs[b] += 1
        bucket_bytes[b] += size
        genres[doc.get("genre") or "unknown"].update(docs=1, text_bytes=size)
        if (doc.get("meta") or {}).get("punct_stripped"):
            punct.update(docs=1, text_bytes=size)
    return {
        "docs": sum(bucket_docs),
        "text_bytes": sum(bucket_bytes),
        "genres": {g: dict(c) for g, c in sorted(genres.items())},
        "punct_stripped_docs": punct["docs"],
        "punct_stripped_bytes": punct["text_bytes"],
        "bucket_docs": bucket_docs,
        "bucket_bytes": bucket_bytes,
    }


def _scan_one(item: tuple[str, str]) -> tuple[str, dict[str, Any], float]:
    source, path = item
    start = time.perf_counter()
    return source, scan_source(Path(path)), time.perf_counter() - start


def file_signature(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def clean_files(clean_dir: Path) -> dict[str, Path]:
    files = {p.name.removesuffix(".jsonl.gz"): p for p in clean_dir.glob("*.jsonl.gz")}
    if not files:
        raise SystemExit(f"no *.jsonl.gz in {clean_dir}; run clean/dedup.py first")
    return dict(sorted(files.items(), key=lambda kv: source_order(kv[0])))


def source_order(source: str) -> tuple[int, str]:
    return (PRIORITY.index(source) if source in PRIORITY else len(PRIORITY), source)


def load_scan(clean_dir: Path, scan_path: Path, rescan: bool, jobs: int, log: Any) -> dict[str, Any]:
    """The scan of every input file: cached rows for files that have not changed, fresh ones for the rest."""
    files = clean_files(clean_dir)
    signatures = {s: file_signature(p) for s, p in files.items()}
    sources: dict[str, Any] = {}
    if scan_path.exists() and not rescan:
        with open(scan_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("version") == SCAN_VERSION:
            sources = {s: row for s, row in cached["sources"].items()
                       if s in signatures and cached["files"].get(s) == signatures[s]}
    todo = [(s, str(p)) for s, p in sorted(files.items(), key=lambda kv: -kv[1].stat().st_size) if s not in sources]
    if todo:
        with multiprocessing.Pool(min(jobs, len(todo))) as pool:
            for source, row, seconds in pool.imap_unordered(_scan_one, todo):
                log.info("scanned %s: %d docs, %.2f MB of text in %.0f s", source, row["docs"],
                         row["text_bytes"] / 1e6, seconds)
                sources[source] = row
    scan = {"version": SCAN_VERSION, "files": signatures, "sources": {s: sources[s] for s in files}}
    if todo:
        write_json(scan_path, scan)
    return scan


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class Source:
    name: str
    scan: dict[str, Any]
    cum_docs: np.ndarray = field(init=False)
    cum_file: np.ndarray = field(init=False)
    cum_text: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        zero = np.zeros(1, dtype=np.int64)
        self.cum_docs = np.concatenate([zero, np.cumsum(np.asarray(self.scan["bucket_docs"], dtype=np.int64))])
        self.cum_text = np.concatenate([zero, np.cumsum(np.asarray(self.scan["bucket_bytes"], dtype=np.int64))])
        self.cum_file = self.cum_text + EOT_BYTES * self.cum_docs

    @property
    def punct_stripped_share(self) -> float:
        return self.scan["punct_stripped_bytes"] / self.scan["text_bytes"] if self.scan["text_bytes"] else 0.0

    def file_bytes(self, k: int) -> int:
        """File bytes of the documents in the first k buckets."""
        return int(self.cum_file[k])

    def buckets_within(self, max_file_bytes: float) -> int:
        """Buckets keeping the most documents that fit in max_file_bytes: the fewest buckets that keep them all."""
        most = int(np.searchsorted(self.cum_file, max_file_bytes, side="right")) - 1
        return int(np.searchsorted(self.cum_file, self.cum_file[most], side="left"))


class Infeasible(Exception):
    pass


def default_caps(sources: dict[str, Source]) -> dict[str, float]:
    caps = dict(DEFAULT_CAPS)
    if "urdu_news_1m" in sources and sources["urdu_news_1m"].punct_stripped_share > 0.5:
        caps["urdu_news_1m"] = PUNCT_STRIPPED_CAP
    return caps


def solve(sources: dict[str, Source], fill: Sequence[str], caps: dict[str, float], fixed_mb: dict[str, float],
          total_max: float, total_min: float) -> dict[str, int]:
    """Buckets kept per source (BUCKETS = all of it) under the policy in the module docstring. Sizes in bytes."""
    k: dict[str, int] = {}
    for s, src in sources.items():
        if s in fixed_mb:
            k[s] = src.buckets_within(fixed_mb[s] * 1e6)
            if int(src.cum_docs[k[s]]) < MIN_SPLIT_DOCS:
                k[s] = 0
        elif s not in fill:
            k[s] = BUCKETS
    fixed = sum(sources[s].file_bytes(n) for s, n in k.items())
    if fixed > total_max:
        whole = ", ".join(f"{s} {sources[s].file_bytes(n) / 1e6:,.2f}" for s, n in k.items())
        raise Infeasible(f"the sources kept whole or fixed already hold {fixed / 1e6:,.2f} MB, above "
                         f"--total-mb {total_max / 1e6:,.2f} ({whole})")
    fill = [s for s in fill if s in sources]

    def fill_at(total: float) -> tuple[dict[str, float], float]:
        left, xs = total - fixed, {}
        for s in fill:
            xs[s] = min(sources[s].file_bytes(BUCKETS), caps.get(s, 1.0) * total, left)
            left -= xs[s]
        return xs, left

    # A cap is a share of the final total, so when the caps bind, the fill sources cannot reach --total-mb.
    # The shortfall is zero from `fixed` up to the largest reachable total and grows after it: bisect.
    total = float(total_max)
    if fill_at(total)[1] > 0.5:
        lo, hi = float(fixed), total
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if fill_at(mid)[1] <= 0.5 else (lo, mid)
        total = lo
    xs, _ = fill_at(total)
    for s in fill:
        k[s] = sources[s].buckets_within(xs[s])
        if int(sources[s].cum_docs[k[s]]) < MIN_SPLIT_DOCS:  # split.py needs one val and one test document
            k[s] = 0
    # Rounding down to whole buckets shrinks the total, which can push a capped share just over its cap.
    while True:
        total = sum(sources[s].file_bytes(n) for s, n in k.items())
        over = [s for s, n in k.items() if s in caps and n > 0 and sources[s].file_bytes(n) > caps[s] * total]
        if not over:
            break
        for s in over:
            if s not in fill:
                raise Infeasible(f"{s} is kept whole or fixed but holds more than its cap of {caps[s]:.0%}")
            k[s] -= 1
    if total < total_min:
        raise Infeasible(f"the budget reaches only {total / 1e6:,.2f} MB, below --min-total-mb "
                         f"{total_min / 1e6:,.2f}: the caps or the available text are too small")
    return k


def budget_rows(sources: dict[str, Source], k: dict[str, int], fill: Sequence[str], caps: dict[str, float],
                fixed_mb: dict[str, float]) -> dict[str, dict[str, Any]]:
    total = sum(sources[s].file_bytes(n) for s, n in k.items())
    rows = {}
    for s, src in sources.items():
        n = k[s]
        rule = "mb" if s in fixed_mb else "fill" if s in fill else "full"
        rows[s] = {
            "rule": rule,
            "rate": n / BUCKETS,
            "keep_below": n * PER_BUCKET,
            "cap": caps.get(s),
            "genres": sorted(src.scan["genres"]),
            "available_docs": src.scan["docs"],
            "available_text_bytes": src.scan["text_bytes"],
            "docs": int(src.cum_docs[n]),
            "text_bytes": int(src.cum_text[n]),
            "file_bytes": src.file_bytes(n),
            "share": src.file_bytes(n) / total if total else 0.0,
            "punct_stripped_share": src.punct_stripped_share,
        }
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def render(rows: list[tuple[str, ...]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for i, r in enumerate(rows):
        out.append("  ".join([r[0].ljust(widths[0]), r[1].ljust(widths[1])] +
                             [r[j].rjust(widths[j]) for j in range(2, len(r))]))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def genre_table(scan: dict[str, Any]) -> str:
    total = sum(row["text_bytes"] for row in scan["sources"].values())
    rows = [("source", "genre", "docs", "MB", "MiB", "share")]
    docs = 0
    for s, row in scan["sources"].items():
        for g, c in row["genres"].items():
            rows.append((s, g, f"{c['docs']:,}", f"{c['text_bytes'] / 1e6:,.2f}", f"{c['text_bytes'] / 2**20:,.2f}",
                         f"{c['text_bytes'] / total:.2%}"))
        docs += row["docs"]
    rows.append(("total", "", f"{docs:,}", f"{total / 1e6:,.2f}", f"{total / 2**20:,.2f}", "100.00%"))
    return render(rows)


def budget_table(rows: dict[str, dict[str, Any]]) -> str:
    out = [("source", "rule", "available MB", "rate", "docs", "text MB", "file MB", "share", "cap")]
    tot: Counter[str] = Counter()
    for s, r in rows.items():
        out.append((s, r["rule"], f"{r['available_text_bytes'] / 1e6:,.2f}", f"{r['rate']:.4f}", f"{r['docs']:,}",
                    f"{r['text_bytes'] / 1e6:,.2f}", f"{r['file_bytes'] / 1e6:,.2f}", f"{r['share']:.2%}",
                    f"{r['cap']:.0%}" if r["cap"] is not None else ""))
        tot.update(available=r["available_text_bytes"], docs=r["docs"], text=r["text_bytes"], file=r["file_bytes"])
    out.append(("total", "", f"{tot['available'] / 1e6:,.2f}", "", f"{tot['docs']:,}", f"{tot['text'] / 1e6:,.2f}",
                f"{tot['file'] / 1e6:,.2f}", "100.00%", ""))
    return render(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def source_value(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected SOURCE=VALUE, got {text!r}")
    return name, float(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--total-mb", type=float, default=DEFAULT_TOTAL_MB, help="largest total, in file MB")
    p.add_argument("--min-total-mb", type=float, default=DEFAULT_MIN_TOTAL_MB, help="smallest total, in file MB")
    p.add_argument("--fill", nargs="*", default=list(DEFAULT_FILL), metavar="SOURCE",
                   help="sources that fill the space left, in order (default: %(default)s); all others are full")
    p.add_argument("--cap", type=source_value, action="append", default=[], metavar="SOURCE=SHARE",
                   help="largest share of the final bytes for a source (overrides the default caps)")
    p.add_argument("--mb", type=source_value, action="append", default=[], metavar="SOURCE=MB",
                   help="keep at most this many file MB of a source")
    p.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    p.add_argument("--scan", type=Path, default=SCAN_JSON, help="scan cache (default data/stats/budget_scan.json)")
    p.add_argument("--rescan", action="store_true", help="ignore the scan cache")
    p.add_argument("--jobs", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1))
    p.add_argument("--out", type=Path, default=BUDGET_JSON)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("budget")
    scan = load_scan(args.clean_dir, args.scan, args.rescan, args.jobs, log)
    sources = {s: Source(s, row) for s, row in scan["sources"].items()}
    print("Cleaned text per source and genre (data/clean, after Step 5b):")
    print(genre_table(scan))
    for s, src in sources.items():
        if src.scan["punct_stripped_docs"]:
            print(f"{s}: meta.punct_stripped on {src.scan['punct_stripped_docs']:,} docs, "
                  f"{src.punct_stripped_share:.2%} of its bytes")

    named = {s for s, _ in args.cap + args.mb} | (set(args.fill) if args.fill != list(DEFAULT_FILL) else set())
    if unknown := sorted(named - set(sources)):
        raise SystemExit(f"no cleaned file for: {', '.join(unknown)}")
    caps = {s: c for s, c in (default_caps(sources) | dict(args.cap)).items() if s in sources}
    fixed_mb = dict(args.mb)
    fill = [s for s in args.fill if s in sources and s not in fixed_mb]
    try:
        k = solve(sources, fill, caps, fixed_mb, args.total_mb * 1e6, args.min_total_mb * 1e6)
    except Infeasible as e:
        print(f"\nNo budget written: {e}.")
        print("Options: raise --total-mb, move a full source into --fill (it is then subsampled into the space "
              "left), or keep part of it with --mb SOURCE=MB.")
        return 2
    rows = budget_rows(sources, k, fill, caps, fixed_mb)
    print(f"\nBudget (file MB = text + {EOT_BYTES} bytes of {ENDOFTEXT} per document; shares of file bytes):")
    print(budget_table(rows))
    budget = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(["python", "clean/budget.py", *(shlex.quote(a) for a in (
            sys.argv[1:] if argv is None else argv))]),
        "keep_rule": "keep a document when int(blake2b(doc_id.encode('utf-8')).hexdigest(), 16) % 10**6 < keep_below",
        "policy": {"total_mb_max": args.total_mb, "total_mb_min": args.min_total_mb, "fill": fill, "caps": caps,
                   "mb": fixed_mb, "punct_stripped_cap": PUNCT_STRIPPED_CAP},
        "inputs": scan["files"],
        "sources": rows,
        "totals": {"docs": sum(r["docs"] for r in rows.values()),
                   "text_bytes": sum(r["text_bytes"] for r in rows.values()),
                   "file_bytes": sum(r["file_bytes"] for r in rows.values())},
    }
    write_json(args.out, budget)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
