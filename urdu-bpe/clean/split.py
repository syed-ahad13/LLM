"""Step 6b: split the budgeted documents into train/val/test (98/1/1, stratified by source) and write the corpus.

    python clean/split.py                          # data/final/budget.json + data/clean -> data/final/*.txt
    python clean/split.py --out-dir /tmp/rerun     # the same split elsewhere (tests/test_split.py compares sha256)

Deterministic: the same budget.json and data/clean files give byte-identical files.

Pass 1 (one process per source) reads every doc_id and keeps the documents the
budget selects: clean.budget.keep_key(doc_id) < keep_below. Then, per source:
sort the kept doc_ids, shuffle them with random.Random(f"1337:{source}"), and
cut them by count into train, val and test: val = test = max(1, round(n/100))
(half up), train = the rest, in that order. Each split's documents from all
sources are then shuffled together with random.Random(f"1337:order:{split}"),
so the files are not grouped by source; a document's index in that list is its
position in the file. Only doc_ids are held in memory.

Pass 2 (one process per source) streams the documents again and appends each
kept one, with its position, to one of BUCKETS temporary files per split:
bucket = position * BUCKETS // documents in the split, so each bucket holds a
contiguous range of positions. It fails on any text containing <|endoftext|>
(Step 5b removed them).

Pass 3 loads one bucket at a time, sorts it by position and appends each text
followed by <|endoftext|> (no other separator) to the split file. The doc_ids go
to <split>.ids, one per line in the same order, so the tests can check that no
document is in two splits. Positions must come out as 0, 1, 2, ... exactly.

Writes data/final/{train,val,test}.txt, {train,val,test}.ids and manifest.json:
seed, budget, documents and bytes per split and source (and genre), the most
common URL domains per source, and the sha256 of every file.
"""

import argparse
import gzip
import hashlib
import json
import multiprocessing
import os
import random
import re
import resource
import shutil
import struct
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.budget import BUDGET_JSON, FINAL_DIR, HASH_MOD, file_signature, keep_key  # noqa: E402
from clean.dedup import CLEAN_DIR, ENDOFTEXT  # noqa: E402
from common import machine_info, setup_logging, sha256_file, write_json  # noqa: E402

SEED = 1337
SPLITS = ("train", "val", "test")
BUCKETS = 64
EOT = ENDOFTEXT.encode("utf-8")
READ_BLOCK = 8 * 1024 * 1024
POS_BITS = 40  # assignment code = split index << POS_BITS | position
RECORD = struct.Struct("<QHI")  # position, doc_id bytes, text bytes
# Step 3-5b write each document with json.dumps and doc_id as the first key.
DOC_ID_RE = re.compile(rb'^\{"doc_id": "((?:[^"\\]|\\.)*)"')
TOP_DOMAINS = 15


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def iter_lines(path: Path) -> Iterator[bytes]:
    """The non-blank lines of a .jsonl.gz file, as bytes."""
    with gzip.open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield line


def line_doc_id(line: bytes) -> str:
    """doc_id of a JSON line without parsing the text (falls back to json.loads)."""
    m = DOC_ID_RE.match(line)
    if m is None:
        return json.loads(line)["doc_id"]
    raw = m.group(1)
    return json.loads(b'"' + raw + b'"') if b"\\" in raw else raw.decode("utf-8")


def read_docs(path: Path, start: int = 0, end: int | None = None, block: int = READ_BLOCK) -> Iterator[bytes]:
    """The documents of a split file between byte offsets start and end, as bytes without the separator.

    start and end must be document boundaries (0, EOF, or just after a separator). Fails if text follows the
    last separator.
    """
    with open(path, "rb") as f:
        f.seek(start)
        remaining = (os.fstat(f.fileno()).st_size if end is None else end) - start
        buf = b""
        while remaining > 0:
            chunk = f.read(min(block, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            buf += chunk
            parts = buf.split(EOT)
            buf = parts.pop()
            yield from parts
    if buf:
        raise ValueError(f"{path}: {len(buf)} bytes after the last {ENDOFTEXT} before offset {end}")


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def cut_sizes(n: int) -> tuple[int, int, int]:
    """(train, val, test) document counts for a source of n documents: 98/1/1, at least 1 each in val and test."""
    if n < 3:
        raise ValueError(f"a source needs at least 3 documents to give val and test one each, got {n}")
    k = max(1, (n + 50) // 100)
    return n - 2 * k, k, k


def split_source(source: str, doc_ids: Sequence[str], seed: int = SEED) -> dict[str, list[str]]:
    """Sort, shuffle with random.Random(f"{seed}:{source}"), then cut train/val/test by count."""
    ids = sorted(doc_ids)
    random.Random(f"{seed}:{source}").shuffle(ids)
    n_train, n_val, _ = cut_sizes(len(ids))
    return {"train": ids[:n_train], "val": ids[n_train:n_train + n_val], "test": ids[n_train + n_val:]}


def order_split(split: str, per_source: dict[str, list[str]], seed: int = SEED) -> list[str]:
    """One split's doc_ids from every source, shuffled together with random.Random(f"{seed}:order:{split}")."""
    pool = [doc_id for source in sorted(per_source) for doc_id in per_source[source]]
    random.Random(f"{seed}:order:{split}").shuffle(pool)
    return pool


def _select(item: tuple[str, str, int]) -> tuple[str, list[str], int, float]:
    """Pass 1 for one source: (source, kept doc_ids in file order, documents read, seconds)."""
    source, path, keep_below = item
    start = time.perf_counter()
    kept: list[str] = []
    n = 0
    prefix = f"{source}:"
    for line in iter_lines(Path(path)):
        n += 1
        doc_id = line_doc_id(line)
        if not doc_id.startswith(prefix):
            raise ValueError(f"{source}: doc_id {doc_id!r} does not start with {prefix!r}")
        if keep_below >= HASH_MOD or keep_key(doc_id) < keep_below:
            kept.append(doc_id)
    if len(set(kept)) != len(kept):
        dup = next(d for d, c in Counter(kept).items() if c > 1)
        raise ValueError(f"{source}: doc_id {dup!r} appears more than once")
    return source, kept, n, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Routing and assembly
# ---------------------------------------------------------------------------

def raise_open_file_limit(want: int = 4096) -> None:
    """Each pass-2 worker keeps up to 3 x BUCKETS files open; macOS shells often start at 256."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = want if hard == resource.RLIM_INFINITY else min(want, hard)
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def bucket_path(tmp_dir: Path, split: str, bucket: int, source: str) -> Path:
    return tmp_dir / f"{split}-{bucket:03d}-{source}.bin"


def domain(url: str | None) -> str:
    """The host's last label (pk, com, ...), or '-' when the document has no URL."""
    if not url:
        return "-"
    host = urlsplit(url).hostname or ""
    return host.rsplit(".", 1)[-1] if "." in host else (host or "-")


def _route(item: tuple[str, str, dict[str, int], list[int], str, int]) -> tuple[str, dict[str, Any], float]:
    """Pass 2 for one source: append every assigned document to its bucket file. Returns per-split counts."""
    source, path, assign, split_sizes, tmp_dir, buckets = item
    start = time.perf_counter()
    raise_open_file_limit()
    files: dict[tuple[int, int], Any] = {}
    counts: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    genres: dict[str, Counter[str]] = {s: Counter() for s in SPLITS}
    domains: Counter[str] = Counter()
    found = 0
    try:
        for line in iter_lines(Path(path)):
            code = assign.get(line_doc_id(line))
            if code is None:
                continue
            doc = json.loads(line)
            text: str = doc["text"]
            if ENDOFTEXT in text:
                raise ValueError(f"{doc['doc_id']}: text contains {ENDOFTEXT}")
            split_idx, pos = code >> POS_BITS, code & ((1 << POS_BITS) - 1)
            split = SPLITS[split_idx]
            bucket = pos * buckets // split_sizes[split_idx]
            key = (split_idx, bucket)
            if key not in files:
                files[key] = open(bucket_path(Path(tmp_dir), split, bucket, source), "wb")
            data = text.encode("utf-8")
            doc_id = doc["doc_id"].encode("utf-8")
            files[key].write(RECORD.pack(pos, len(doc_id), len(data)) + doc_id + data)
            counts[split].update(docs=1, text_bytes=len(data))
            genres[split][doc.get("genre") or "unknown"] += len(data)
            domains[domain(doc.get("url"))] += 1
            found += 1
    finally:
        for f in files.values():
            f.close()
    if found != len(assign):
        raise ValueError(f"{source}: {len(assign) - found} assigned documents were not found in pass 2")
    result = {split: {"docs": counts[split]["docs"], "text_bytes": counts[split]["text_bytes"],
                      "bytes": counts[split]["text_bytes"] + len(EOT) * counts[split]["docs"],
                      "genres": dict(sorted(genres[split].items()))} for split in SPLITS}
    result["domains"] = dict(domains.most_common(TOP_DOMAINS))  # type: ignore[assignment]
    return source, result, time.perf_counter() - start


def _replace_on_success(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def assemble(split: str, n_docs: int, sources: Sequence[str], tmp_dir: Path, out_dir: Path,
             buckets: int) -> dict[str, Any]:
    """Pass 3 for one split: write <split>.txt and <split>.ids from the bucket files, in position order."""
    txt_path, ids_path = out_dir / f"{split}.txt", out_dir / f"{split}.ids"
    txt_tmp, ids_tmp = _replace_on_success(txt_path), _replace_on_success(ids_path)
    h_txt, h_ids = hashlib.sha256(), hashlib.sha256()
    expected = size = 0
    try:
        with open(txt_tmp, "wb") as out, open(ids_tmp, "wb") as ids:
            for b in range(buckets):
                records: list[tuple[int, bytes, bytes]] = []
                for source in sources:
                    part = bucket_path(tmp_dir, split, b, source)
                    if not part.exists():
                        continue
                    buf = part.read_bytes()
                    off = 0
                    while off < len(buf):
                        pos, n_id, n_text = RECORD.unpack_from(buf, off)
                        off += RECORD.size
                        doc_id = buf[off:off + n_id]
                        off += n_id
                        records.append((pos, doc_id, buf[off:off + n_text]))
                        off += n_text
                    del buf
                records.sort(key=lambda r: r[0])
                for pos, doc_id, text in records:
                    if pos != expected:
                        raise ValueError(f"{split}: expected position {expected}, found {pos}")
                    expected += 1
                    out.write(text)
                    out.write(EOT)
                    h_txt.update(text)
                    h_txt.update(EOT)
                    ids.write(doc_id + b"\n")
                    h_ids.update(doc_id + b"\n")
                    size += len(text) + len(EOT)
        if expected != n_docs:
            raise ValueError(f"{split}: wrote {expected} documents, expected {n_docs}")
        os.replace(txt_tmp, txt_path)
        os.replace(ids_tmp, ids_path)
    finally:
        txt_tmp.unlink(missing_ok=True)
        ids_tmp.unlink(missing_ok=True)
    return {"file": txt_path.name, "ids_file": ids_path.name, "docs": n_docs, "bytes": size,
            "sha256": h_txt.hexdigest(), "ids_sha256": h_ids.hexdigest()}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def load_budget(path: Path, clean_dir: Path) -> dict[str, Any]:
    """budget.json, checked against the current data/clean files."""
    if not path.exists():
        raise SystemExit(f"no {path}; run clean/budget.py and approve the budget first")
    with open(path, encoding="utf-8") as f:
        budget = json.load(f)
    present = {p.name.removesuffix(".jsonl.gz") for p in clean_dir.glob("*.jsonl.gz")}
    if extra := sorted(present - set(budget["sources"])):
        raise SystemExit(f"{path} has no entry for {', '.join(extra)}; rerun clean/budget.py")
    for source in budget["sources"]:
        clean = clean_dir / f"{source}.jsonl.gz"
        if not clean.exists():
            raise SystemExit(f"missing {clean}")
        if file_signature(clean) != budget["inputs"][source]:
            raise SystemExit(f"{clean} changed after {path} was written; rerun clean/budget.py")
    return budget


def run(budget_path: Path, clean_dir: Path, out_dir: Path, buckets: int, jobs: int, seed: int,
        log: Any) -> dict[str, Any]:
    budget = load_budget(budget_path, clean_dir)
    sources = list(budget["sources"])
    paths = {s: str(clean_dir / f"{s}.jsonl.gz") for s in sources}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".split_tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir()
    seconds: dict[str, float] = {}
    ctx = multiprocessing.get_context("spawn")
    try:
        # Pass 1: which documents, and where each one goes.
        start = time.perf_counter()
        kept: dict[str, list[str]] = {s: [] for s in sources}
        read: dict[str, int] = {s: budget["sources"][s]["available_docs"] for s in sources}
        items = [(s, paths[s], budget["sources"][s]["keep_below"]) for s in sources
                 if budget["sources"][s]["keep_below"] > 0]
        with ctx.Pool(max(1, min(jobs, len(items)))) as pool:
            for source, ids, n, secs in pool.imap_unordered(_select, items):
                log.info("pass 1: %s: %d of %d docs kept (%.0f s)", source, len(ids), n, secs)
                want = budget["sources"][source]["docs"]
                if len(ids) != want:
                    raise ValueError(f"{source}: kept {len(ids)} documents, budget.json says {want}")
                kept[source], read[source] = ids, n
        # A source the budget gives no documents is left out; split_source refuses 1 or 2.
        active = [s for s in sources if kept[s]]
        per_split: dict[str, dict[str, list[str]]] = {split: {} for split in SPLITS}
        for source in active:
            for split, ids in split_source(source, kept[source], seed).items():
                per_split[split][source] = ids
        del kept
        assign: dict[str, dict[str, int]] = {s: {} for s in active}
        split_sizes = []
        for split_idx, split in enumerate(SPLITS):
            order = order_split(split, per_split[split], seed)
            split_sizes.append(len(order))
            for pos, doc_id in enumerate(order):  # pass 1 checked that every doc_id starts with "<source>:"
                assign[doc_id.partition(":")[0]][doc_id] = split_idx << POS_BITS | pos
        seconds["pass1"] = time.perf_counter() - start

        # Pass 2: route the documents into bucket files.
        start = time.perf_counter()
        empty = {"docs": 0, "text_bytes": 0, "bytes": 0, "genres": {}}
        routed: dict[str, dict[str, Any]] = {s: {**{sp: empty for sp in SPLITS}, "domains": {}} for s in sources}
        with ctx.Pool(max(1, min(jobs, len(active)))) as pool:
            items2 = [(s, paths[s], assign[s], split_sizes, str(tmp_dir), buckets) for s in active]
            for source, result, secs in pool.imap_unordered(_route, items2):
                log.info("pass 2: %s routed (%.0f s)", source, secs)
                routed[source] = result
        del assign
        seconds["pass2"] = time.perf_counter() - start

        # Pass 3: one bucket at a time, in position order.
        start = time.perf_counter()
        splits: dict[str, Any] = {}
        for split_idx, split in enumerate(SPLITS):
            row = assemble(split, split_sizes[split_idx], sources, tmp_dir, out_dir, buckets)
            row["text_bytes"] = sum(routed[s][split]["text_bytes"] for s in sources)
            row["sources"] = {s: {k: routed[s][split][k] for k in ("docs", "text_bytes", "bytes")} for s in sources}
            genre_bytes: Counter[str] = Counter()
            for s in sources:
                genre_bytes.update(routed[s][split]["genres"])
            row["genres"] = dict(sorted(genre_bytes.items()))
            if row["bytes"] != row["text_bytes"] + len(EOT) * row["docs"]:
                raise ValueError(f"{split}: byte counts do not add up")
            splits[split] = row
            log.info("pass 3: %s.txt: %d docs, %.2f MB, sha256 %s", split, row["docs"], row["bytes"] / 1e6,
                     row["sha256"])
        seconds["pass3"] = time.perf_counter() - start
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "split_rule": "per source: sort doc_ids, shuffle with random.Random(f'{seed}:{source}'), then take "
                      "train = n - 2k, val = k, test = k in that order, k = max(1, round(n / 100)) half up",
        "order_rule": "each split's documents from all sources shuffled with random.Random(f'{seed}:order:{split}')",
        "separator": ENDOFTEXT,
        "buckets": buckets,
        "budget": {"path": str(budget_path), "sha256": sha256_file(budget_path), "command": budget.get("command"),
                   "policy": budget["policy"],
                   "sources": {s: {k: budget["sources"][s][k] for k in ("rule", "rate", "keep_below")}
                               for s in sources}},
        "sources": {s: {"available_docs": read[s], "docs": sum(routed[s][sp]["docs"] for sp in SPLITS),
                        **{sp: routed[s][sp]["docs"] for sp in SPLITS},
                        "text_bytes": sum(routed[s][sp]["text_bytes"] for sp in SPLITS),
                        "bytes": sum(routed[s][sp]["bytes"] for sp in SPLITS),
                        "url_domains": routed[s]["domains"]} for s in sources},
        "splits": splits,
        "totals": {k: sum(splits[sp][k] for sp in SPLITS) for k in ("docs", "text_bytes", "bytes")},
        "seconds": {k: round(v, 1) for k, v in seconds.items()},
        "machine": machine_info(),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=Path, default=BUDGET_JSON, help="default data/final/budget.json")
    p.add_argument("--clean-dir", type=Path, default=CLEAN_DIR)
    p.add_argument("--out-dir", type=Path, default=FINAL_DIR, help="where the split files go (default data/final)")
    p.add_argument("--buckets", type=int, default=BUCKETS, help="temporary bucket files per split")
    p.add_argument("--jobs", type=int, default=max(1, min(6, (multiprocessing.cpu_count() or 2) - 1)))
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("split")
    manifest = run(args.budget, args.clean_dir, args.out_dir, args.buckets, args.jobs, args.seed, log)
    rows = [("split", "docs", "MB", "MiB", "sha256")]
    for split, r in manifest["splits"].items():
        rows.append((split, f"{r['docs']:,}", f"{r['bytes'] / 1e6:,.2f}", f"{r['bytes'] / 2**20:,.2f}", r["sha256"]))
    t = manifest["totals"]
    rows.append(("total", f"{t['docs']:,}", f"{t['bytes'] / 1e6:,.2f}", f"{t['bytes'] / 2**20:,.2f}", ""))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, r in enumerate(rows):
        print("  ".join([r[0].ljust(widths[0])] + [r[j].rjust(widths[j]) for j in range(1, 4)] + [r[4]]))
        if i == 0:
            print("  ".join("-" * w for w in widths))
    print(f"wrote {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
