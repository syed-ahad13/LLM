"""Shared plumbing for the Step 3 extractors (clean/extract_<source>.py).

Every extractor turns one raw container into data/interim/<source>.jsonl.gz,
one document per line, with the same fields everywhere:

    {"doc_id": "<source>:<native id or sha1>", "source": str, "genre": str,
     "url": str|null, "title": str|null, "date": str|null, "text": str, "meta": {...}}

`text` keeps paragraph breaks as "\\n". Nothing is normalized or filtered for
language here (CLAUDE.md, Corpus spec), so later steps can measure exactly
what each cleaning step removes.

An extractor is a generator of documents that reports what it read and dropped
to a Sink:

    sink = Sink("ufal_umc", limit=args.limit)
    write_docs(args, sink, iter_docs(path, sink))

Sink.accept() drops any document whose strings are not valid UTF-8 (lone
surrogates from errors="surrogateescape") or whose text is blank, stops after
--limit documents, counts docs/lines/bytes written, and keeps a deterministic
sample of about 2 MB: the documents with the smallest blake2b(doc_id), written
in corpus order to data/samples/<source>.sample.jsonl. Stats go to
data/stats/extract.json through common.Stats.
"""

import argparse
import hashlib
import heapq
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import STATS_DIR, Stats, write_jsonl  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
SAMPLES_DIR = ROOT / "data" / "samples"
SOURCES_JSON = ROOT / "collect" / "sources.json"

STAGE = "extract"
SAMPLE_BYTES = 2_000_000


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "surrogatepass")).hexdigest()


def decode_strict(raw: bytes) -> str | None:
    """UTF-8 decode, or None if the bytes are not valid UTF-8."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def is_valid_utf8(s: str) -> bool:
    """False for strings holding lone surrogates (what errors='surrogateescape' makes of bad bytes)."""
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def make_doc(source: str, native_id: str, genre: str, text: str, *, url: str | None = None,
             title: str | None = None, date: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"doc_id": f"{source}:{native_id}", "source": source, "genre": genre, "url": url or None,
            "title": title or None, "date": date or None, "text": text, "meta": meta or {}}


def _strings(obj: Any) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


class Sink:
    """Counts, limit and sample for one extractor run. See the module docstring."""

    def __init__(self, source: str, *, limit: int | None = None, sample_bytes: int = SAMPLE_BYTES) -> None:
        self.source = source
        self.limit = limit
        self.sample_bytes = sample_bytes
        self.counts: Counter[str] = Counter()
        self.drops: Counter[str] = Counter()
        self.extra: dict[str, Any] = {}
        self._sample: list[tuple[int, int, str]] = []  # max-heap by hash: (-hash, seq, json line)
        self._sample_size = 0
        self._seq = 0

    @property
    def full(self) -> bool:
        return self.limit is not None and self.counts["docs_written"] >= self.limit

    def read(self, n: int = 1, raw_bytes: int = 0) -> None:
        """Record n raw records read, holding raw_bytes of UTF-8 text."""
        self.counts["records_read"] += n
        self.counts["raw_text_bytes"] += raw_bytes

    def drop(self, reason: str, n: int = 1) -> None:
        self.drops[reason] += n

    def accept(self, docs: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Pass valid documents through, counting them; stop at the limit."""
        if self.full:
            return
        for doc in docs:
            if not all(is_valid_utf8(s) for s in _strings(doc)):
                self.drop("utf8_error")
                continue
            if not doc["text"].strip():
                self.drop("empty_text")
                continue
            self._count(doc)
            yield doc
            if self.full:
                return

    def _count(self, doc: dict[str, Any]) -> None:
        text = doc["text"]
        self.counts["docs_written"] += 1
        self.counts["lines_written"] += text.count("\n") + 1
        self.counts["bytes_written"] += len(text.encode("utf-8"))
        self._offer_sample(doc)

    def _offer_sample(self, doc: dict[str, Any]) -> None:
        h = int.from_bytes(hashlib.blake2b(doc["doc_id"].encode("utf-8"), digest_size=8).digest(), "big")
        heap = self._sample
        if heap and self._sample_size >= self.sample_bytes and h >= -heap[0][0]:
            return
        line = json.dumps(doc, ensure_ascii=False)
        heapq.heappush(heap, (-h, self._seq, line))
        self._seq += 1
        self._sample_size += len(line.encode("utf-8")) + 1
        while self._sample_size > self.sample_bytes and len(heap) > 1:
            _, _, dropped = heapq.heappop(heap)
            self._sample_size -= len(dropped.encode("utf-8")) + 1

    def sample_lines(self) -> list[str]:
        """The sampled documents as JSON lines, in the order they were written."""
        return [line for _, _, line in sorted(self._sample, key=lambda e: e[1])]

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.counts)
        for key in ("records_read", "docs_written", "lines_written", "bytes_written", "raw_text_bytes"):
            out.setdefault(key, 0)
        out.update({f"drop_{reason}": n for reason, n in self.drops.items()})
        out["limit"] = self.limit or 0
        out.update(self.extra)
        return out


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser, source: str) -> None:
    p.add_argument("--limit", type=int, default=None, metavar="N", help="stop after N documents written (smoke runs)")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR / source, help=f"input folder (default data/raw/{source})")
    p.add_argument("--out", type=Path, default=INTERIM_DIR / f"{source}.jsonl.gz",
                   help=f"output JSONL (default data/interim/{source}.jsonl.gz)")
    p.add_argument("--sample-out", type=Path, default=SAMPLES_DIR / f"{source}.sample.jsonl",
                   help=f"2 MB sample (default data/samples/{source}.sample.jsonl)")
    p.add_argument("--sample-bytes", type=int, default=SAMPLE_BYTES)
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR, help="where extract.json goes (default data/stats)")


def write_docs(args: argparse.Namespace, sink: Sink, docs: Iterable[dict[str, Any]], log: logging.Logger) -> dict[str, Any]:
    """Write accepted documents, the sample and the stats. Returns the stats row."""
    write_jsonl(args.out, sink.accept(docs))
    if isinstance(docs, Iterator) and hasattr(docs, "close"):
        docs.close()  # stop a generator early at --limit, so its files and pools are released
    args.sample_out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.sample_out, (json.loads(line) for line in sink.sample_lines()))
    stats = Stats(args.stats_dir)
    row = sink.stats()
    stats.update(STAGE, sink.source, row)
    stats.dump()
    log.info("%s: %d records read, %d docs written (%.2f MB), drops %s -> %s",
             sink.source, row["records_read"], row["docs_written"], row["bytes_written"] / 1e6,
             dict(sink.drops) or "none", args.out)
    return row


def input_files(raw_dir: Path, suffix: str, log: logging.Logger) -> list[Path]:
    """Finished data files in raw_dir ending in suffix, sorted. Unfinished .part downloads are skipped with a warning."""
    if not raw_dir.is_dir():
        raise SystemExit(f"no folder {raw_dir}; download the source first (collect/download.py)")
    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.name.endswith(suffix))
    for p in sorted(raw_dir.glob(f"*{suffix}.part")):
        log.warning("skipping unfinished download %s", p.name)
    if not files:
        raise SystemExit(f"no *{suffix} file in {raw_dir}")
    return files


def own_scrape_sites(path: Path = SOURCES_JSON) -> dict[str, dict[str, Any]]:
    """site_id -> site entry of the own_scrape source in sources.json."""
    with open(path, encoding="utf-8") as f:
        sources = json.load(f)["sources"]
    entry = next((s for s in sources if s["id"] == "own_scrape"), {})
    return {s["site_id"]: s for s in entry.get("sites", [])}
