"""Extract FineWeb-2 urd_Arab parquet files (Step 3).

    python clean/extract_fineweb2.py --limit 2000
    python clean/extract_fineweb2.py --keep-per-mille 250

Reads every finished data/raw/fineweb2_urd/*.parquet in name order with
pyarrow.parquet.ParquetFile(...).iter_batches over the columns id, text, url,
date, language_score (plus dump and minhash_cluster_size when present, kept in
meta). Unfinished .part files are skipped: a parquet footer is at the end.

--keep-per-mille K keeps a document only if int(sha1(id), 16) % 1000 < K: a
deterministic subsample, tuned in Step 6 to hit the size budget. Documents
left out are counted as drop_not_sampled.
"""

import argparse
import hashlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import Sink, add_common_args, input_files, make_doc, write_docs  # noqa: E402
from common import setup_logging  # noqa: E402

SOURCE = "fineweb2_urd"
GENRE = "web"
COLUMNS = ("id", "text", "url", "date", "language_score")
META_COLUMNS = ("language_score", "dump", "minhash_cluster_size")


def kept(doc_id: str, per_mille: int) -> bool:
    return per_mille >= 1000 or int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % 1000 < per_mille


def iter_docs(path: Path, sink: Sink, per_mille: int, batch_size: int = 1024) -> Iterator[dict[str, Any]]:
    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    missing = [c for c in ("id", "text") if c not in names]
    if missing:
        raise SystemExit(f"{path.name}: missing column(s) {missing}")
    columns = [c for c in dict.fromkeys(COLUMNS + META_COLUMNS) if c in names]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        for row in batch.to_pylist():
            text = row.get("text") or ""
            sink.read(raw_bytes=len(text.encode("utf-8", "surrogatepass")))
            doc_id = str(row["id"])
            if not kept(doc_id, per_mille):
                sink.drop("not_sampled")
                continue
            meta = {c: row[c] for c in META_COLUMNS if c in row}
            meta["file"] = path.name
            date = row.get("date")
            yield make_doc(SOURCE, doc_id, GENRE, text, url=row.get("url"),
                           date=str(date) if date is not None else None, meta=meta)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    p.add_argument("--keep-per-mille", type=int, default=1000, metavar="K",
                   help="keep documents with int(sha1(id), 16) %% 1000 < K (default 1000 = all)")
    args = p.parse_args(argv)
    if not 0 <= args.keep_per_mille <= 1000:
        p.error("--keep-per-mille must be between 0 and 1000")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_fineweb2")
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)
    sink.extra["keep_per_mille"] = args.keep_per_mille

    def docs() -> Iterator[dict[str, Any]]:
        for path in input_files(args.raw_dir, ".parquet", log):
            log.info("reading %s", path)
            yield from iter_docs(path, sink, args.keep_per_mille)

    write_docs(args, sink, docs(), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
