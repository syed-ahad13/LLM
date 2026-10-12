"""Extract the UFAL Urdu Monolingual Corpus (Step 3).

    python clean/extract_ufal.py --limit 2000
    python clean/extract_ufal.py --group auto            # default

Reads data/raw/ufal_umc/urdu-plain-text-corpus.gz line by line (one sentence
per line). Each line is decoded as strict UTF-8; a line that fails is dropped
and counted, and blank lines are skipped.

Grouping. The Step 1 order test (collect/inspect_raw.order_test) compares the
mean word-set Jaccard of consecutive lines with that of random line pairs over
the first --order-lines lines. With --group auto: if the ratio is at least
ORDER_RATIO (1.5), document order is preserved and every GROUP_LINES (32)
consecutive sentences become one pseudo-document; otherwise each line is its
own document. Both Jaccard numbers and the decision go to the stats.
--group N forces N lines per document.
"""

import argparse
import gzip
import logging
import sys
from collections.abc import Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import Sink, add_common_args, decode_strict, input_files, make_doc, write_docs  # noqa: E402
from collect.inspect_raw import order_test  # noqa: E402
from common import setup_logging  # noqa: E402

SOURCE = "ufal_umc"
GENRE = "mixed"
GROUP_LINES = 32
ORDER_RATIO = 1.5


def iter_lines(path: Path) -> Iterator[tuple[int, bytes]]:
    """(1-based line number, raw line without its line break)."""
    with gzip.open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            yield lineno, raw.rstrip(b"\r\n")


def choose_group(path: Path, order_lines: int, seed: int, log: logging.Logger) -> tuple[int, dict[str, Any]]:
    """Lines per document from the order test, and the numbers behind the choice."""
    head = [s for _, raw in islice(iter_lines(path), order_lines) if (s := decode_strict(raw)) is not None]
    consecutive, random_pairs = order_test(head, seed)
    ratio = consecutive / random_pairs if random_pairs else float("inf")
    group = GROUP_LINES if ratio >= ORDER_RATIO else 1
    log.info("order test over %d lines: consecutive Jaccard %.4f, random %.4f, ratio %.2f -> %d line(s) per document",
             len(head), consecutive, random_pairs, ratio, group)
    return group, {"order_test_lines": len(head), "order_jaccard_consecutive": round(consecutive, 6),
                   "order_jaccard_random": round(random_pairs, 6), "order_ratio": round(ratio, 4)}


def iter_docs(path: Path, sink: Sink, group: int) -> Iterator[dict[str, Any]]:
    buf: list[str] = []
    first = 0
    for lineno, raw in iter_lines(path):
        sink.read(raw_bytes=len(raw))
        text = decode_strict(raw)
        if text is None:
            sink.drop("utf8_error")
            continue
        if not text.strip():
            sink.counts["blank_lines"] += 1
            continue
        if not buf:
            first = lineno
        buf.append(text)
        if len(buf) == group:
            yield _doc(first, lineno, buf)
            buf = []
    if buf:
        yield _doc(first, first + len(buf) - 1, buf)


def _doc(first: int, last: int, lines: list[str]) -> dict[str, Any]:
    return make_doc(SOURCE, f"L{first}", GENRE, "\n".join(lines),
                    meta={"first_line": first, "last_line": last, "sentences": len(lines)})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    p.add_argument("--group", default="auto", help="'auto' (order test decides) or lines per document (default auto)")
    p.add_argument("--order-lines", type=int, default=20000, help="lines for the order test (default 20000)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_ufal")
    [path] = input_files(args.raw_dir, ".gz", log)[-1:]
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)
    if args.group == "auto":
        group, numbers = choose_group(path, args.order_lines, args.seed, log)
        sink.extra.update(numbers)
    else:
        group = int(args.group)
    sink.extra["group_lines"] = group
    write_docs(args, sink, iter_docs(path, sink, group), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
