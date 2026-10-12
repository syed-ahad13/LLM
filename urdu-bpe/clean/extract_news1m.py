"""Extract the Urdu News Dataset 1M (Step 3).

    python clean/extract_news1m.py --limit 2000

Streams data/raw/urdu_news_1m/*.csv with the stdlib csv module (field size
limit raised). Columns: Index, Headline, News Text, Category, Date, URL,
Source, News length. text = headline + "\\n" + story. Category, source outlet
and the dataset's own length field go to meta; date and url to their fields.

The file is read with errors="surrogateescape", so a row with bytes that are
not UTF-8 survives parsing and is then dropped by the Sink as utf8_error.

meta.punct_stripped: the dataset's methods text says preprocessing kept "Urdu
text and numbers only", which the Step 1 check (collect/inspect_raw.py) tests
for. A story is flagged when it holds none of the sentence punctuation in
SENTENCE_PUNCT (Urdu or ASCII). Step 6 can cap or drop flagged documents.
"""

import argparse
import csv
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import Sink, add_common_args, input_files, make_doc, sha1_hex, write_docs  # noqa: E402
from common import setup_logging  # noqa: E402

SOURCE = "urdu_news_1m"
GENRE = "news"
SENTENCE_PUNCT = frozenset("۔،؟.,?!")  # ۔ ، ؟ and their ASCII counterparts


def punct_stripped(story: str) -> bool:
    return not any(c in SENTENCE_PUNCT for c in story)


def _field(row: dict[str, str], name: str) -> str:
    value = row.get(name)
    if value is None:  # tolerate case or spacing differences in the header
        value = next((v for k, v in row.items() if k and k.strip().lower() == name.lower()), "")
    return value or ""


def iter_docs(path: Path, sink: Sink) -> Iterator[dict[str, Any]]:
    csv.field_size_limit(2**31 - 1)
    with open(path, encoding="utf-8-sig", errors="surrogateescape", newline="") as f:
        for row in csv.DictReader(f):
            headline = _field(row, "Headline").strip()
            story = _field(row, "News Text").strip()
            sink.read(raw_bytes=len((headline + story).encode("utf-8", "surrogateescape")))
            if not story:
                sink.drop("empty_story")
                continue
            flagged = punct_stripped(story)
            sink.counts["punct_stripped_docs"] += flagged
            url = _field(row, "URL").strip()
            native = _field(row, "Index").strip() or sha1_hex(url or headline + story)
            meta = {"category": _field(row, "Category").strip() or None,
                    "outlet": _field(row, "Source").strip() or None,
                    "news_length": _field(row, "News length").strip() or None,
                    "punct_stripped": flagged}
            yield make_doc(SOURCE, native, GENRE, f"{headline}\n{story}" if headline else story,
                           url=url, title=headline, date=_field(row, "Date").strip(), meta=meta)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_news1m")
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)

    def docs() -> Iterator[dict[str, Any]]:
        for path in input_files(args.raw_dir, ".csv", log):
            log.info("reading %s", path)
            yield from iter_docs(path, sink)

    write_docs(args, sink, docs(), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
