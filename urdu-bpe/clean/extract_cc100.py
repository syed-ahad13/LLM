"""Extract CC-100 Urdu (Step 3; fallback source, disabled in sources.json).

    python clean/extract_cc100.py --limit 2000

Streams data/raw/cc100_ur/ur.txt.xz through lzma. The file has one paragraph
per line and a blank line between documents, so a blank line ends a document
and its paragraphs are joined with "\\n". A document holding any line that is
not valid UTF-8 is dropped and counted.
"""

import argparse
import lzma
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import Sink, add_common_args, decode_strict, input_files, make_doc, write_docs  # noqa: E402
from common import setup_logging  # noqa: E402

SOURCE = "cc100_ur"
GENRE = "web"


def iter_docs(path: Path, sink: Sink) -> Iterator[dict[str, Any]]:
    paragraphs: list[str] = []
    raw_bytes = 0
    bad = False
    ordinal = 0

    def finish() -> Iterator[dict[str, Any]]:
        nonlocal ordinal
        sink.read(raw_bytes=raw_bytes)
        ordinal += 1
        if bad:
            sink.drop("utf8_error")
        else:
            yield make_doc(SOURCE, str(ordinal), GENRE, "\n".join(paragraphs), meta={"ordinal": ordinal})

    with lzma.open(path, "rb") as f:
        for raw in f:
            raw = raw.rstrip(b"\r\n")
            if not raw.strip():
                if paragraphs or bad:
                    yield from finish()
                paragraphs, raw_bytes, bad = [], 0, False
                continue
            raw_bytes += len(raw)
            text = decode_strict(raw)
            if text is None:
                bad = True
            else:
                paragraphs.append(text)
    if paragraphs or bad:
        yield from finish()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_cc100")
    [path] = input_files(args.raw_dir, ".xz", log)[-1:]
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)
    write_docs(args, sink, iter_docs(path, sink), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
