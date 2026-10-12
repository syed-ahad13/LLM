"""Extract our own collection (Step 3): data/raw/own_scrape/<site>/articles.jsonl.

    python clean/extract_scrape.py --limit 2000

Sites are read in name order, each file in crawl order. Each line is decoded
as strict UTF-8 and parsed as JSON. Articles whose canonical URL (or URL, when
there is no canonical one) was already seen are dropped and counted.
text = title + "\\n" + body; genre and site name come from the own_scrape
entry in collect/sources.json.
"""

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import (  # noqa: E402
    Sink,
    add_common_args,
    decode_strict,
    make_doc,
    own_scrape_sites,
    sha1_hex,
    write_docs,
)
from common import setup_logging  # noqa: E402

SOURCE = "own_scrape"


def iter_docs(raw_dir: Path, sink: Sink, sites: dict[str, dict[str, Any]]) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for path in sorted(raw_dir.glob("*/articles.jsonl")):
        site = path.parent.name
        genre = sites.get(site, {}).get("genre", "unknown")
        with open(path, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                sink.read(raw_bytes=len(raw))
                line = decode_strict(raw)
                if line is None:
                    sink.drop("utf8_error")
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    sink.drop("json_error")
                    continue
                canonical = row.get("canonical_url") or row.get("url") or ""
                if canonical in seen:
                    sink.drop("duplicate_canonical_url")
                    continue
                seen.add(canonical)
                title = (row.get("title") or "").strip()
                body = (row.get("text") or "").strip()
                text = f"{title}\n{body}" if title and not body.startswith(title) else body
                meta = {"site": site, "canonical_url": canonical or None,
                        "extractor_version": row.get("extractor_version"), "fetched_at_utc": row.get("fetched_at_utc")}
                yield make_doc(SOURCE, sha1_hex(canonical or body), genre, text,
                               url=row.get("url"), title=title, date=row.get("published"), meta=meta)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_scrape")
    if not any(args.raw_dir.glob("*/articles.jsonl")):
        raise SystemExit(f"no */articles.jsonl under {args.raw_dir}; run collect/scrape/crawl.py first")
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)
    write_docs(args, sink, iter_docs(args.raw_dir, sink, own_scrape_sites()), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
