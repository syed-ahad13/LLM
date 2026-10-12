"""Extract Urdu Wikipedia articles from the pages-articles dump (Step 3).

    python clean/extract_wiki.py --limit 2000            # smoke run
    python clean/extract_wiki.py --jobs 8                # full run

Streams data/raw/wiki_ur/*.xml.bz2 through bz2 into ElementTree.iterparse and
clears each <page> after use, so memory stays flat. The localized namespace
names come from <siteinfo><namespaces>, plus the aliases in clean/wikitext.py.

Kept: namespace-0 pages that are not redirects and not disambiguation pages
(detected on the raw wikitext by template, category or title; each method is
counted separately). Markup is stripped by clean/wikitext.py, in worker
processes over ordered batches of pages; parsing stays in this process.
text = title + "\\n" + body. Articles whose body is under --min-body-chars
(default 200) are dropped and counted.
"""

import argparse
import bz2
import logging
import multiprocessing
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import Sink, add_common_args, input_files, make_doc, write_docs  # noqa: E402
from clean.wikitext import Namespaces, disambiguation_reason, is_redirect_text, strip_wikitext  # noqa: E402
from common import ordered_imap, setup_logging  # noqa: E402

SOURCE = "wiki_ur"
GENRE = "encyclopedia"
URL_BASE = "https://ur.wikipedia.org/wiki/"
BATCH_PAGES = 200

Page = dict[str, Any]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(elem: ET.Element, name: str) -> str | None:
    for child in elem:
        if _local(child.tag) == name:
            return child.text or ""
    return None


def iter_pages(path: Path, sink: Sink, namespaces_out: dict[int, str]) -> Iterator[Page]:
    """Raw pages from the dump. Fills namespaces_out from <siteinfo> before the first page."""
    with bz2.open(path, "rb") as f:
        context = ET.iterparse(f, events=("start", "end"))
        root = None
        for event, elem in context:
            if root is None:
                root = elem
            if event != "end":
                continue
            tag = _local(elem.tag)
            if tag == "namespace":
                namespaces_out[int(elem.get("key", "0"))] = (elem.text or "").strip()
            elif tag == "page":
                page: Page = {"redirect": False, "revision": {}}
                for child in elem:
                    name = _local(child.tag)
                    if name == "redirect":
                        page["redirect"] = True
                    elif name == "revision":
                        page["revision"] = {
                            "id": _child_text(child, "id"),
                            "timestamp": _child_text(child, "timestamp"),
                            "text": _child_text(child, "text") or "",
                        }
                    elif name in ("title", "ns", "id"):
                        page[name] = child.text or ""
                text = page["revision"].get("text", "")
                sink.read(raw_bytes=len(text.encode("utf-8")))
                root.clear()  # drops this page and everything before it
                yield page


def candidate_pages(pages: Iterator[Page], sink: Sink) -> Iterator[Page]:
    """Namespace-0, non-redirect pages."""
    for page in pages:
        if page.get("ns") != "0":
            sink.drop("not_article_namespace")
        elif page["redirect"] or is_redirect_text(page["revision"].get("text", "")):
            sink.drop("redirect")
        else:
            yield page


def batches(pages: Iterator[Page], size: int) -> Iterator[list[Page]]:
    batch: list[Page] = []
    for page in pages:
        batch.append(page)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(namespaces: Namespaces, min_body_chars: int) -> None:
    _WORKER["ns"] = namespaces
    _WORKER["min_body_chars"] = min_body_chars


def page_to_doc(page: Page, ns: Namespaces, min_body_chars: int) -> tuple[str, Any]:
    """("doc", document) or ("drop", reason) for one candidate page."""
    title = page.get("title", "")
    raw = page["revision"].get("text", "")
    reason = disambiguation_reason(title, raw, ns)
    if reason:
        return "drop", f"disambiguation_{reason}"
    body = strip_wikitext(raw, ns)
    if len(body) < min_body_chars:
        return "drop", "short_body"
    meta = {"page_id": page.get("id"), "revision_id": page["revision"].get("id"),
            "revision_timestamp": page["revision"].get("timestamp")}
    url = URL_BASE + quote(title.replace(" ", "_"), safe="/:()_,'-")
    return "doc", make_doc(SOURCE, page.get("id", ""), GENRE, f"{title}\n{body}", url=url, title=title, meta=meta)


def process_batch(batch: list[Page]) -> list[tuple[str, Any]]:
    return [page_to_doc(p, _WORKER["ns"], _WORKER["min_body_chars"]) for p in batch]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def iter_docs(path: Path, sink: Sink, jobs: int, min_body_chars: int, log: logging.Logger) -> Iterator[dict[str, Any]]:
    namespaces: dict[int, str] = {}
    pages = candidate_pages(iter_pages(path, sink, namespaces), sink)
    first = next(pages, None)  # <siteinfo> has been read once the first page is out
    if first is None:
        return
    ns = Namespaces.from_siteinfo(namespaces)
    log.info("namespaces from siteinfo: %d; file names %s; category names %s",
             len(namespaces), sorted(ns.file), sorted(ns.category))

    def all_pages() -> Iterator[Page]:
        yield first
        yield from pages

    pool = multiprocessing.Pool(jobs, _init_worker, (ns, min_body_chars)) if jobs > 1 else None
    if pool is None:
        _init_worker(ns, min_body_chars)
    try:
        for results in ordered_imap(pool, process_batch, batches(all_pages(), BATCH_PAGES), window=4 * jobs):
            for kind, value in results:
                if kind == "doc":
                    yield value
                else:
                    sink.drop(value)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    p.add_argument("--jobs", type=int, default=max(1, (multiprocessing.cpu_count() or 2) - 1),
                   help="worker processes for markup stripping (default: CPUs - 1; 1 = no pool)")
    p.add_argument("--min-body-chars", type=int, default=200)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_wiki")
    [path] = input_files(args.raw_dir, ".xml.bz2", log)[-1:]  # newest dated dump
    log.info("reading %s with %d job(s)", path, args.jobs)
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)
    write_docs(args, sink, iter_docs(path, sink, args.jobs, args.min_body_chars, log), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
