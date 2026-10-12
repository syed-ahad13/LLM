"""Extract Makhzan articles from the repo tarball (Step 3).

    python clean/extract_makhzan.py --limit 200
    python clean/extract_makhzan.py --show-one          # print one raw file and its extraction, write nothing

Streams data/raw/makhzan/makhzan-<commit>.tar.gz and reads every *.xml file
under text/, one article per file. Expected shape (sources.json notes):

    <document>
      <meta> <title/> <author/> <publication> <name/> <year/> <copyright-holder/> ... </publication>
             <contains-non-urdu-languages/> ... </meta>
      <body> <section> <heading/> <p/> <blockquote/> <annotation lang=".."/> ... </section> </body>
    </document>

Text comes from <body> only, never from <meta>, which holds titles, author
names and links (docs/decisions.md, "Makhzan download findings"). Each block
element (BLOCK_TAGS: <p>, <heading>, <blockquote>, ...) becomes one line, in
document order, plus any loose text between blocks. Inside a block, XML layout
whitespace (spaces, tabs, newlines from pretty-printing) collapses to one
space. <annotation> spans (non-Urdu passages) are kept inline: the Step 5 line
language filter decides about them like any other code-mixed text.

doc_id is makhzan:<file stem>. meta keeps the publication name, year,
copyright holder (for the data card's genre breakdown) and the non-Urdu flag.
url and title stay null. Makhzan text must never be quoted in anything
committed to git (sources.json terms_notes).
"""

import argparse
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import Sink, add_common_args, decode_strict, input_files, make_doc, write_docs  # noqa: E402
from common import setup_logging  # noqa: E402

SOURCE = "makhzan"
GENRE = "journalism/essays"
BLOCK_TAGS = frozenset({"p", "heading", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "li", "verse"})
_XML_WS = re.compile(r"[ \t\r\n]+")


def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _has_block_child(elem: ET.Element) -> bool:
    return any(_local(d.tag) in BLOCK_TAGS for d in elem.iter() if d is not elem)


def _blocks(elem: ET.Element) -> Iterator[str]:
    if elem.text and elem.text.strip():
        yield elem.text
    for child in elem:
        if _local(child.tag) in BLOCK_TAGS and not _has_block_child(child):
            yield "".join(child.itertext())
        else:
            yield from _blocks(child)
        if child.tail and child.tail.strip():
            yield child.tail


def body_text(body: ET.Element | None) -> str:
    if body is None:
        return ""
    lines = (_XML_WS.sub(" ", b).strip() for b in _blocks(body))
    return "\n".join(line for line in lines if line)


def _find_text(elem: ET.Element | None, path: str) -> str | None:
    if elem is None:
        return None
    found = elem.find(path)
    return found.text.strip() if found is not None and found.text and found.text.strip() else None


def parse_article(name: str, data: bytes) -> tuple[str, Any]:
    """("doc", document) or ("drop", reason) for one XML file of the tarball."""
    xml_text = decode_strict(data)
    if xml_text is None:
        return "drop", "utf8_error"
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return "drop", "xml_parse_error"
    meta_el = root.find("meta")
    body = body_text(root.find("body"))
    if not body:
        return "drop", "empty_body"
    meta = {
        "file": name,
        "publication": _find_text(meta_el, "publication/name"),
        "year": _find_text(meta_el, "publication/year"),
        "copyright_holder": _find_text(meta_el, "publication/copyright-holder"),
        "contains_non_urdu": _find_text(meta_el, "contains-non-urdu-languages"),
    }
    return "doc", make_doc(SOURCE, PurePosixPath(name).stem, GENRE, body, date=meta["year"], meta=meta)


def iter_xml_members(path: Path) -> Iterator[tuple[str, bytes]]:
    """(member path, bytes) for each .xml file under a text/ folder, in tarball order."""
    with tarfile.open(path, "r|gz") as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(".xml") and "/text/" in f"/{member.name}":
                f = tf.extractfile(member)
                if f is not None:
                    yield member.name, f.read()


def iter_docs(path: Path, sink: Sink) -> Iterator[dict[str, Any]]:
    for name, data in iter_xml_members(path):
        sink.read(raw_bytes=len(data))
        kind, value = parse_article(name, data)
        if kind == "doc":
            yield value
        else:
            sink.drop(value)


def show_one(path: Path) -> None:
    for name, data in iter_xml_members(path):
        print(f"=== raw {name} ({len(data):,} bytes) ===")
        print(data.decode("utf-8", errors="replace"))
        kind, value = parse_article(name, data)
        print(f"=== extracted ({kind}) ===")
        print(value if kind == "drop" else {k: v for k, v in value.items() if k != "text"})
        print(value["text"] if kind == "doc" else "")
        return


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p, SOURCE)
    p.add_argument("--show-one", action="store_true", help="print the first raw file and its extraction; write nothing")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_makhzan")
    [path] = input_files(args.raw_dir, ".tar.gz", log)[-1:]
    if args.show_one:
        show_one(path)
        return 0
    sink = Sink(SOURCE, limit=args.limit, sample_bytes=args.sample_bytes)
    write_docs(args, sink, iter_docs(path, sink), log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
