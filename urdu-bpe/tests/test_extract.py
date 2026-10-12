"""End-to-end tests of the Step 3 extractors on tiny synthetic raw files."""

import bz2
import csv
import gzip
import io
import json
import lzma
import random
import tarfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clean import extract_all, extract_cc100, extract_fineweb2, extract_makhzan, extract_news1m, extract_scrape
from clean import extract_ufal, extract_wiki
from clean.extract_common import Sink, make_doc
from common import iter_jsonl

BODY = "یہ ایک لمبا اردو جملہ ہے جو مضمون کے متن کو دو سو حروف سے زیادہ بنانے کے لیے بار بار دہرایا جاتا ہے۔ "


def run(module: Any, tmp_path: Path, raw_dir: Path, *extra: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = tmp_path / "interim.jsonl.gz"
    argv = ["--raw-dir", str(raw_dir), "--out", str(out), "--sample-out", str(tmp_path / "sample.jsonl"),
            "--stats-dir", str(tmp_path / "stats"), *extra]
    assert module.main(argv) == 0
    stats = json.loads((tmp_path / "stats" / "extract.json").read_text(encoding="utf-8"))
    [row] = stats.values()
    return list(iter_jsonl(out)), row


def check_schema(docs: list[dict[str, Any]], source: str) -> None:
    for d in docs:
        assert list(d) == ["doc_id", "source", "genre", "url", "title", "date", "text", "meta"]
        assert d["source"] == source and d["doc_id"].startswith(source + ":")
        assert isinstance(d["meta"], dict) and d["text"].strip()
    assert len({d["doc_id"] for d in docs}) == len(docs)


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------

def doc(i: int, text: str = "متن") -> dict[str, Any]:
    return make_doc("s", str(i), "g", f"{text} {i}")


def test_sink_counts_drops_and_limit() -> None:
    sink = Sink("s", limit=2)
    docs = [doc(1), make_doc("s", "x", "g", "  \n "), make_doc("s", "y", "g", "bad \udcff"), doc(2), doc(3)]
    assert [d["doc_id"] for d in sink.accept(docs)] == ["s:1", "s:2"]
    stats = sink.stats()
    assert stats["docs_written"] == 2 and stats["drop_empty_text"] == 1 and stats["drop_utf8_error"] == 1
    assert stats["bytes_written"] == len("متن 1".encode()) * 2 and stats["lines_written"] == 2 and stats["limit"] == 2


def test_sink_sample_is_deterministic_bounded_and_in_corpus_order() -> None:
    docs = [doc(i, "لفظ " * 50) for i in range(2000)]
    a, b = Sink("s", sample_bytes=20_000), Sink("s", sample_bytes=20_000)
    list(a.accept(docs))
    list(b.accept(list(reversed(docs))))
    ids_a = [json.loads(line)["doc_id"] for line in a.sample_lines()]
    ids_b = [json.loads(line)["doc_id"] for line in b.sample_lines()]
    assert sorted(ids_a) == sorted(ids_b)  # the same documents whatever the order they arrive in
    assert ids_a == sorted(ids_a, key=lambda s: int(s.split(":")[1]))  # written in corpus order
    size = sum(len(line.encode()) + 1 for line in a.sample_lines())
    assert 15_000 < size <= 20_000


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def wiki_page(pid: int, title: str, text: str, ns: int = 0, redirect: bool = False) -> str:
    esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    red = f'<redirect title="{title}" />' if redirect else ""
    return (f"<page><title>{title}</title><ns>{ns}</ns><id>{pid}</id>{red}<revision><id>{pid}00</id>"
            f"<timestamp>2026-09-01T00:00:00Z</timestamp><text bytes=\"1\">{esc}</text></revision></page>")


def write_wiki_dump(raw: Path, pages: list[str]) -> None:
    xml = ('<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/" version="0.11"><siteinfo>'
           "<sitename>ویکیپیڈیا</sitename><namespaces>"
           '<namespace key="-2" case="first-letter">میڈیا</namespace><namespace key="0" case="first-letter" />'
           '<namespace key="6" case="first-letter">فائل</namespace><namespace key="10" case="first-letter">سانچہ</namespace>'
           '<namespace key="14" case="first-letter">زمرہ</namespace></namespaces></siteinfo>'
           + "".join(pages) + "</mediawiki>")
    raw.mkdir(parents=True)
    (raw / "urwiki-20261001-pages-articles.xml.bz2").write_bytes(bz2.compress(xml.encode()))


WIKI_PAGES = [
    wiki_page(1, "لاہور", "'''لاہور''' [[پاکستان]] کا شہر ہے۔ " + BODY * 2 + "\n[[میڈیا:x.ogg]]\n== حوالہ جات ==\n{{reflist}}"),
    wiki_page(2, "زمرہ:شہر", BODY * 3, ns=14),
    wiki_page(3, "لاھور", "#REDIRECT [[لاہور]]", redirect=True),
    wiki_page(4, "مرکری", "* [[مرکری (سیارہ)]]\n{{ضد ابہام}}" + BODY * 3),
    wiki_page(5, "مختصر", "بہت چھوٹا مضمون۔"),
    wiki_page(6, "کراچی", "{{Infobox|a=b}}\nکراچی بڑا شہر ہے۔ " + BODY * 2),
]


@pytest.mark.parametrize("jobs", ["1", "2"])
def test_wiki_extraction(tmp_path: Path, jobs: str) -> None:
    raw = tmp_path / "raw"
    write_wiki_dump(raw, WIKI_PAGES)
    docs, stats = run(extract_wiki, tmp_path, raw, "--jobs", jobs)
    check_schema(docs, "wiki_ur")
    assert [d["doc_id"] for d in docs] == ["wiki_ur:1", "wiki_ur:6"]
    first = docs[0]
    assert first["text"].startswith("لاہور\nلاہور پاکستان کا شہر ہے۔ ")
    assert "reflist" not in first["text"] and "حوالہ جات" not in first["text"] and "x.ogg" not in first["text"]
    assert first["url"] == "https://ur.wikipedia.org/wiki/%D9%84%D8%A7%DB%81%D9%88%D8%B1"
    assert first["meta"] == {"page_id": "1", "revision_id": "100", "revision_timestamp": "2026-09-01T00:00:00Z"}
    assert stats["records_read"] == 6 and stats["docs_written"] == 2
    assert stats["drop_not_article_namespace"] == 1 and stats["drop_redirect"] == 1
    assert stats["drop_disambiguation_template"] == 1 and stats["drop_short_body"] == 1


def test_wiki_limit_and_parallel_order(tmp_path: Path) -> None:
    pages = [wiki_page(i, f"مضمون {i}", f"مضمون نمبر {i}۔ " + BODY * 2) for i in range(1, 451)]
    raw = tmp_path / "raw"
    write_wiki_dump(raw, pages)
    docs, stats = run(extract_wiki, tmp_path, raw, "--jobs", "3")
    assert [d["doc_id"] for d in docs] == [f"wiki_ur:{i}" for i in range(1, 451)]
    docs, stats = run(extract_wiki, tmp_path, raw, "--jobs", "3", "--limit", "7")
    assert [d["doc_id"] for d in docs] == [f"wiki_ur:{i}" for i in range(1, 8)] and stats["limit"] == 7


# ---------------------------------------------------------------------------
# UFAL
# ---------------------------------------------------------------------------

def write_ufal(raw: Path, lines: list[bytes]) -> None:
    raw.mkdir(parents=True)
    (raw / "urdu-plain-text-corpus.gz").write_bytes(gzip.compress(b"\n".join(lines) + b"\n"))


def topical_lines(n: int) -> list[str]:
    """Runs of 10 sentences sharing a topic word set, as in a corpus that keeps document order."""
    rng = random.Random(0)
    vocab = [f"لفظ{i}" for i in range(3000)]
    lines = []
    for start in range(0, n, 10):
        topic = rng.sample(vocab, 12)
        lines += [" ".join(rng.sample(topic, 8) + rng.sample(vocab, 3)) for _ in range(min(10, n - start))]
    return lines


def test_ufal_groups_when_order_is_preserved(tmp_path: Path) -> None:
    lines = topical_lines(100)
    write_ufal(tmp_path / "raw", [s.encode() for s in lines[:50]] + [b"", b"\xff\xfe broken"] + [s.encode() for s in lines[50:]])
    docs, stats = run(extract_ufal, tmp_path, tmp_path / "raw")
    check_schema(docs, "ufal_umc")
    assert stats["group_lines"] == 32 and stats["order_ratio"] >= 1.5
    assert [d["meta"]["sentences"] for d in docs] == [32, 32, 32, 4]
    assert docs[0]["doc_id"] == "ufal_umc:L1" and docs[0]["text"].split("\n") == lines[:32]
    assert docs[1]["meta"]["first_line"] == 33 and docs[1]["meta"]["last_line"] == 66  # skips the blank and bad lines
    assert stats["drop_utf8_error"] == 1 and stats["blank_lines"] == 1 and stats["records_read"] == 102


def test_ufal_one_line_per_doc_when_shuffled(tmp_path: Path) -> None:
    lines = topical_lines(200)
    random.Random(1).shuffle(lines)
    write_ufal(tmp_path / "raw", [s.encode() for s in lines])
    docs, stats = run(extract_ufal, tmp_path, tmp_path / "raw")
    assert stats["group_lines"] == 1 and stats["order_ratio"] < 1.5 and len(docs) == 200
    assert docs[5]["doc_id"] == "ufal_umc:L6" and docs[5]["text"] == lines[5]


def test_ufal_forced_group(tmp_path: Path) -> None:
    write_ufal(tmp_path / "raw", [f"جملہ {i}".encode() for i in range(10)])
    docs, _ = run(extract_ufal, tmp_path, tmp_path / "raw", "--group", "4")
    assert [d["meta"]["sentences"] for d in docs] == [4, 4, 2]


# ---------------------------------------------------------------------------
# Urdu News 1M
# ---------------------------------------------------------------------------

def test_news1m(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["Index", "Headline", "News Text", "Category", "Date", "URL", "Source", "News length"])
    w.writerow(["0", "سرخی ایک", "پہلی خبر کا متن۔ دوسرا جملہ،\nنیا پیراگراف", "Sports", "2019-01-01", "https://a.pk/1", "Express", "40"])
    w.writerow(["1", "سرخی دو", "بغیر رموز کے خبر کا متن", "Business & Economics", "2019-01-02", "https://a.pk/2", "Dawn", "20"])
    w.writerow(["2", "خالی", "", "Sports", "", "", "", "0"])
    w.writerow(["3", "خراب", "BADBYTES", "Sports", "", "", "", "0"])
    data = buf.getvalue().encode().replace(b"BADBYTES", b"\xc3\x28 text")
    (raw / "urdu-news-dataset-1M.csv").write_bytes(b"\xef\xbb\xbf" + data)
    docs, stats = run(extract_news1m, tmp_path, raw)
    check_schema(docs, "urdu_news_1m")
    assert [d["doc_id"] for d in docs] == ["urdu_news_1m:0", "urdu_news_1m:1"]
    assert docs[0]["text"] == "سرخی ایک\nپہلی خبر کا متن۔ دوسرا جملہ،\nنیا پیراگراف"
    assert docs[0]["meta"] == {"category": "Sports", "outlet": "Express", "news_length": "40", "punct_stripped": False}
    assert docs[1]["meta"]["punct_stripped"] is True and docs[1]["date"] == "2019-01-02"
    assert stats["drop_empty_story"] == 1 and stats["drop_utf8_error"] == 1 and stats["records_read"] == 4


# ---------------------------------------------------------------------------
# FineWeb-2
# ---------------------------------------------------------------------------

def test_fineweb2(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 400
    table = pa.table({
        "text": [f"ویب صفحہ نمبر {i}" for i in range(n)],
        "id": [f"<urn:uuid:{i:08d}>" for i in range(n)],
        "dump": ["CC-MAIN-2024-18"] * n,
        "url": [f"https://x.pk/{i}" for i in range(n)],
        "date": ["2024-04-20T00:00:00Z"] * n,
        "language_score": [0.9] * n,
        "minhash_cluster_size": [1] * n,
    })
    pq.write_table(table, raw / "001_00000.parquet", row_group_size=64)
    (raw / "002_00000.parquet.part").write_bytes(b"PAR1 unfinished")
    docs, stats = run(extract_fineweb2, tmp_path, raw)
    check_schema(docs, "fineweb2_urd")
    assert len(docs) == n and docs[3]["doc_id"] == "fineweb2_urd:<urn:uuid:00000003>"
    assert docs[3]["meta"] == {"language_score": 0.9, "dump": "CC-MAIN-2024-18", "minhash_cluster_size": 1,
                               "file": "001_00000.parquet"}
    sub, stats = run(extract_fineweb2, tmp_path, raw, "--keep-per-mille", "250")
    native = [d["doc_id"].removeprefix("fineweb2_urd:") for d in docs]
    assert [d["doc_id"] for d in sub] == [f"fineweb2_urd:{i}" for i in native if extract_fineweb2.kept(i, 250)]
    assert 60 < len(sub) < 140 and stats["drop_not_sampled"] == n - len(sub) and stats["keep_per_mille"] == 250


# ---------------------------------------------------------------------------
# CC-100
# ---------------------------------------------------------------------------

def test_cc100(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    data = "پہلا پیراگراف\nدوسرا پیراگراف\n\nدوسری دستاویز\n\n".encode() + b"\xff bad\n\n" + "آخری\n".encode()
    (raw / "ur.txt.xz").write_bytes(lzma.compress(data))
    docs, stats = run(extract_cc100, tmp_path, raw)
    check_schema(docs, "cc100_ur")
    assert [d["text"] for d in docs] == ["پہلا پیراگراف\nدوسرا پیراگراف", "دوسری دستاویز", "آخری"]
    assert [d["doc_id"] for d in docs] == ["cc100_ur:1", "cc100_ur:2", "cc100_ur:4"]
    assert stats["drop_utf8_error"] == 1 and stats["records_read"] == 4


# ---------------------------------------------------------------------------
# Makhzan
# ---------------------------------------------------------------------------

MAKHZAN_XML = """<?xml version="1.0" encoding="utf-8"?>
<document>
    <meta>
        <title>ایک مضمون</title>
        <author><name>نام مصنف</name><gender>Male</gender></author>
        <publication>
            <name>Bunyad</name><year>2012</year><city>Lahore</city>
            <link>https://example.org/b.pdf</link>
            <copyright-holder>Gurmani Centre</copyright-holder>
        </publication>
        <num-words>20</num-words>
        <contains-non-urdu-languages>Yes</contains-non-urdu-languages>
    </meta>
    <body>
        <section>
            <heading>پہلا حصہ</heading>
            <p>پہلا پیراگراف
               جو دو سطروں میں ہے اور <annotation lang="en">English words</annotation> بھی۔</p>
            <blockquote><p>اقتباس</p></blockquote>
        </section>
        <section><p>دوسرا پیراگراف</p></section>
    </body>
</document>
"""


def write_tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_makhzan(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_tar(raw / "makhzan-2c7ed4f78d98.tar.gz", {
        "repo/README.md": b"readme",
        "repo/text/0001.xml": MAKHZAN_XML.encode(),
        "repo/text/0002.xml": b"<document><meta><title>x</title></meta><body><p>unclosed</body></document>",
        "repo/text/0003.xml": b"<document><body><p>\xff</p></body></document>",
    })
    docs, stats = run(extract_makhzan, tmp_path, raw)
    check_schema(docs, "makhzan")
    [d] = docs
    assert d["doc_id"] == "makhzan:0001" and d["date"] == "2012"
    assert d["text"] == "پہلا حصہ\nپہلا پیراگراف جو دو سطروں میں ہے اور English words بھی۔\nاقتباس\nدوسرا پیراگراف"
    dumped = json.dumps(d, ensure_ascii=False)
    assert "نام مصنف" not in dumped and "ایک مضمون" not in dumped and "b.pdf" not in dumped  # nothing from <meta>
    assert d["url"] is None and d["title"] is None
    assert d["meta"]["copyright_holder"] == "Gurmani Centre" and d["meta"]["contains_non_urdu"] == "Yes"
    assert stats["drop_xml_parse_error"] == 1 and stats["drop_utf8_error"] == 1 and stats["records_read"] == 3


# ---------------------------------------------------------------------------
# Own scrape
# ---------------------------------------------------------------------------

def test_scrape(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for site, rows in {
        "voa_urdu": [
            {"url": "https://www.urduvoa.com/a/1.html", "canonical_url": "https://www.urduvoa.com/a/1.html",
             "title": "خبر", "text": "پہلا پیراگراف\nدوسرا", "published": "2015-01-01", "extractor_version": "voa_urdu/2"},
            {"url": "https://www.urduvoa.com/a/1.html?x", "canonical_url": "https://www.urduvoa.com/a/1.html",
             "title": "خبر", "text": "نقل", "published": "2015-01-01"},
        ],
        "humsub": [{"url": "https://www.humsub.com.pk/9/", "canonical_url": None, "title": "کالم",
                    "text": "کالم کا متن", "published": "2020-02-02"}],
    }.items():
        (raw / site).mkdir(parents=True)
        lines = [json.dumps(r, ensure_ascii=False).encode() for r in rows]
        (raw / site / "articles.jsonl").write_bytes(b"\n".join(lines) + b"\n{broken\n")
    docs, stats = run(extract_scrape, tmp_path, raw)
    check_schema(docs, "own_scrape")
    assert [d["meta"]["site"] for d in docs] == ["humsub", "voa_urdu"]
    assert [d["genre"] for d in docs] == ["opinion/blogs", "news"]
    assert docs[1]["text"] == "خبر\nپہلا پیراگراف\nدوسرا" and docs[1]["date"] == "2015-01-01"
    assert stats["drop_duplicate_canonical_url"] == 1 and stats["drop_json_error"] == 2


# ---------------------------------------------------------------------------
# extract_all
# ---------------------------------------------------------------------------

def test_format_table() -> None:
    table = extract_all.format_table({
        "wiki_ur": {"records_read": 10, "docs_written": 4, "bytes_written": 2_500_000, "drop_redirect": 3, "limit": 0},
        "ufal_umc": {"records_read": 5, "docs_written": 5, "bytes_written": 500_000, "limit": 5},
    }, ["wiki_ur", "ufal_umc", "makhzan"])
    lines = table.splitlines()
    assert lines[0].split()[:3] == ["source", "records", "read"]
    assert "redirect=3" in lines[2] and "2.50" in lines[2]
    assert "(limit 5)" in lines[3] and "none" in lines[3]
    assert "no stats" in lines[4] and lines[5].split()[:4] == ["total", "15", "9", "3.00"]


def test_command_passes_options() -> None:
    args = extract_all.parse_args(["--limit", "20", "--wiki-jobs", "3", "--fineweb-keep-per-mille", "250"])
    assert extract_all.command("wiki_ur", args)[2:] == ["--limit", "20", "--jobs", "3"]
    assert extract_all.command("fineweb2_urd", args)[2:] == ["--limit", "20", "--keep-per-mille", "250"]
    assert extract_all.command("makhzan", args)[1].endswith("clean/extract_makhzan.py")
