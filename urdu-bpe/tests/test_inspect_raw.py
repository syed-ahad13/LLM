import bz2
import csv
import gzip
import io
import lzma
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collect import inspect_raw
from collect.inspect_raw import (
    data_files,
    iter_csv_rows,
    iter_lines,
    iter_wiki_pages,
    order_test,
    order_verdict,
    reservoir_sample,
)

LINES = [f"سطر نمبر {i} میں کچھ الفاظ ہیں" for i in range(200)]


def write_compressed(path: Path, text: str, truncate: bool = False) -> Path:
    opener = {".gz": gzip.compress, ".xz": lzma.compress, ".bz2": bz2.compress}[Path(path.name.removesuffix(".part")).suffix]
    data = opener(text.encode())
    path.write_bytes(data[: len(data) // 2] if truncate else data)
    return path


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".gz", ".xz", ".bz2"])
def test_iter_lines_complete_and_truncated(tmp_path: Path, suffix: str) -> None:
    lines = LINES * 400  # about 4 MB, so even bz2 (900 kB blocks) has complete blocks before the cut
    text = "\n".join(lines) + "\n"
    full = write_compressed(tmp_path / f"c.txt{suffix}", text)
    assert list(iter_lines(full)) == lines
    cut = write_compressed(tmp_path / f"c.txt{suffix}.part", text, truncate=True)
    got = list(iter_lines(cut))  # must stop quietly, not raise
    assert 0 < len(got) < len(lines)
    assert got[:-1] == lines[: len(got) - 1]


WIKI_XML = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
<page><title>پاکستان</title><ns>0</ns><id>1</id><revision><id>9</id><text>'''پاکستان''' ایک ملک ہے۔</text></revision></page>
<page><title>کراچی</title><ns>0</ns><id>2</id><revision><id>10</id><text>#REDIRECT [[کراچی شہر]]</text></revision></page>
</mediawiki>"""


def test_iter_wiki_pages(tmp_path: Path) -> None:
    path = tmp_path / "w.xml.bz2"
    path.write_bytes(bz2.compress(WIKI_XML.encode()))
    pages = list(iter_wiki_pages(path))
    assert [p["title"] for p in pages] == ["پاکستان", "کراچی"]
    assert pages[0] == {"title": "پاکستان", "ns": "0", "id": "1", "text": "'''پاکستان''' ایک ملک ہے۔"}
    cut = tmp_path / "w.xml.bz2.part"
    cut.write_bytes(bz2.compress(WIKI_XML.encode())[:40])
    assert list(iter_wiki_pages(cut)) == []


def write_csv(path: Path, rows: list[list[str]]) -> None:
    buf = io.StringIO()
    csv.writer(buf).writerows([["Index", "Headline", "News Text"], *rows])
    path.write_text("﻿" + buf.getvalue(), encoding="utf-8")


def test_iter_csv_rows_handles_bom_quotes_and_partial_tail(tmp_path: Path) -> None:
    rows = [["0", "سرخی", "پہلی سطر\nدوسری سطر"], ["1", "دوسری", "متن۔"], ["2", "تیسری", "آخری متن"]]
    path = tmp_path / "n.csv"
    write_csv(path, rows)
    got = list(iter_csv_rows(path))
    assert [r["News Text"] for r in got] == ["پہلی سطر\nدوسری سطر", "متن۔", "آخری متن"]
    assert list(got[0]) == ["Index", "Headline", "News Text"]  # BOM stripped from the first column name

    part = tmp_path / "n.csv.part"
    part.write_bytes(path.read_bytes()[:-6])  # cut inside the last row
    assert [r["Index"] for r in iter_csv_rows(part)] == ["0", "1"]
    part.write_bytes(path.read_bytes() + b'3,"open quote')  # ends inside a quoted field
    assert [r["Index"] for r in iter_csv_rows(part)] == ["0", "1", "2"]


def test_data_files_skips_manifest_meta_and_stale_parts(tmp_path: Path) -> None:
    for name in ["MANIFEST.json", "a.gz", "a.gz.part", "a.gz.part.json", "b.csv.part", ".hidden"]:
        (tmp_path / name).write_text("x")
    assert [p.name for p in data_files(tmp_path)] == ["a.gz", "b.csv.part"]


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def make_documents(n_docs: int = 400, lines_per_doc: int = 8, seed: int = 0) -> list[str]:
    """Lines whose documents each draw from their own small vocabulary, plus shared function words."""
    rng = random.Random(seed)
    common = ["کا", "کی", "ہے", "میں", "اور"]
    lines = []
    for d in range(n_docs):
        topic = [f"لفظ{d}_{k}" for k in range(12)]
        for _ in range(lines_per_doc):
            lines.append(" ".join(rng.sample(topic, 5) + rng.sample(common, 3)))
    return lines


def test_order_test_separates_ordered_from_shuffled() -> None:
    ordered = make_documents()
    consecutive, random_pairs = order_test(ordered)
    assert consecutive / random_pairs > 1.5
    assert "preserved" in order_verdict(consecutive, random_pairs)

    shuffled = ordered[:]
    random.Random(1).shuffle(shuffled)
    consecutive, random_pairs = order_test(shuffled)
    assert 0.8 < consecutive / random_pairs < 1.2
    assert "shuffled" in order_verdict(consecutive, random_pairs) or "weak" in order_verdict(consecutive, random_pairs)


def test_order_test_needs_lines() -> None:
    with pytest.raises(ValueError):
        order_test(["ایک", "", " "])


def test_reservoir_sample_is_uniform_size_and_deterministic() -> None:
    rows = [{"i": str(i)} for i in range(5000)]
    s1, n = reservoir_sample(iter(rows), 1000, seed=1337)
    s2, _ = reservoir_sample(iter(rows), 1000, seed=1337)
    assert n == 5000 and len(s1) == 1000 and s1 == s2
    assert len({r["i"] for r in s1}) == 1000
    small, n = reservoir_sample(iter(rows[:10]), 1000, seed=1)
    assert n == 10 and small == rows[:10]


def test_report_news_counts_punctuation(tmp_path: Path, capsys) -> None:
    rows = [["0", "ایک", "یہ جملہ ہے۔ دوسرا"], ["1", "دو", "آپ کہاں ہیں؟"], ["2", "تین", "بغیر نشان"], ["3", "چار", "یہاں، وہاں"]]
    path = tmp_path / "news.csv"
    write_csv(path, rows)
    inspect_raw.report_news(path, sample_rows=1000, seed=1337)
    out = capsys.readouterr().out
    assert "rows: 4" in out
    news = out[out.index("whose 'News Text'"):].splitlines()

    def share(label: str) -> str:
        return next(line for line in news if label in line).split()[-1]

    assert share("Arabic Full Stop") == "25.0%"
    assert share("Arabic Question Mark") == "25.0%"
    assert share("any of the three") == "75.0%"
    assert share("ASCII") == "0.0%"
    assert share("alef madda") == "25.0%"


def test_report_ufal_prints_head_and_order_test(tmp_path: Path, capsys) -> None:
    path = write_compressed(tmp_path / "u.txt.gz", "\n".join(make_documents()) + "\n")
    inspect_raw.report_ufal(path, head_lines=5, order_lines=2000, seed=1337, max_chars=50)
    out = capsys.readouterr().out
    assert "first 5 lines" in out and "order test over 2,000 lines" in out and "preserved" in out


def test_report_parquet(tmp_path: Path, capsys) -> None:
    table = pa.table({
        "text": ["پہلا متن", "دوسرا متن", "تیسرا", "چوتھا"],
        "id": ["a", "b", "c", "d"],
        "url": [f"https://example.pk/{i}" for i in range(4)],
        "date": ["2024-01-01T00:00:00Z"] * 4,
        "language_score": [0.99, 0.98, 0.97, 0.96],
    })
    path = tmp_path / "000_00000.parquet"
    pq.write_table(table, path)
    inspect_raw.report_parquet(path, 3, 100)
    out = capsys.readouterr().out
    assert "text: string" in out and "rows: 4 in 1 row groups" in out
    assert "[3] text=تیسرا; url=https://example.pk/2; date=2024-01-01T00:00:00Z; language_score=0.97" in out
    assert "[4]" not in out

    part = tmp_path / "001_00000.parquet.part"
    part.write_bytes(path.read_bytes()[:100])
    inspect_raw.report_parquet(part, 3, 100)
    assert "PARTIAL parquet" in capsys.readouterr().out


def test_main_runs_over_a_raw_dir(tmp_path: Path, capsys) -> None:
    (tmp_path / "ufal_umc").mkdir()
    write_compressed(tmp_path / "ufal_umc" / "urdu-plain-text-corpus.gz.part", "\n".join(make_documents()) + "\n", truncate=True)
    (tmp_path / "urdu_news_1m").mkdir()
    write_csv(tmp_path / "urdu_news_1m" / "urdu-news-dataset-1M.csv", [["0", "ایک", "متن۔"]])
    (tmp_path / "empty").mkdir()
    assert inspect_raw.main(["--raw-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "=== ufal_umc ===" in out and "PARTIAL" in out and "order test" in out
    assert "=== urdu_news_1m ===" in out and "columns:" in out
    assert "=== empty ===" in out and "no data files yet" in out
