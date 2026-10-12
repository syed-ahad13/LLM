"""End-to-end tests of clean/normalize_filter.py (Step 5) on small synthetic interim files."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from clean import normalize_filter
from clean.extract_common import make_doc
from clean.normalize_filter import Examples, process_doc
from common import iter_jsonl, write_jsonl

URDU = "یہ ایک لمبا اردو جملہ ہے جو دستاویز کو کافی الفاظ دینے کے لیے لکھا گیا ہے۔"
SINDHI = "پا\u06AAستان جی پر\u068Fیہی وزیر \u06FE \u067Dی بار \u0680ائر آیو"
SINDHI_NO_SPECIAL = "مارجی ویل عمل دارن کی دفنایو ویو"  # Sindhi without Sindhi-only letters
QURAN = "".join(c + "\u064E" if "\u0621" <= c <= "\u06FF" else c for c in "ان مثل عیسی عند اللہ کمثل آدم خلقہ")


def doc(source: str, i: int, text: str) -> dict[str, Any]:
    return make_doc(source, str(i), "test", text, url=f"https://example.invalid/{i}", title=f"t{i}")


def interim(tmp_path: Path) -> Path:
    d = tmp_path / "interim"
    write_jsonl(d / "news.jsonl.gz", [
        doc("news", 1, "\uFEFF" + URDU + "\nEnglish line only here\n" + QURAN + "\n2008\n" + URDU.replace("\u06CC", "\u064A")),
        doc("news", 2, "\n".join([SINDHI] * 8 + [SINDHI_NO_SPECIAL])),
        doc("news", 3, URDU + "\n" + URDU + "\n" + SINDHI),  # an Urdu article with a Sindhi title in it
        doc("news", 4, "مختصر متن ہے"),
        doc("news", 5, "\u200B \u200E\n  \n"),
        doc("news", 6, "Only English here\n123"),
    ])
    write_jsonl(d / "ufal_umc.jsonl.gz", [doc("ufal_umc", i, t) for i, t in
                                          enumerate(["تقلید کیا ہے؟", "دیگر امور", "وہ گھر آیا ۔"])])
    return d


def run(tmp_path: Path, *extra: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    argv = ["--interim-dir", str(interim(tmp_path)), "--out-dir", str(tmp_path / "out"),
            "--samples-dir", str(tmp_path / "samples"), "--stats-dir", str(tmp_path / "stats"), "--jobs", "1", *extra]
    assert normalize_filter.main(argv) == 0
    out = {p.name.removesuffix(".jsonl.gz"): list(iter_jsonl(p)) for p in sorted((tmp_path / "out").glob("*.jsonl.gz"))}
    norm = json.loads((tmp_path / "stats" / "normalize.json").read_text(encoding="utf-8"))
    filt = json.loads((tmp_path / "stats" / "langfilter.json").read_text(encoding="utf-8"))
    return out, norm, filt


def test_documents_kept_and_text(tmp_path: Path) -> None:
    out, _, _ = run(tmp_path)
    news = {d["doc_id"]: d for d in out["news"]}
    assert list(news) == ["news:1", "news:3"]
    assert news["news:1"]["text"] == URDU + "\n" + URDU  # BOM gone, Arabic yeh mapped, 3 lines dropped
    assert news["news:3"]["text"] == URDU + "\n" + URDU
    assert list(news["news:1"]) == ["doc_id", "source", "genre", "url", "title", "date", "text", "meta"]
    assert news["news:1"]["url"] == "https://example.invalid/1"
    assert [d["text"] for d in out["ufal_umc"]] == ["تقلید کیا ہے؟", "وہ گھر آیا۔"]  # min 3 words; space before ۔ removed


def test_stats(tmp_path: Path) -> None:
    _, norm, filt = run(tmp_path)
    n, f = norm["news"], filt["news"]
    assert n["docs_in"] == 6 and n["docs_out"] == 5 and n["drop_docs_empty"] == 1
    assert f["docs_in"] == 5 and f["docs_out"] == 2
    assert f["drop_docs_sindhi_doc"] == 1
    assert f["drop_docs_no_lines_left"] == 1
    assert f["drop_docs_min_words"] == 1
    assert f["drop_lines_mostly_not_urdu_script"] == 2
    assert f["drop_lines_arabic_or_persian"] == 1
    assert f["drop_lines_no_letters"] == 2
    assert f["drop_lines_sindhi"] == 8 + 1
    assert f["drop_bytes_no_letters"] == len("2008") + len("123")
    assert f["min_words"] == 10 and filt["ufal_umc"]["min_words"] == 3
    assert f["bytes_out"] == 2 * len((URDU + "\n" + URDU).encode())
    assert f["words_out"] == 4 * len(URDU.split())
    assert f["lines_out"] == 4
    assert n["bytes_out"] == f["bytes_in"]
    assert filt["ufal_umc"]["drop_docs_min_words"] == 1


def test_min_words_override_and_limit(tmp_path: Path) -> None:
    out, norm, filt = run(tmp_path, "--min-words", "1", "--limit", "4", "--source", "news")
    assert [d["doc_id"] for d in out["news"]] == ["news:1", "news:3", "news:4"]
    assert norm["news"]["docs_in"] == 4 and norm["news"]["limit"] == 4
    assert filt["news"]["min_words"] == 1
    assert "ufal_umc" not in filt


def test_pool_gives_the_same_output(tmp_path: Path) -> None:
    one, _, f1 = run(tmp_path / "a")
    two, _, f2 = run(tmp_path / "b", "--jobs", "2")
    assert one == two
    for f in (f1, f2):
        for row in f.values():
            row.pop("seconds"), row.pop("jobs")
    assert f1 == f2


def test_examples_and_samples_written(tmp_path: Path) -> None:
    run(tmp_path)
    rows = list(iter_jsonl(tmp_path / "samples" / "dropped_lines" / "news.jsonl"))
    reasons = Counter(r["reason"] for r in rows)
    assert reasons == {"sindhi": 1, "mostly_not_urdu_script": 2, "arabic_or_persian": 1, "no_letters": 2}  # distinct lines
    assert all(set(r) == {"reason", "line", "doc_id"} for r in rows)
    sample = list(iter_jsonl(tmp_path / "samples" / "filtered" / "news.sample.jsonl"))
    assert [d["doc_id"] for d in sample] == ["news:1", "news:3"]


def test_examples_keep_smallest_hashes_whatever_the_order() -> None:
    lines = [("r", f"line {i}", f"d{i}") for i in range(50)]
    a, b = Examples(5), Examples(5)
    a.offer(lines)
    b.offer(reversed(lines))
    assert sorted(x["line"] for x in a.rows()) == sorted(x["line"] for x in b.rows())
    assert len(a.rows()) == 5
    c = Examples(5)
    c.offer([("r", "same", "d1"), ("r", "same", "d2")])
    assert len(c.rows()) == 1


def test_process_doc_counts_in_and_out() -> None:
    norm: Counter[str] = Counter()
    filt: Counter[str] = Counter()
    dropped: list[tuple[str, str, str]] = []
    kept = process_doc(doc("s", 1, URDU + "\n\nabc def"), 3, norm, filt, dropped)
    assert kept is not None and kept["text"] == URDU
    assert norm["lines_in"] == 3 and norm["lines_out"] == 2
    assert dropped == [("mostly_not_urdu_script", "abc def", "s:1")]


def test_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(tmp_path)
    capsys.readouterr()
    assert normalize_filter.main(["--table", "--stats-dir", str(tmp_path / "stats")]) == 0
    table = capsys.readouterr().out
    assert "news" in table and "ufal_umc" in table and "sindhi" in table and "total" in table


def test_missing_source(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="nope"):
        normalize_filter.main(["--interim-dir", str(interim(tmp_path)), "--source", "nope",
                               "--stats-dir", str(tmp_path / "stats")])
