"""Tests for clean/dedup.py (Step 5b: boilerplate lines and exact duplicate documents)."""

import json
from pathlib import Path
from typing import Any

import pytest

from clean import dedup
from clean.extract_common import make_doc
from common import iter_jsonl, write_jsonl

BODY = "یہ ایک لمبا اردو جملہ ہے جو دستاویز کو کافی الفاظ دینے کے لیے لکھا گیا ہے۔"
CREDIT = "اس رپورٹ کا مواد اے ایف پی سے لیا گیا ہے۔"


def doc(source: str, i: int, text: str) -> dict[str, Any]:
    return make_doc(source, str(i), "test", text)


def body(i: int) -> str:
    return f"{BODY} نمبر {i}"


def filtered(tmp_path: Path) -> Path:
    d = tmp_path / "filtered"
    write_jsonl(d / "fineweb2_urd.jsonl.gz", [
        doc("fineweb2_urd", 1, body(1) + "\n" + CREDIT),
        doc("fineweb2_urd", 2, body(2) + "\n" + CREDIT + "\n" + CREDIT),  # repeated inside one doc: counts once
        doc("fineweb2_urd", 3, body(3) + "\n" + CREDIT),
        doc("fineweb2_urd", 4, body(4)),
        doc("fineweb2_urd", 5, body(4)),  # duplicate in source
        doc("fineweb2_urd", 6, body(100)),  # duplicate of own_scrape, which has priority
        doc("fineweb2_urd", 7, "مختصر\n" + CREDIT),  # too short once the credit is gone
        doc("fineweb2_urd", 8, body(8) + " <|endoftext|> " + "آخری جملہ"),
    ])
    write_jsonl(d / "own_scrape.jsonl.gz", [doc("own_scrape", 1, body(100)), doc("own_scrape", 2, body(101))])
    return d


def run(tmp_path: Path, *extra: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    argv = ["--filtered-dir", str(filtered(tmp_path)), "--out-dir", str(tmp_path / "clean"),
            "--samples-dir", str(tmp_path / "samples"), "--stats-dir", str(tmp_path / "stats"), "--min-docs", "3",
            *extra]
    assert dedup.main(argv) == 0
    out = {p.name.removesuffix(".jsonl.gz"): list(iter_jsonl(p)) for p in sorted((tmp_path / "clean").glob("*.jsonl.gz"))}
    bp = json.loads((tmp_path / "stats" / "boilerplate.json").read_text(encoding="utf-8"))
    dd = json.loads((tmp_path / "stats" / "dedup.json").read_text(encoding="utf-8"))
    return out, bp, dd


def test_output_documents(tmp_path: Path) -> None:
    out, _, _ = run(tmp_path)
    assert [d["text"] for d in out["own_scrape"]] == [body(100), body(101)]
    fw = {d["doc_id"]: d["text"] for d in out["fineweb2_urd"]}
    assert fw == {
        "fineweb2_urd:1": body(1),
        "fineweb2_urd:2": body(2),
        "fineweb2_urd:3": body(3),
        "fineweb2_urd:4": body(4),
        "fineweb2_urd:8": body(8) + " آخری جملہ",
    }


def test_stats(tmp_path: Path) -> None:
    _, bp, dd = run(tmp_path)
    b, d = bp["fineweb2_urd"], dd["fineweb2_urd"]
    assert b["docs_in"] == 8 and b["docs_out"] == 7
    assert b["drop_lines_boilerplate"] == 5  # every copy of the credit, including the repeat in doc 2
    assert b["drop_bytes_boilerplate"] == 5 * len(CREDIT.encode())
    assert b["boilerplate_distinct_lines"] == 1 and b["min_docs"] == 3
    assert b["drop_docs_min_words"] == 1
    assert b["endoftext_removed"] == 1
    assert d["docs_in"] == 7 and d["docs_out"] == 5
    assert d["drop_docs_duplicate_in_source"] == 1
    assert d["drop_docs_duplicate_other_source"] == 1
    assert d["drop_bytes_duplicate_other_source"] == len(body(100).encode())
    assert dd["own_scrape"]["docs_out"] == 2 and bp["own_scrape"]["drop_lines_boilerplate"] == 0
    assert b["bytes_out"] == d["bytes_in"]


def test_priority_order_not_argument_order(tmp_path: Path) -> None:
    out, _, _ = run(tmp_path, "--source", "fineweb2_urd", "--source", "own_scrape")
    assert len(out["own_scrape"]) == 2 and "fineweb2_urd:6" not in {d["doc_id"] for d in out["fineweb2_urd"]}
    assert dedup.find_sources(tmp_path / "filtered", None) == ["own_scrape", "fineweb2_urd"]


def test_high_cutoff_keeps_lines(tmp_path: Path) -> None:
    out, bp, _ = run(tmp_path, "--min-docs", "5")
    assert bp["fineweb2_urd"]["drop_lines_boilerplate"] == 0
    assert any(CREDIT in d["text"] for d in out["fineweb2_urd"])


def test_top_lines_and_samples(tmp_path: Path) -> None:
    run(tmp_path)
    rows = list(iter_jsonl(tmp_path / "samples" / "boilerplate" / "fineweb2_urd.jsonl"))
    assert rows == [{"docs": 4, "line": CREDIT}]
    assert (tmp_path / "samples" / "clean" / "fineweb2_urd.sample.jsonl").exists()


def test_deterministic(tmp_path: Path) -> None:
    a, _, _ = run(tmp_path / "a")
    b, _, _ = run(tmp_path / "b")
    assert a == b


def test_report_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["--filtered-dir", str(filtered(tmp_path)), "--out-dir", str(tmp_path / "clean"),
            "--stats-dir", str(tmp_path / "stats"), "--report"]
    assert dedup.main(argv) == 0
    text = capsys.readouterr().out
    assert "== fineweb2_urd" in text and CREDIT[:20] in text
    assert not (tmp_path / "clean").exists() and not (tmp_path / "stats").exists()


def test_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(tmp_path)
    capsys.readouterr()
    assert dedup.main(["--table", "--stats-dir", str(tmp_path / "stats")]) == 0
    table = capsys.readouterr().out
    assert table.index("own_scrape") < table.index("fineweb2_urd") and "total" in table


def test_h64_is_stable() -> None:
    assert dedup.h64("abc") == 0xd8bb14d833d59559  # blake2b-64, big-endian: changing it changes which lines match
    assert dedup.h64("abc") == dedup.h64("abc") != dedup.h64("abd")
    assert 0 <= dedup.h64(BODY) < 1 << 64


def test_in_sorted_small_and_large_values() -> None:
    import numpy as np
    arr = np.array(sorted([3, 1 << 40, (1 << 63) + 5, (1 << 64) - 1]), dtype=np.uint64)
    for v in (3, 1 << 40, (1 << 63) + 5, (1 << 64) - 1):
        assert dedup.in_sorted(arr, v)
    for v in (0, 4, (1 << 63) + 4, (1 << 64) - 2):
        assert not dedup.in_sorted(arr, v)
    assert not dedup.in_sorted(np.empty(0, dtype=np.uint64), 3)
