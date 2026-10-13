"""Tests for clean/budget.py (Step 6a: sizes per source and genre, and the byte budget)."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from clean import budget
from clean.extract_common import make_doc
from common import iter_jsonl, write_jsonl

WORD = "اردو"  # 8 bytes


def text(n_words: int, i: int) -> str:
    return " ".join([WORD] * n_words) + f" {i}"


def clean_dir(tmp_path: Path) -> Path:
    d = tmp_path / "clean"
    write_jsonl(d / "own_scrape.jsonl.gz",
                [make_doc("own_scrape", str(i), "news" if i % 4 else "opinion/blogs", text(100, i)) for i in range(20)])
    write_jsonl(d / "wiki_ur.jsonl.gz", [make_doc("wiki_ur", str(i), "encyclopedia", text(200, i)) for i in range(50)])
    write_jsonl(d / "fineweb2_urd.jsonl.gz", [make_doc("fineweb2_urd", str(i), "web", text(60, i)) for i in range(3000)])
    write_jsonl(d / "urdu_news_1m.jsonl.gz", [
        make_doc("urdu_news_1m", str(i), "news", text(60, i), meta={"punct_stripped": i % 10 != 0})
        for i in range(1500)])
    return d


def run(tmp_path: Path, *extra: str) -> tuple[int, dict[str, Any] | None]:
    out = tmp_path / "final" / "budget.json"
    argv = ["--clean-dir", str(tmp_path / "clean"), "--scan", str(tmp_path / "stats" / "budget_scan.json"),
            "--out", str(out), "--jobs", "2", *extra]
    code = budget.main(argv)
    return code, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


def kept_by_rule(path: Path, keep_below: int) -> tuple[int, int]:
    """(docs, text bytes) kept by the rule in the brief, recomputed document by document."""
    docs = size = 0
    for doc in iter_jsonl(path):
        if int(hashlib.blake2b(doc["doc_id"].encode("utf-8")).hexdigest(), 16) % 10**6 < keep_below:
            docs += 1
            size += len(doc["text"].encode("utf-8"))
    return docs, size


def test_keep_key_is_the_brief_formula() -> None:
    for doc_id in ("wiki_ur:1", "fineweb2_urd:<urn:uuid:abc>", "اردو:۱"):
        expected = int(hashlib.blake2b(doc_id.encode("utf-8")).hexdigest(), 16) % 10**6
        assert budget.keep_key(doc_id) == expected
        assert 0 <= budget.keep_key(doc_id) < budget.HASH_MOD


def test_scan_counts(tmp_path: Path) -> None:
    d = clean_dir(tmp_path)
    row = budget.scan_source(d / "own_scrape.jsonl.gz")
    docs = list(iter_jsonl(d / "own_scrape.jsonl.gz"))
    assert row["docs"] == 20 == sum(row["bucket_docs"])
    assert row["text_bytes"] == sum(len(x["text"].encode("utf-8")) for x in docs) == sum(row["bucket_bytes"])
    assert row["genres"]["opinion/blogs"]["docs"] == 5 and row["genres"]["news"]["docs"] == 15
    news = budget.scan_source(d / "urdu_news_1m.jsonl.gz")
    assert news["punct_stripped_docs"] == 1350


def test_default_policy(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = clean_dir(tmp_path)
    code, b = run(tmp_path, "--total-mb", "1.0", "--min-total-mb", "0.1")
    assert code == 0 and b is not None
    rows = b["sources"]
    total = b["totals"]["file_bytes"]
    assert total <= 1.0e6
    for s in ("own_scrape", "wiki_ur"):
        assert rows[s]["rule"] == "full" and rows[s]["rate"] == 1.0 and rows[s]["keep_below"] == 10**6
        assert rows[s]["docs"] == rows[s]["available_docs"]
    # urdu_news_1m is mostly punct_stripped, so its cap is 10%; both caps bind before 1 MB is reached.
    assert b["policy"]["caps"] == {"fineweb2_urd": 0.45, "urdu_news_1m": 0.10}
    assert rows["fineweb2_urd"]["share"] <= 0.45 and rows["fineweb2_urd"]["share"] > 0.44
    assert rows["urdu_news_1m"]["share"] <= 0.10 and rows["urdu_news_1m"]["share"] > 0.09
    for s, r in rows.items():
        docs, size = kept_by_rule(d / f"{s}.jsonl.gz", r["keep_below"])
        assert (r["docs"], r["text_bytes"]) == (docs, size)
        assert r["file_bytes"] == size + docs * len(b"<|endoftext|>")
        assert r["keep_below"] % budget.PER_BUCKET == 0
    assert sum(r["file_bytes"] for r in rows.values()) == total
    assert "opinion/blogs" in capsys.readouterr().out


def test_total_bounds_fill(tmp_path: Path) -> None:
    clean_dir(tmp_path)
    code, b = run(tmp_path, "--total-mb", "0.25", "--min-total-mb", "0.1")
    assert code == 0 and b is not None
    total = b["totals"]["file_bytes"]
    assert 0.24e6 < total <= 0.25e6
    assert b["sources"]["fineweb2_urd"]["share"] <= 0.45


def test_infeasible_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean_dir(tmp_path)
    code, b = run(tmp_path, "--total-mb", "0.05", "--min-total-mb", "0.01")
    assert code == 2 and b is None
    assert "No budget written" in capsys.readouterr().out


def test_min_total_not_reached(tmp_path: Path) -> None:
    clean_dir(tmp_path)
    code, b = run(tmp_path, "--total-mb", "5", "--min-total-mb", "2")
    assert code == 2 and b is None


def test_fill_moves_a_full_source(tmp_path: Path) -> None:
    clean_dir(tmp_path)
    # Listing the sources replaces the default fill list: urdu_news_1m would be kept whole without it.
    assert run(tmp_path, "--total-mb", "0.2", "--min-total-mb", "0.1", "--fill", "fineweb2_urd", "wiki_ur")[0] == 2
    code, b = run(tmp_path, "--total-mb", "0.2", "--min-total-mb", "0.1",
                  "--fill", "fineweb2_urd", "urdu_news_1m", "wiki_ur")
    assert code == 0 and b is not None
    rows = b["sources"]
    assert rows["wiki_ur"]["rule"] == "fill" and 0.0 < rows["wiki_ur"]["rate"] < 1.0 and rows["wiki_ur"]["cap"] is None
    assert rows["fineweb2_urd"]["share"] > 0.44 and rows["urdu_news_1m"]["share"] > 0.09
    assert 0.19e6 < b["totals"]["file_bytes"] <= 0.2e6  # within a few documents of the target


def test_mb_fixes_a_source(tmp_path: Path) -> None:
    clean_dir(tmp_path)
    code, b = run(tmp_path, "--total-mb", "1.0", "--min-total-mb", "0.1", "--mb", "urdu_news_1m=0.02")
    assert code == 0 and b is not None
    news = b["sources"]["urdu_news_1m"]
    assert news["rule"] == "mb" and 0.015e6 < news["file_bytes"] <= 0.02e6


def test_scan_cache_reused_until_input_changes(tmp_path: Path) -> None:
    d = clean_dir(tmp_path)
    run(tmp_path, "--total-mb", "1.0", "--min-total-mb", "0.1")
    scan_path = tmp_path / "stats" / "budget_scan.json"
    cached = json.loads(scan_path.read_text(encoding="utf-8"))
    cached["sources"]["own_scrape"]["marker"] = True  # survives only if the cache is reused
    scan_path.write_text(json.dumps(cached), encoding="utf-8")
    log = budget.setup_logging("test_budget", tmp_path / "logs")
    assert budget.load_scan(d, scan_path, False, 1, log)["sources"]["own_scrape"]["marker"] is True
    write_jsonl(d / "own_scrape.jsonl.gz", [make_doc("own_scrape", "x", "news", text(100, 0))])
    fresh = budget.load_scan(d, scan_path, False, 1, log)["sources"]["own_scrape"]
    assert "marker" not in fresh and fresh["docs"] == 1
