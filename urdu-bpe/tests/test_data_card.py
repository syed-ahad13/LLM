"""Tests for clean/data_card.py (Step 6d: DATA_CARD.md and docs/cleaning_detail.md from the stats)."""

import json
from pathlib import Path
from typing import Any

import pytest

from clean import budget, corpus_stats, data_card, split
from clean.extract_common import make_doc
from common import write_jsonl

SOURCES = {"own_scrape": 40, "wiki_ur": 60, "fineweb2_urd": 300}
GAPS = """<!-- instructions that must not reach the card -->
- <!-- REVIEW --> **Genre.** {top_genre} leads at {top_genre_share}; Wikipedia {share_wiki_ur}; FineWeb .pk {fineweb_pk}.
- <!-- REVIEW --> **Unknown.** {no_such_value} stays as typed.
"""


def stage_rows(n: int, size: int) -> dict[str, dict[str, Any]]:
    """Plausible per-stage stats for one source: each stage removes a little."""
    return {
        "extract": {"records_read": n + 5, "docs_written": n, "raw_text_bytes": size + 1000, "bytes_written": size},
        "normalize": {"docs_in": n, "docs_out": n, "bytes_in": size, "bytes_out": size - 100},
        "langfilter": {"docs_in": n, "docs_out": n - 1, "bytes_in": size - 100, "bytes_out": size - 300,
                       "drop_bytes_arabic_or_persian": 150, "drop_lines_arabic_or_persian": 3,
                       "drop_docs_sindhi_doc": 1, "drop_bytes_sindhi": 50},
        "boilerplate": {"docs_in": n - 1, "docs_out": n - 1, "bytes_in": size - 300, "bytes_out": size - 400,
                        "drop_bytes_boilerplate": 100, "drop_lines_boilerplate": 4, "min_docs": 5},
        "dedup": {"docs_in": n - 1, "docs_out": n - 1, "bytes_in": size - 400, "bytes_out": size - 400},
    }


def audit(docs: int, yeh: int, space_runs: int) -> dict[str, Any]:
    cats = {"arabic_yeh": {"count": yeh, "matches": {"U+064A ARABIC LETTER YEH": yeh} if yeh else {}},
            "arabic_heh": {"count": 7, "matches": {"U+0647 ARABIC LETTER HEH": 7}},
            "space_run": {"count": space_runs, "matches": {"U+0020 U+0020": space_runs} if space_runs else {}}}
    return {"sources": {"wiki_ur": {"docs": docs, "chars": 1_000_000, "categories": cats}}}


def build(tmp_path: Path) -> list[str]:
    clean = tmp_path / "clean"
    for s, n in SOURCES.items():
        write_jsonl(clean / f"{s}.jsonl.gz", [
            make_doc(s, str(i), {"own_scrape": "news", "wiki_ur": "encyclopedia", "fineweb2_urd": "web"}[s],
                     f"{s} متن {i}\n" + " ".join(["لفظ"] * (10 + i % 30)), url=f"https://x.{('pk', 'com')[i % 2]}/")
            for i in range(n)])
    stats, final = tmp_path / "stats", tmp_path / "final"
    assert budget.main(["--clean-dir", str(clean), "--scan", str(stats / "budget_scan.json"),
                        "--out", str(final / "budget.json"), "--total-mb", "0.1", "--min-total-mb", "0.01",
                        "--fill", "fineweb2_urd", "--jobs", "1"]) == 0
    assert split.main(["--budget", str(final / "budget.json"), "--clean-dir", str(clean), "--out-dir", str(final),
                       "--jobs", "1"]) == 0
    assert corpus_stats.main(["--final-dir", str(final), "--out", str(stats / "corpus_totals.json"),
                              "--jobs", "1"]) == 0
    per_stage: dict[str, dict[str, Any]] = {st: {} for st in data_card.STAGES}
    for s, n in SOURCES.items():
        for st, row in stage_rows(n, 100_000).items():
            per_stage[st][s] = row
    for st, rows in per_stage.items():
        (stats / f"{st}.json").write_text(json.dumps(rows), encoding="utf-8")
    (stats / "char_audit.json").write_text(json.dumps(audit(10, 25, 4)), encoding="utf-8")
    (stats / "char_audit_normalized.json").write_text(json.dumps(audit(10, 0, 0)), encoding="utf-8")
    (tmp_path / "gaps.md").write_text(GAPS, encoding="utf-8")
    return ["--stats-dir", str(stats), "--manifest", str(final / "manifest.json"), "--gaps", str(tmp_path / "gaps.md"),
            "--out", str(tmp_path / "DATA_CARD.md"), "--detail-out", str(tmp_path / "cleaning_detail.md")]


def test_card_and_detail(tmp_path: Path) -> None:
    argv = build(tmp_path)
    assert data_card.main(argv) == 0
    card = (tmp_path / "DATA_CARD.md").read_text(encoding="utf-8")
    for heading in ("## Sources", "## Corpus totals", "## Cleaning steps", "## Normalization",
                    "## Known gaps and biases", "## Rebuild"):
        assert heading in card
    totals = json.loads((tmp_path / "stats" / "corpus_totals.json").read_text(encoding="utf-8"))
    t = totals["total"]
    assert f"| **total** | {t['mb']:,.2f} | {t['mib']:,.2f} | {t['docs']:,} | {t['words']:,} | {t['unique_words']:,} |" in card
    assert "| Urdu Wikipedia | encyclopedia | <https://dumps.wikimedia.org/urwiki/> | CC BY-SA 4.0" in card
    assert "| Arabic yeh ي ى → Urdu yeh ی | 25 | 25 |" in card
    assert "| No space before ۔ ، ؟ ؛ | 0 | 0 |" in card
    assert "| 2. Normalize | normalize.py, rules below | 0.00 | 0.1% | 0 |" in card  # 300 bytes of 300,000
    assert card.count("<!-- REVIEW -->") == 2 and "instructions that must not reach the card" not in card
    assert "{no_such_value} stays as typed" in card and "{top_genre}" not in card
    manifest = json.loads((tmp_path / "final" / "manifest.json").read_text(encoding="utf-8"))
    share = manifest["sources"]["wiki_ur"]["bytes"] / manifest["totals"]["bytes"]
    assert f"Wikipedia {share:.1%}" in card
    assert "python clean/budget.py " in card and "--fill fineweb2_urd" in card
    detail = (tmp_path / "cleaning_detail.md").read_text(encoding="utf-8")
    assert "## Step 6: budget and split" in detail and "| fineweb2_urd | fill |" in detail


def test_audits_of_different_samples_are_refused(tmp_path: Path) -> None:
    argv = build(tmp_path)
    (tmp_path / "stats" / "char_audit_normalized.json").write_text(json.dumps(audit(11, 0, 0)), encoding="utf-8")
    with pytest.raises(SystemExit, match="different samples"):
        data_card.main(argv)


def test_stale_corpus_totals_are_refused(tmp_path: Path) -> None:
    argv = build(tmp_path)
    path = tmp_path / "stats" / "corpus_totals.json"
    totals = json.loads(path.read_text(encoding="utf-8"))
    totals["manifest_sha256"]["val"] = "0" * 64
    path.write_text(json.dumps(totals), encoding="utf-8")
    with pytest.raises(SystemExit, match="corpus_stats"):
        data_card.main(argv)
