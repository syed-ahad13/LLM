"""Tests for analysis/equivalence_1mb.py and analysis/scaling.py (Step 8 reports)."""

import json
from pathlib import Path

import pytest

from analysis import equivalence_1mb, scaling
from bpe import BPETokenizer
from common import sha256_file
from tests.urdu_text import urdu_text


def reference(tmp_path: Path, merges: int = 60) -> tuple[Path, Path]:
    sample = tmp_path / "train_1mb.txt"
    sample.write_text(urdu_text(30_000), encoding="utf-8")
    tok = BPETokenizer()
    summary = tok.train_reference(sample.read_text(encoding="utf-8"), 257 + merges)
    ref = tok.save(tmp_path / "reference.json", meta={"input_sha256": sha256_file(sample),
                                                     "train_seconds": summary["seconds"] + 1.0})
    return sample, ref


def test_equivalence_report(tmp_path: Path) -> None:
    sample, ref = reference(tmp_path)
    out = tmp_path / "equivalence.md"
    js = tmp_path / "equivalence.json"
    assert equivalence_1mb.main(["--input", str(sample), "--reference", str(ref), "--out", str(out),
                                 "--json-out", str(js)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "**Result: IDENTICAL.**" in text and "| Identical pair counts | yes |" in text and "Speed-up" in text
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["identical_merges"] and data["identical_counts"] and data["merges_compared"] == 60
    assert data["speedup"] > 0 and data["reference_seconds"] > data["fast_seconds"] > 0


def test_equivalence_reports_the_first_difference(tmp_path: Path) -> None:
    sample, ref = reference(tmp_path)
    data = json.loads(ref.read_text(encoding="utf-8"))
    data["merges"][5], data["merges"][6] = data["merges"][6], data["merges"][5]
    data["meta"].pop("merge_counts")
    bad = tmp_path / "swapped.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "equivalence.md"
    assert equivalence_1mb.main(["--input", str(sample), "--reference", str(bad), "--out", str(out),
                                 "--json-out", str(tmp_path / "e.json")]) == 1
    assert "first difference at merge 6" in out.read_text(encoding="utf-8")


def test_equivalence_needs_the_same_sample(tmp_path: Path) -> None:
    sample, ref = reference(tmp_path)
    sample.write_text(urdu_text(30_000, seed=9), encoding="utf-8")
    with pytest.raises(SystemExit, match="another file"):
        equivalence_1mb.main(["--input", str(sample), "--reference", str(ref), "--out", str(tmp_path / "x.md")])


def test_scaling_run_and_report(tmp_path: Path) -> None:
    small, large = tmp_path / "a.txt", tmp_path / "b.txt"
    small.write_text(urdu_text(20_000), encoding="utf-8")
    large.write_text(urdu_text(200_000, seed=3), encoding="utf-8")
    results = {size: scaling.run_one(path, 300, 1, tmp_path / "cache", profile=(size == 2))
               for size, path in ((1, small), (2, large))}
    assert results[1]["train"]["merges"] == 300 and results[2]["distinct_words"] >= results[1]["distinct_words"]
    assert [e["merge"] for e in results[2]["train_log"]] == [300]  # log_every=1000: only the final merge
    text = scaling.report(results, 10_000_000, 300, large)
    assert "| Prefix | Words |" in text and "## Estimate for the full train.txt" in text
    assert "Super-linear?" in text and "## cProfile" in text and "ncalls" in text
    header, rule = text.splitlines()[4:6]
    assert header.count("|") == rule.count("|")
