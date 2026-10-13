"""Tests for train_tokenizer.py (Step 9).

Part 1 runs the post-training checks on the trained tokenizer,
tokenizer/urdu_bpe_50k.json, against data/final/val.txt; it is skipped while
either file is missing. URDU_TOKENIZER and URDU_VAL point it at other files
(a smoke run's tokenizer, say). Run with -s to see the numbers.

Part 2 runs the CLI end to end on generated text: phases, train_log.json,
the refusal of anything but the training split, sample runs, resume,
--check-only and the reference trainer.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

import train_tokenizer
from bpe import BPETokenizer
from common import REPO_ROOT, sha256_file
from tests.urdu_text import urdu_text

# ---------------------------------------------------------------------------
# Part 1: the trained tokenizer
# ---------------------------------------------------------------------------

TOKENIZER = Path(os.environ.get("URDU_TOKENIZER", REPO_ROOT / "tokenizer" / "urdu_bpe_50k.json"))
VAL = Path(os.environ.get("URDU_VAL", REPO_ROOT / "data" / "final" / "val.txt"))
real = pytest.mark.skipif(not (TOKENIZER.exists() and VAL.exists()),
                          reason=f"needs {TOKENIZER} and {VAL} (run train_tokenizer.py first)")


@pytest.fixture(scope="module")
def checks() -> dict[str, Any]:
    tok = BPETokenizer.load(TOKENIZER)
    res = train_tokenizer.post_training_checks(tok, TOKENIZER, VAL, 49_743)
    print("\n" + train_tokenizer.checks_text(res))
    return res


@real
def test_merges_and_ids(checks: dict[str, Any]) -> None:
    assert checks["merges"] == 49_743 and checks["vocab_size"] == 50_000
    assert checks["merge_ids"] == [257, 49_999]


@real
def test_no_merge_uses_endoftext(checks: dict[str, Any]) -> None:
    assert checks["merges_touching_256"] == 0


@real
def test_save_and_load_give_identical_encodings(checks: dict[str, Any]) -> None:
    assert checks["save_load"]["identical_merges"] and checks["save_load"]["identical_encodings"]
    assert checks["save_load"]["bytes"] > 4_000_000


@real
def test_val_documents_round_trip(checks: dict[str, Any]) -> None:
    assert checks["round_trip"] == {"docs": 1000, "failures": 0}


@real
def test_repeated_tokens_and_metrics_are_reported(checks: dict[str, Any]) -> None:
    print(f"merged tokens repeating earlier bytes: {checks['tokens_repeating_earlier_bytes']} (expected 0)")
    m = checks["metrics"]
    assert 1.0 <= m["tokens_per_word"] <= 2.0 and m["bytes_per_token"] > 2.0
    assert 0.0 < m["single_token_word_share"] <= 1.0
    assert checks["passed"], checks["failures"]


# ---------------------------------------------------------------------------
# Part 2: the CLI on generated text
# ---------------------------------------------------------------------------

@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    final = tmp_path / "final"
    final.mkdir()
    (final / "train.txt").write_text(urdu_text(150_000), encoding="utf-8")
    (final / "val.txt").write_text(urdu_text(60_000, seed=2), encoding="utf-8")
    return tmp_path


def cli(root: Path, *extra: str, out: str = "tok/t.json") -> list[str]:
    return ["--input", str(root / "final" / "train.txt"), "--val", str(root / "final" / "val.txt"),
            "--vocab-size", "400", "--out", str(root / out), "--workers", "1", "--log-every", "50",
            "--checkpoint-every", "100", "--val-mb", "0.02", "--round-trip-docs", "20",
            "--cache-dir", str(root / "cache"), "--samples-dir", str(root / "samples"),
            "--training-run-doc", str(root / "docs" / "training_run.md"), *extra]


def test_full_run(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert train_tokenizer.main(cli(corpus)) == 0
    out = capsys.readouterr().out
    assert "ALL CHECKS PASSED" in out and "Merges at ranks 1-20" in out
    tok_dir = corpus / "tok"
    for name in ("t.json", "t.merges.txt", "t.checkpoint.json", "train_log.json"):
        assert (tok_dir / name).exists(), name
    run = json.loads((tok_dir / "train_log.json").read_text(encoding="utf-8"))
    assert set(run["phases"]) == {"count_words", "init", "merges", "save"}
    assert run["total_seconds"] == pytest.approx(sum(run["phases"].values()), abs=0.05)
    assert run["merges"] == 143 and run["distinct_words"] > 0 and run["word_occurrences"] > run["distinct_words"]
    assert run["train_txt"]["sha256"] == sha256_file(corpus / "final" / "train.txt")
    assert run["train_txt"] == {**run["trained_on"], "path": run["train_txt"]["path"]}
    assert [w["to"] for w in run["merges_per_s_windows"]] == [143]
    assert run["started_utc"] <= run["finished_utc"]
    assert run["command"].startswith("python train_tokenizer.py --input ") and "--vocab-size 400" in run["command"]
    assert {"hash", "dirty"} <= set(run["git"]) and run["machine"]["cpu_model"] is not None
    assert {"trainer", "count_workers"} == set(run["peak_rss_mb"])
    assert run["checks"]["passed"] and run["checks"]["round_trip"] == {"docs": 20, "failures": 0}
    doc = (corpus / "docs" / "training_run.md").read_text(encoding="utf-8")
    assert "Wall clock by phase: word counting" in doc and "All post-training checks passed" in doc
    tok = BPETokenizer.load(tok_dir / "t.json")
    assert tok.meta["input_sha256"] == run["train_txt"]["sha256"] and "merges" in tok.meta["phase_seconds"]


def test_only_the_training_split(corpus: Path, tmp_path: Path) -> None:
    final = corpus / "final"
    other = final / "other.txt"
    shutil.copy(final / "train.txt", other)
    shutil.copy(final / "val.txt", final / "test_copy.txt")
    for argv in (["--input", str(final / "val.txt")], ["--input", str(other)],
                 ["--input", str(final / "test_copy.txt"), "--sample-mb", "0.05"],
                 ["--val", str(final / "train.txt")]):
        with pytest.raises(SystemExit, match="refusing"):
            train_tokenizer.main([*cli(corpus), *argv])
    assert train_tokenizer.main([*cli(corpus, out="tok/s.json"), "--input", str(other), "--sample-mb", "0.05"]) == 0


def test_sample_run(corpus: Path) -> None:
    assert train_tokenizer.main([*cli(corpus, out="tok/smoke.json"), "--sample-mb", "0.05"]) == 0
    sample = corpus / "samples" / "train_0.05mb.txt"
    assert sample.exists() and sample.stat().st_size <= 50_000
    assert (corpus / "final" / "train.txt").read_bytes().startswith(sample.read_bytes())
    run = json.loads((corpus / "tok" / "smoke.train_log.json").read_text(encoding="utf-8"))
    assert run["sample_mb"] == 0.05 and run["trained_on"]["sha256"] == sha256_file(sample)
    assert not (corpus / "tok" / "train_log.json").exists() and not (corpus / "docs" / "training_run.md").exists()


def test_resume_gives_the_same_tokenizer(corpus: Path) -> None:
    assert train_tokenizer.main(cli(corpus, out="tok/full.json")) == 0
    full = BPETokenizer.load(corpus / "tok" / "full.json")
    shutil.copy(corpus / "tok" / "full.checkpoint.json", corpus / "tok" / "again.checkpoint.json")
    assert train_tokenizer.main([*cli(corpus, out="tok/again.json"), "--resume"]) == 0
    again = BPETokenizer.load(corpus / "tok" / "again.json")
    assert again.merges == full.merges and again.merge_counts == full.merge_counts
    run = json.loads((corpus / "tok" / "train_log.json").read_text(encoding="utf-8"))
    assert run["resumed_from"] == 100 and run["merge_seconds_before_resume"] >= 0
    assert "resumed after 100 merges" in (corpus / "docs" / "training_run.md").read_text(encoding="utf-8")


def test_check_only(corpus: Path) -> None:
    assert train_tokenizer.main(cli(corpus)) == 0
    doc = corpus / "docs" / "training_run.md"
    doc.unlink()
    assert train_tokenizer.main([*cli(corpus), "--check-only"]) == 0
    assert doc.exists()
    assert train_tokenizer.main([*cli(corpus), "--check-only", "--vocab-size", "500"]) == 1


def test_reference_trainer_matches_fast(corpus: Path) -> None:
    base = [*cli(corpus), "--sample-mb", "0.03", "--vocab-size", "300", "--no-checks"]
    assert train_tokenizer.main([*base, "--trainer", "reference", "--out", str(corpus / "tok" / "ref.json")]) == 0
    assert train_tokenizer.main([*base, "--out", str(corpus / "tok" / "fast.json")]) == 0
    ref, fast = BPETokenizer.load(corpus / "tok" / "ref.json"), BPETokenizer.load(corpus / "tok" / "fast.json")
    assert ref.merges == fast.merges and ref.merge_counts == fast.merge_counts
    assert "train_reference" in ref.meta["phase_seconds"]
    assert ref.meta["input_sha256"] == sha256_file(corpus / "samples" / "train_0.03mb.txt")


def test_window_rates() -> None:
    log = [{"merge": m, "seconds": m / 1000} for m in (1000, 2000, 3000, 4000, 5000, 6000, 7000)]
    assert train_tokenizer.window_rates(log, 0, window=5000) == [
        {"from": 1, "to": 5000, "seconds": 5.0, "merges_per_s": 1000.0},
        {"from": 5001, "to": 7000, "seconds": 2.0, "merges_per_s": 1000.0}]
    assert train_tokenizer.window_rates(log[5:], 5000, window=5000)[0]["from"] == 5001


def test_repeated_bytes_are_reported_not_failed(corpus: Path) -> None:
    tok = BPETokenizer()
    for a, b in ((97, 98), (98, 99), (257, 99), (97, 258)):  # 259 and 260 are both b"abc"
        tok._add_merge(a, b)
    path = tok.save(corpus / "tiny.json")
    res = train_tokenizer.post_training_checks(tok, path, corpus / "final" / "val.txt", 4, val_bytes=20_000,
                                               round_trip_docs=10)
    assert res["tokens_repeating_earlier_bytes"] == 1 and res["repeat_examples"][0]["id"] == 260
    assert res["passed"]
