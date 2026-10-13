"""Tests for clean/corpus_stats.py (Step 6c: MB, documents, words and unique words per split)."""

import json
from pathlib import Path

import pytest
import regex

from clean import corpus_stats

EOT = "<|endoftext|>"
DOCS = {
    "train": ["اُردُو زبان\nایک دو تین", "می‌خواہم abc123def", "ݨ اور ڪ ایک", "ایک  دو\n\nتین"],
    "val": ["زبان نئی", "ایک"],
    "test": ["English words only", "ه باقی"],
}


def write_split(d: Path, docs: dict[str, list[str]] = DOCS) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for split, texts in docs.items():
        (d / f"{split}.txt").write_bytes("".join(t + EOT for t in texts).encode("utf-8"))
    return d


def runs(texts: list[str]) -> set[str]:
    return {r for t in texts for r in regex.findall(r"[\p{L}\p{M}‌]+", t)}


def test_word_runs_keep_marks_and_zwnj() -> None:
    assert corpus_stats.WORD_RUN.findall("اُردُو") == ["اُردُو"]
    assert corpus_stats.WORD_RUN.findall("می‌خواہم") == ["می‌خواہم"]
    assert corpus_stats.WORD_RUN.findall("abc123def ۔ x") == ["abc", "def", "x"]


@pytest.mark.parametrize("n", [1, 2, 3, 7, 50])
def test_ranges_cover_the_file_at_boundaries(tmp_path: Path, n: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(corpus_stats, "READ_BLOCK", 5)
    path = write_split(tmp_path) / "train.txt"
    data = path.read_bytes()
    rs = corpus_stats.ranges(path, n)
    assert rs[0][0] == 0 and rs[-1][1] == len(data)
    assert all(a[1] == b[0] for a, b in zip(rs, rs[1:]))
    starts = {0} | {i + len(EOT) for i in range(len(data)) if data.startswith(EOT.encode(), i)}
    assert all(a in starts for a, _ in rs)


def test_counts(tmp_path: Path) -> None:
    d = write_split(tmp_path / "final")
    out = tmp_path / "totals.json"
    assert corpus_stats.main(["--final-dir", str(d), "--out", str(out), "--jobs", "2"]) == 0
    stats = json.loads(out.read_text(encoding="utf-8"))
    for split, texts in DOCS.items():
        row = stats["splits"][split]
        size = (d / f"{split}.txt").stat().st_size
        assert row["bytes"] == size and row["mb"] == round(size / 1e6, 2)
        assert row["text_bytes"] == sum(len(t.encode()) for t in texts)
        assert row["separator_bytes"] == len(texts) * len(EOT)
        assert row["docs"] == len(texts)
        assert row["words"] == sum(len(t.split()) for t in texts)
        assert row["unique_words"] == len(runs(texts))
    every = [t for texts in DOCS.values() for t in texts]
    assert stats["total"]["unique_words"] == len(runs(every))
    assert stats["total"]["words"] == sum(len(t.split()) for t in every)
    assert stats["total"]["markers"]["char_U+0768"] == 1 and stats["total"]["markers"]["docs_with_U+0647"] == 1


def test_manifest_mismatch_fails(tmp_path: Path) -> None:
    d = write_split(tmp_path / "final")
    splits = {s: {"docs": len(t), "bytes": (d / f"{s}.txt").stat().st_size, "sha256": "x"} for s, t in DOCS.items()}
    splits["val"]["docs"] += 1
    (d / "manifest.json").write_text(json.dumps({"splits": splits}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        corpus_stats.main(["--final-dir", str(d), "--out", str(tmp_path / "t.json"), "--jobs", "1"])
