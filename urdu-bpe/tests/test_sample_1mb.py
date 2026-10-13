"""Tests for clean/sample_1mb.py (Step 7). The train_tokenizer.py CLI is tested in tests/test_trained_tokenizer.py."""

from pathlib import Path

import pytest

from clean import sample_1mb
from clean.extract_common import make_doc
from common import write_jsonl
from tests.urdu_text import urdu_text

EOT = "<|endoftext|>"


def test_prefix_ends_at_a_document_boundary(tmp_path: Path) -> None:
    docs = ["ایک", "دو تین", "a" * 50, "چار"]
    path = tmp_path / "train.txt"
    path.write_text("".join(d + EOT for d in docs), encoding="utf-8")
    sizes = [len((d + EOT).encode()) for d in docs]
    assert sample_1mb.prefix_at_boundary(path, sizes[0]) == (docs[0] + EOT).encode()
    assert sample_1mb.prefix_at_boundary(path, sizes[0] + sizes[1] - 1) == (docs[0] + EOT).encode()
    assert sample_1mb.prefix_at_boundary(path, 10**6) == path.read_bytes()
    with pytest.raises(SystemExit):
        sample_1mb.prefix_at_boundary(path, sizes[0] - 1)


def test_main_writes_the_sample(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    train.write_text(urdu_text(30_000), encoding="utf-8")
    out = tmp_path / "train_1mb.txt"
    assert sample_1mb.main(["--input", str(train), "--bytes", "10000", "--out", str(out)]) == 0
    data = out.read_bytes()
    assert len(data) <= 10_000 and data.endswith(EOT.encode()) and train.read_bytes().startswith(data)
    data.decode("utf-8")


def test_fallback_takes_sources_in_turn(tmp_path: Path) -> None:
    d = tmp_path / "clean"
    write_jsonl(d / "wiki_ur.sample.jsonl", [make_doc("wiki_ur", str(i), "encyclopedia", f"وکی {i}") for i in range(3)])
    write_jsonl(d / "own_scrape.sample.jsonl", [make_doc("own_scrape", str(i), "news", f"خبر {i}") for i in range(3)])
    data = sample_1mb.from_clean_samples(d, 10**6).decode("utf-8")
    assert data.split(EOT)[:-1] == ["خبر 0", "وکی 0", "خبر 1", "وکی 1", "خبر 2", "وکی 2"]
    assert sample_1mb.from_clean_samples(d, 30).decode("utf-8").count(EOT) == 1

