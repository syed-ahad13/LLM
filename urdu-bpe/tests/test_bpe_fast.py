"""Tests for the fast trainer in bpe.py (Step 8): count_words, reverse_key and BPETokenizer.train.

The central property: train() learns exactly the merges of train_reference() (Step 7), with the same pair
counts and the same early stop, on 300 random corpora drawn from small alphabets (which make many ties).
"""

import hashlib
import logging
import pickle
import random
from pathlib import Path

import pytest

import bpe
from bpe import ENDOFTEXT, BPETokenizer, count_words, reverse_key
from tests.urdu_text import urdu_text

ALPHABETS = {
    "abc": "abc ",
    "urdu": "ابتک ُ",  # four letters, a space and one diacritic (pesh)
    "digits": "0123456789.,!? ",
}
EXHAUST = 1 << 16  # largest vocab size: both trainers stop when no pair is left


def corpus(seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    name = ("abc", "urdu", "digits")[seed % 3]
    chars = rng.choices(ALPHABETS[name], k=rng.randint(50, 3000))
    if seed % 4 == 0:  # some corpora hold separators, as train.txt does
        for _ in range(rng.randint(1, 5)):
            chars.insert(rng.randrange(len(chars) + 1), ENDOFTEXT)
    return name, "".join(chars)


def both(text: str, vocab_size: int, caplog: pytest.LogCaptureFixture) -> tuple[BPETokenizer, BPETokenizer, str, str]:
    ref, fast = BPETokenizer(), BPETokenizer()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bpe"):
        ref.train_reference(text, vocab_size)
    ref_log = caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bpe"):
        fast.train(count_words(text), vocab_size, log_every=0, debug=25)
    return ref, fast, ref_log, caplog.text


# ---------------------------------------------------------------------------
# Same merges as the reference trainer
# ---------------------------------------------------------------------------

def test_tie_break_aaabdaaabac(caplog: pytest.LogCaptureFixture) -> None:
    ref, fast, ref_log, fast_log = both("aaabdaaabac", 257 + 8, caplog)
    assert fast.merges == ref.merges and len(fast.merges) == 7
    assert [fast.vocab[257 + i] for i in range(7)] == [b"aa", b"aaa", b"aaab", b"daaab", b"daaaba", b"daaabac",
                                                       b"aaabdaaabac"]
    assert "no pair left after 7 merges" in ref_log and "no pair left after 7 merges" in fast_log
    assert fast.encode("aaabdaaabac") == [263]


@pytest.mark.parametrize("block", range(10))
def test_random_corpora_match_the_reference(block: int, caplog: pytest.LogCaptureFixture) -> None:
    """300 corpora in 10 blocks of 30, trained to exhaustion by both trainers."""
    for seed in range(block * 30, block * 30 + 30):
        name, text = corpus(seed)
        ref, fast, ref_log, fast_log = both(text, EXHAUST, caplog)
        where = f"seed {seed} ({name}, {len(text)} chars)"
        assert fast.merges == ref.merges, where
        assert fast.merge_counts == ref.merge_counts, where
        assert fast.vocab == ref.vocab, where
        stop = f"no pair left after {len(ref.merges)} merges"
        assert stop in ref_log and stop in fast_log, where
        assert fast.encode(text) == ref.encode(text), where
        assert fast.decode(fast.encode(text)) == text, where


def test_urdu_text_matches_the_reference() -> None:
    text = urdu_text(100_000)
    ref, fast = BPETokenizer(), BPETokenizer()
    ref.train_reference(text, 257 + 400)
    fast.train(count_words(text), 257 + 400, log_every=0, debug=50)
    assert fast.merges == ref.merges and fast.merge_counts == ref.merge_counts


def test_target_reached_before_pairs_run_out() -> None:
    text = urdu_text(20_000)
    ref, fast = BPETokenizer(), BPETokenizer()
    ref.train_reference(text, 300)
    summary = fast.train(count_words(text), 300, log_every=0)
    assert fast.merges == ref.merges and summary["merges"] == 43


# ---------------------------------------------------------------------------
# The heap key
# ---------------------------------------------------------------------------

def random_bytes(rng: random.Random) -> bytes:
    return bytes(rng.choice(b"\x00\x01\x7f\x80\xfe\xffab") for _ in range(rng.randint(0, 6)))


def test_reverse_key_reverses_byte_order() -> None:
    rng = random.Random(1337)
    for _ in range(100_000):
        x = random_bytes(rng)
        r = rng.random()
        if r < 0.25:
            y = x + random_bytes(rng)  # x is a prefix of y (or equal)
        elif r < 0.35:
            y = x
        elif r < 0.5:
            y = x[:rng.randint(0, len(x))]  # y is a prefix of x
        else:
            y = random_bytes(rng)
        kx, ky = reverse_key(x), reverse_key(y)
        assert (x < y) == (kx > ky), (x, y)
        assert (x == y) == (kx == ky), (x, y)


def test_reverse_key_on_pairs() -> None:
    """Tuples of keys reverse the tuple order of (bytes_a, bytes_b), as the heap needs."""
    rng = random.Random(7)
    for _ in range(20_000):
        p = (random_bytes(rng), random_bytes(rng))
        q = (random_bytes(rng), random_bytes(rng)) if rng.random() < 0.7 else (p[0], random_bytes(rng))
        kp, kq = (reverse_key(p[0]), reverse_key(p[1])), (reverse_key(q[0]), reverse_key(q[1]))
        assert (p < q) == (kp > kq) and (p == q) == (kp == kq)


def test_heap_breaks_identical_byte_ties_like_the_reference() -> None:
    tok = BPETokenizer()
    for a, b in ((97, 98), (98, 99), (257, 99), (97, 258)):  # 259 and 260 are both b"abc"
        tok._add_merge(a, b)
    entries = [(-2, reverse_key(tok.vocab[x]), reverse_key(tok.vocab[100]), -((x << 16) | 100)) for x in (259, 260)]
    assert min(entries)[3] == -((260 << 16) | 100)
    from collections import Counter
    assert tok._choose(Counter({(259, 100): 2, (260, 100): 2}))[0] == (260, 100)


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

def test_count_words_on_a_string() -> None:
    assert count_words("یہ اردو ہے۔ یہ<|endoftext|>اردو یہ") == {"یہ": 1, " اردو": 1, " ہے": 1, "۔": 1, " یہ": 2,
                                                              "اردو": 1}
    assert count_words("") == {}


@pytest.mark.parametrize("workers, block", [(1, 37), (1, 10_000), (2, 1000)])
def test_count_words_on_a_file(tmp_path: Path, workers: int, block: int) -> None:
    text = urdu_text(60_000) + "آخری دستاویز بغیر علامت کے"  # the last document has no separator
    path = tmp_path / "train.txt"
    path.write_text(text, encoding="utf-8")
    assert count_words(path, workers, block_bytes=block) == count_words(text)


def test_count_words_cache(tmp_path: Path) -> None:
    path = tmp_path / "train.txt"
    path.write_text(urdu_text(20_000), encoding="utf-8")
    cache = tmp_path / "cache"
    first = count_words(path, 1, cache_dir=cache)
    files = list(cache.glob("word_counts_*.pkl"))
    assert len(files) == 1
    assert files[0].name.startswith(f"word_counts_{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}_")
    data = pickle.loads(files[0].read_bytes())
    data["counts"] = {"marker": 1}
    files[0].write_bytes(pickle.dumps(data))
    assert count_words(path, 1, cache_dir=cache) == {"marker": 1}  # loaded, not recounted
    assert count_words(path, 1, cache_dir=cache, pattern=r"\S+|\s+") != {"marker": 1}  # another pattern, another key
    path.write_text(urdu_text(20_000, seed=2), encoding="utf-8")
    assert count_words(path, 1, cache_dir=cache) not in ({"marker": 1}, first)  # another file, another key


# ---------------------------------------------------------------------------
# Checkpoints, heap rebuilds, limits, progress log
# ---------------------------------------------------------------------------

def test_checkpoint_and_resume(tmp_path: Path) -> None:
    counts = count_words(urdu_text(60_000))
    full = BPETokenizer()
    full.train(counts, 257 + 200, log_every=0)
    ck = tmp_path / "run.checkpoint.json"
    partial = BPETokenizer()
    partial.train(counts, 257 + 150, log_every=0, checkpoint_every=40, checkpoint_path=ck)
    saved = BPETokenizer.load(ck)
    assert len(saved.merges) == 120 and saved.meta["merges_done"] == 120
    resumed = BPETokenizer()
    summary = resumed.train(counts, 257 + 200, log_every=0, checkpoint_path=ck, resume=True, debug=10)
    assert summary["resumed_from"] == 120
    assert resumed.merges == full.merges and resumed.merge_counts == full.merge_counts


def test_resume_needs_a_matching_checkpoint(tmp_path: Path) -> None:
    counts = count_words("aaabdaaabac")
    other = BPETokenizer(pattern=r"\S+|\s+")
    other.train(counts, 260, log_every=0)
    ck = other.save(tmp_path / "other.json")
    with pytest.raises(ValueError):
        BPETokenizer().train(counts, 264, checkpoint_path=ck, resume=True)
    with pytest.raises(FileNotFoundError):
        BPETokenizer().train(counts, 264, checkpoint_path=tmp_path / "missing.json", resume=True)


def test_heap_rebuilds_do_not_change_the_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    text = urdu_text(40_000)
    plain = BPETokenizer()
    plain.train(count_words(text), 257 + 300, log_every=0)
    monkeypatch.setattr(bpe, "HEAP_SLACK", -10**9)  # rebuild after every merge
    rebuilt = BPETokenizer()
    summary = rebuilt.train(count_words(text), 257 + 300, log_every=0, debug=30)
    assert summary["heap_compactions"] == 300
    assert rebuilt.merges == plain.merges


def test_vocab_size_limits() -> None:
    with pytest.raises(ValueError):
        BPETokenizer().train({"ab": 1}, 256)
    with pytest.raises(ValueError):
        BPETokenizer().train({"ab": 1}, (1 << 16) + 1)


def test_progress_log() -> None:
    tok = BPETokenizer()
    tok.train(count_words(urdu_text(20_000)), 257 + 40, log_every=10)
    assert [e["merge"] for e in tok.train_log] == [10, 20, 30, 40]
    assert set(tok.train_log[0]) == {"merge", "count", "token", "seconds", "merges_per_s", "heap", "pairs",
                                     "peak_rss_mb"}


def test_train_tokenizer_cli_fast(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import train_tokenizer
    sample = tmp_path / "sample.txt"
    sample.write_text(urdu_text(30_000), encoding="utf-8")
    common = ["--input", str(sample), "--vocab-size", "400", "--out-dir", str(tmp_path / "tok"), "--show", "3"]
    assert train_tokenizer.main([*common, "--trainer", "reference", "--name", "ref"]) == 0
    fast_args = [*common, "--trainer", "fast", "--name", "fast", "--workers", "1", "--cache-dir",
                 str(tmp_path / "cache"), "--checkpoint-every", "50", "--log-every", "25"]
    assert train_tokenizer.main(fast_args) == 0
    assert "fast trainer: 143 merges" in capsys.readouterr().out
    ref, fast = BPETokenizer.load(tmp_path / "tok" / "ref.json"), BPETokenizer.load(tmp_path / "tok" / "fast.json")
    assert fast.merges == ref.merges and fast.merge_counts == ref.merge_counts
    assert fast.meta["distinct_words"] > 0 and len(fast.meta["train_log"]) == 5 and "count_seconds" in fast.meta
    assert len(BPETokenizer.load(tmp_path / "tok" / "fast.checkpoint.json").merges) == 100
    assert list((tmp_path / "cache").glob("word_counts_*.pkl"))
    assert train_tokenizer.main([*fast_args, "--resume"]) == 0
    resumed = BPETokenizer.load(tmp_path / "tok" / "fast.json")
    assert resumed.merges == ref.merges and resumed.meta["resumed_from"] == 100
