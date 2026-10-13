"""Tests for bpe.py, Step 7: the pre-tokenizer, the reference trainer, encode/decode and save/load.

Several tests share one tokenizer trained with the reference trainer on
200 KB of tests/urdu_text.py text (500 merges, a few seconds). Run with -s to
see the GPT-2 pieces printed for the report.
"""

import json
import logging
import random
from collections import Counter
from pathlib import Path

import pytest
import regex

from bpe import (ENDOFTEXT, FORMAT, GPT2_PATTERN, URDU_PATTERN, BPETokenizer, count_pairs, merge_word, merges_path,
                 token_display)
from tests.urdu_text import urdu_text

URDU_TEXT_BYTES = 200_000
URDU_MERGES = 500


def random_string(rng: random.Random, max_len: int = 40) -> str:
    """Characters from every plane, weighted towards ASCII, Arabic, whitespace, marks and emoji."""
    out = []
    for _ in range(rng.randint(0, max_len)):
        r = rng.random()
        if r < 0.25:
            cp = rng.randint(0x20, 0x7E)
        elif r < 0.45:
            cp = rng.randint(0x0600, 0x06FF)
        elif r < 0.55:
            cp = rng.choice([0x20, 0x0A, 0x09, 0x0D, 0x0B, 0x0C, 0x85, 0xA0, 0x2028, 0x3000, 0x200C, 0x200D, 0xFE0F])
        elif r < 0.62:
            cp = rng.randint(0x0300, 0x036F)
        elif r < 0.70:
            cp = rng.randint(0x1F300, 0x1FAFF)
        else:
            cp = rng.randint(0, 0x10FFFF)  # any code point, lone surrogates included
        out.append(chr(cp))
    return "".join(out)


@pytest.fixture(scope="module")
def trained() -> tuple[BPETokenizer, str]:
    text = urdu_text(URDU_TEXT_BYTES)
    tok = BPETokenizer()
    tok.train_reference(text, 257 + URDU_MERGES)
    assert len(tok.merges) == URDU_MERGES
    return tok, text


# ---------------------------------------------------------------------------
# Pre-tokenizer
# ---------------------------------------------------------------------------

def test_urdu_words_stay_whole() -> None:
    tok = BPETokenizer()
    assert tok.pretokenize("اُردو") == ["اُردو"]
    assert tok.pretokenize("اُردُو") == ["اُردُو"]
    assert tok.pretokenize(" اُردو زبان") == [" اُردو", " زبان"]
    assert tok.pretokenize("صورت‌حال") == ["صورت‌حال"]
    assert tok.pretokenize("یہ صورت‌حال ہے") == ["یہ", " صورت‌حال", " ہے"]


def test_digits_and_punctuation() -> None:
    tok = BPETokenizer()
    assert tok.pretokenize("2024ء") == ["202", "4", "ء"]
    assert tok.pretokenize("1234567") == ["123", "456", "7"]
    assert tok.pretokenize(" 12 روپے") == [" 12", " روپے"]
    assert tok.pretokenize("یہ ہے۔") == ["یہ", " ہے", "۔"]
    assert tok.pretokenize("کیا؟ ہاں، ٹھیک!") == ["کیا", "؟", " ہاں", "،", " ٹھیک", "!"]


def test_whitespace_and_contractions() -> None:
    tok = BPETokenizer()
    assert tok.pretokenize("a  b") == ["a", " ", " b"]
    assert tok.pretokenize("a \n b") == ["a", " \n", " b"]
    assert tok.pretokenize("a\n\n b") == ["a", "\n\n", " b"]
    assert tok.pretokenize("it's DON'T") == ["it", "'s", " DON", "'T"]
    assert tok.pretokenize("") == []


def test_pretokenize_covers_every_character() -> None:
    tok = BPETokenizer()
    rng = random.Random(1337)
    for _ in range(5000):
        s = random_string(rng)
        assert "".join(tok.pretokenize(s)) == s, repr(s)


def test_gpt2_pattern_cuts_urdu_words() -> None:
    """The handout: GPT-2's pattern cuts اُردُو into five pieces at its diacritics. Printed for the report."""
    gpt2 = regex.compile(GPT2_PATTERN)
    handout = gpt2.findall("اُردُو")
    assert handout == ["ا", "ُ", "رد", "ُ", "و"]
    print(f"\nGPT-2 pattern on اُردُو: {len(handout)} pieces {handout}")
    for word in ("اُردو", " اُردو زبان", "صورت‌حال"):
        print(f"GPT-2 pattern on {word!r}: {gpt2.findall(word)}; ours: {regex.findall(URDU_PATTERN, word)}")
    assert gpt2.findall("صورت‌حال") == ["صورت", "‌", "حال"]


# ---------------------------------------------------------------------------
# Base vocabulary
# ---------------------------------------------------------------------------

def test_base_vocabulary() -> None:
    tok = BPETokenizer()
    assert all(tok.vocab[i] == bytes([i]) for i in range(256))
    assert tok.vocab[256] == b"<|endoftext|>" and tok.special_tokens == {ENDOFTEXT: 256}
    assert tok.vocab_size == 257 and tok.merges == [] and tok.first_merge_id == 257


def test_kitab_is_eight_bytes_without_merges() -> None:
    tok = BPETokenizer()
    assert tok.encode("کتاب") == [218, 169, 216, 170, 216, 167, 216, 168]
    assert tok.decode([218, 169, 216, 170, 216, 167, 216, 168]) == "کتاب"


def test_special_token_ids_must_follow_the_bytes() -> None:
    with pytest.raises(ValueError):
        BPETokenizer(special_tokens={"<|x|>": 300})


# ---------------------------------------------------------------------------
# Reference trainer
# ---------------------------------------------------------------------------

AAAB = [b"aa", b"aaa", b"aaab", b"daaab", b"daaaba", b"daaabac", b"aaabdaaabac"]


def test_tie_break_aaabdaaabac() -> None:
    """The reference trainer's sanity check (the handout names one but does not print it): the classic BPE
    example "aaabdaaabac", trained to the end, must give these seven merges in this order. Merge 4 is a
    four-way tie that only the spec's bytes comparison resolves this way."""
    tok = BPETokenizer()
    tok.train_reference("aaabdaaabac", vocab_size=257 + 7)
    assert [tok.vocab[257 + i] for i in range(7)] == AAAB
    assert tok.encode("aaabdaaabac") == [263]
    # Merge 4 is a four-way tie at count 1. Comparing by ID would pick (259, 100) = "aaabd".
    before = BPETokenizer()
    before.train_reference("aaabdaaabac", vocab_size=257 + 3)
    state = before.encode("aaabdaaabac")
    assert state == [259, 100, 259, 97, 99]
    counts = count_pairs([state])
    assert set(counts.values()) == {1}
    assert max(counts) == (259, 100)
    assert tok.merges[3] == (100, 259)


def test_stops_with_a_warning_when_no_pair_is_left(caplog: pytest.LogCaptureFixture) -> None:
    tok = BPETokenizer()
    with caplog.at_level(logging.WARNING, logger="bpe"):
        summary = tok.train_reference("aaabdaaabac", vocab_size=257 + 8)
    assert len(tok.merges) == 7 and summary["merges"] == 7
    assert [tok.vocab[257 + i] for i in range(7)] == AAAB
    assert "no pair left after 7 merges" in caplog.text


def test_overlapping_pairs() -> None:
    assert count_pairs([[97, 97, 97, 97]])[(97, 97)] == 3
    assert count_pairs([[97, 97, 97], [98], []]) == Counter({(97, 97): 2})
    assert merge_word([97, 97, 97], (97, 97), 257) == [257, 97]
    assert merge_word([97, 97, 97, 97], (97, 97), 257) == [257, 257]
    assert merge_word([98, 97, 97, 98], (97, 97), 257) == [98, 257, 98]
    tok = BPETokenizer()
    tok.train_reference("aaaa", vocab_size=258)
    assert tok.merges == [(97, 97)] and tok.merge_counts == [3]
    assert tok.encode("aaaa") == [257, 257]


def test_vocab_size_below_the_base_is_refused() -> None:
    with pytest.raises(ValueError):
        BPETokenizer().train_reference("abc", vocab_size=256)


def test_special_tokens_take_no_part_in_training(caplog: pytest.LogCaptureFixture) -> None:
    tok = BPETokenizer()
    tok.train_reference(ENDOFTEXT * 100 + "ab", vocab_size=258)
    assert tok.merges == [(97, 98)]
    with caplog.at_level(logging.WARNING, logger="bpe"):
        tok.train_reference(ENDOFTEXT * 100 + "ab", vocab_size=260)
    assert tok.merges == [(97, 98)] and "no pair left after 1 merges" in caplog.text
    # Pieces between separators are single characters, so there is nothing to merge across a separator.
    tok.train_reference(("x" + ENDOFTEXT) * 50 + "y", vocab_size=300)
    assert tok.merges == []


def test_training_starts_from_scratch() -> None:
    tok = BPETokenizer()
    tok.train_reference("aaabdaaabac", vocab_size=257 + 7)
    tok.train_reference("xyxyxy", vocab_size=258)
    assert tok.merges == [(120, 121)] and tok.vocab_size == 258 and 258 not in tok.vocab


def test_pairs_with_identical_bytes_fall_back_to_ids(caplog: pytest.LogCaptureFixture) -> None:
    tok = BPETokenizer()
    tok._add_merge(97, 98)    # 257 ab
    tok._add_merge(98, 99)    # 258 bc
    tok._add_merge(257, 99)   # 259 abc
    tok._add_merge(97, 258)   # 260 abc again, built the other way
    assert tok.vocab[259] == tok.vocab[260] == b"abc"
    with caplog.at_level(logging.WARNING, logger="bpe"):
        pair, count = tok._choose(Counter({(259, 100): 2, (260, 100): 2, (97, 98): 1}))
    assert (pair, count) == ((260, 100), 2)
    assert "same bytes" in caplog.text


# ---------------------------------------------------------------------------
# Encode and decode
# ---------------------------------------------------------------------------

def test_special_tokens_in_encode() -> None:
    tok = BPETokenizer()
    tok.train_reference(ENDOFTEXT * 100 + "ab", vocab_size=258)
    assert tok.encode("x<|endoftext|>y") == [120, 256, 121]
    assert tok.encode("x<|endoftext|>y", allowed_special={ENDOFTEXT}) == [120, 256, 121]
    plain = tok.encode("x<|endoftext|>y", allowed_special="none")
    assert 256 not in plain and plain == tok.encode_ordinary("x<|endoftext|>y")
    assert plain == tok.encode("x<|endoftext|>y", allowed_special=set())
    assert tok.decode(plain) == tok.decode([120, 256, 121]) == "x<|endoftext|>y"
    assert tok.encode("ab<|endoftext|><|endoftext|>ab") == [257, 256, 256, 257]
    with pytest.raises(ValueError):
        tok.encode("x", allowed_special={"<|unknown|>"})
    with pytest.raises(ValueError):
        tok.encode("x", allowed_special="some")  # type: ignore[arg-type]


def test_decode_falls_back_when_a_character_is_cut() -> None:
    tok = BPETokenizer()
    assert tok.decode_bytes([0xD8]) == b"\xd8"
    assert tok.decode([0xD8]) == "�"
    assert tok.decode([0xD8, 0xA7, 0xD8]) == "ا�"
    with pytest.raises(ValueError):
        tok.decode([10**6])


def test_lone_surrogates_round_trip() -> None:
    tok = BPETokenizer()
    for s in ("\ud800", "a\udfffb", "😀", "اردو\udc80"):
        assert tok.decode(tok.encode(s)) == s


def test_encode_replays_the_merges_in_order(trained: tuple[BPETokenizer, str]) -> None:
    """Lowest merge ID first, per word, gives what applying every merge in learned order gives."""
    tok, text = trained
    words = {w for part in text.split(ENDOFTEXT) for w in tok.pretokenize(part)}
    for word in sorted(words):
        replay = list(word.encode("utf-8"))
        for k, (a, b) in enumerate(tok.merges):
            if a in replay:
                replay = merge_word(replay, (a, b), 257 + k)
        assert list(tok.encode_ordinary(word)) == replay, word


ROUND_TRIP = [
    "", "\n", "  \t\n\n", " ", "\r\n", "Hello, world! It's 2024 and we're testing.",
    "مُحَبَّت اُردُو زبان کی پہچان ہے۔", "یہ صورت‌حال دل‌چسپ ہے؟",
    "👨‍👩‍👧‍👦 ❤️ ☺️ 🏳️‍🌈", "汉字测试，中文。", "\ud800", ENDOFTEXT,
    "اردو<|endoftext|>English", "<|endoftext", "1234567890 ۱۲۳ ١٢٣", "\x00\x01\x7f\u0085 ﻿",
]


def test_round_trip_fixed_strings(trained: tuple[BPETokenizer, str]) -> None:
    tok, _ = trained
    for s in ROUND_TRIP:
        assert tok.decode(tok.encode(s)) == s, repr(s)
        assert tok.decode(tok.encode(s, allowed_special="none")) == s, repr(s)
    assert tok.encode(ENDOFTEXT) == [256]


def test_round_trip_random_strings(trained: tuple[BPETokenizer, str]) -> None:
    tok, _ = trained
    rng = random.Random(2024)
    for _ in range(2000):
        s = random_string(rng, max_len=60)
        assert tok.decode(tok.encode(s)) == s, repr(s)


def test_trained_tokenizer_learns_urdu(trained: tuple[BPETokenizer, str]) -> None:
    tok, text = trained
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert len(ids) < 0.3 * len(text.encode("utf-8"))
    assert ids.count(256) == text.count(ENDOFTEXT)
    first = {tok.vocab[257 + i] for i in range(10)}
    assert {"ا".encode(), "ی".encode()} <= first  # two-byte letters are rebuilt first
    assert " ہے".encode() in tok.vocab.values()


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------

def test_token_display() -> None:
    assert token_display("ک".encode()) == "ک"
    assert token_display(b" \xda") == "␣\\xda"
    assert token_display(" ہے".encode()) == "␣ہے"
    assert token_display(b"\n\n\t") == "\\n\\n\\t"
    assert token_display(b"a\\b") == "a\\\\b"
    assert token_display("‌".encode()) == "\\u200c"
    assert token_display(b"\xed\xa0\x80") == "\\xed\\xa0\\x80"  # an encoded lone surrogate is not valid UTF-8


def test_merges_file(tmp_path: Path) -> None:
    tok = BPETokenizer()
    tok.train_reference(" ک ک ک گ", vocab_size=258)
    assert tok.merges == [(32, 218)]
    path = tok.save(tmp_path / "tiny.json", meta={"note": "test"})
    lines = merges_path(path).read_text(encoding="utf-8").splitlines()
    assert merges_path(path).name == "tiny.merges.txt"
    assert lines[0].startswith("# rank\tnew_id\tleft_id\tright_id\ttoken_hex\ttoken")
    assert lines[1:] == ["1\t257\t32\t218\t20da\t␣\\xda"]


def test_load_rejects_bad_files(tmp_path: Path) -> None:
    tok = BPETokenizer()
    tok.train_reference("aaabdaaabac", vocab_size=257 + 7)
    path = tok.save(tmp_path / "t.json")
    good = json.loads(path.read_text(encoding="utf-8"))
    for change in ({"format": "other"}, {"num_merges": 6}, {"vocab_size": 300}, {"merges": good["merges"][:-1]},
                   {"merges": [[97, 97], [999, 97]], "num_merges": 2, "vocab_size": 259}):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(good | change), encoding="utf-8")
        with pytest.raises(ValueError):
            BPETokenizer.load(bad)


def test_save_and_load(trained: tuple[BPETokenizer, str], tmp_path: Path) -> None:
    tok, text = trained
    path = tok.save(tmp_path / "urdu_500.json", meta={"trainer": "reference"})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == FORMAT and data["pattern"] == URDU_PATTERN
    assert data["special_tokens"] == {ENDOFTEXT: 256}
    assert data["vocab_size"] == 257 + URDU_MERGES and data["num_merges"] == URDU_MERGES
    assert data["merges"] == [list(m) for m in tok.merges]
    assert data["meta"]["trainer"] == "reference" and len(data["meta"]["merge_counts"]) == URDU_MERGES
    assert len(merges_path(path).read_text(encoding="utf-8").splitlines()) == 1 + URDU_MERGES
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges and loaded.vocab == tok.vocab
    assert loaded.merge_ranks == tok.merge_ranks and loaded.merge_counts == tok.merge_counts
    assert loaded.special_tokens == tok.special_tokens and loaded.pattern == tok.pattern
    assert loaded.meta == {"trainer": "reference"}
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.save(tmp_path / "again.json").read_bytes() == path.read_bytes()
