"""Tests for normalize.normalize (Step 4). Look-alike letters are written as escapes on purpose."""

import random
import unicodedata

import pytest

from normalize import KEPT_LIGATURES, LETTERS, PRESENTATION, PRESENTATION_AND_REMOVED, REMOVED, SPACES, normalize

# Urdu letters used to build inputs.
ALEF, BEH, TEH, DAL, REH, SEEN, LAM, MEEM, NOON, WAW = "\u0627", "\u0628", "\u062A", "\u062F", "\u0631", "\u0633", "\u0644", "\u0645", "\u0646", "\u0648"
HEH_GOAL, DO_CHASHMI, YEH, BARI_YEH, KEHEH = "\u06C1", "\u06BE", "\u06CC", "\u06D2", "\u06A9"
FULL_STOP, COMMA, QUESTION, SEMICOLON = "\u06D4", "\u060C", "\u061F", "\u061B"
ZWNJ = "\u200C"


@pytest.mark.parametrize("raw, expected", [
    # Handout table: yeh, kaf, heh.
    ("\u064A", YEH),
    ("\u0649", YEH),
    (ALEF + "\u0639\u0644\u0649\u0670", ALEF + "\u0639\u0644" + YEH + "\u0670"),
    ("\u0643", KEHEH),
    ("\u0647", HEH_GOAL),
    (DO_CHASHMI, DO_CHASHMI),
    (HEH_GOAL, HEH_GOAL),
    (TEH + DO_CHASHMI + ALEF, TEH + DO_CHASHMI + ALEF),
    # Further heh-like letters.
    ("\u06D5", HEH_GOAL),
    ("\u0629", "\u06C3"),
    ("\u06C3", "\u06C3"),
    ("\u06C0", "\u06C2"),
    # Sindhi letters stay, so the language filter can see them.
    ("\u06AA\u067D\u068F", "\u06AA\u067D\u068F"),
])
def test_letters(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    (LAM + "\u0640" + ALEF, LAM + ALEF),
    ("\u0640\u0640", ""),
    ("a\u200Db", "ab"),
    ("a\u200Bb", "ab"),
    ("a\u2060b", "ab"),
    ("a\u200E\u200F\u061Cb", "ab"),
    ("\u202B" + BEH + "\u202C", BEH),
    ("\u2067" + BEH + "\u2069", BEH),
    ("\uFEFF" + BEH, BEH),
    ("a\u00ADb", "ab"),
    ("a\x00b\x07c\x7fd\x9fe", "abcde"),
    ("\uFFFD", "\uFFFD"),  # a bad decode upstream is not ours to hide
])
def test_removed_characters(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    (SEEN + WAW + REH + TEH + ZWNJ + "\u062D" + ALEF + LAM, SEEN + WAW + REH + TEH + ZWNJ + "\u062D" + ALEF + LAM),
    (TEH + ZWNJ + ZWNJ + "\u062D", TEH + ZWNJ + "\u062D"),
    (LAM + "\u0650" + ZWNJ + DAL, LAM + "\u0650" + ZWNJ + DAL),  # a mark after the first letter
    (LAM + ZWNJ + "\u0650", LAM + "\u0650"),  # never in front of a mark
    (BEH + " " + ZWNJ + " " + BEH, BEH + " " + BEH),
    (BEH + ZWNJ + "\n" + BEH, BEH + "\n" + BEH),
    (FULL_STOP + ZWNJ + BEH, FULL_STOP + BEH),
    (ZWNJ + BEH, BEH),
    (BEH + ZWNJ, BEH),
    ("a" + ZWNJ + "b", "ab"),
    ("1" + ZWNJ + "2", "12"),
    (TEH + ZWNJ + "\u200E" + "\u062D", TEH + ZWNJ + "\u062D"),  # removed invisibles are gone first
])
def test_zwnj(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    (ALEF + "\u0653", "\u0622"),
    (WAW + "\u0654", "\u0624"),
    (HEH_GOAL + "\u0654", "\u06C2"),
    (BARI_YEH + "\u0654", "\u06D3"),
    ("\u064A\u0654", "\u0626"),  # Arabic yeh + hamza is yeh-hamza, not Urdu yeh + loose hamza
    ("\u0647\u0654", "\u06C2"),  # Arabic heh + hamza, after mapping to gol heh
    ("\u06D5\u0654", "\u06C2"),
    (YEH + "\u0654", YEH + "\u0654"),  # no precomposed form: left as typed
    (BEH + "\u0651\u064E", BEH + "\u064E\u0651"),  # canonical order of stacked harakat
    (ALEF + "\u200E\u0653", "\u0622"),  # an invisible between letter and madda is removed first
    (ALEF + "\u0640\u0653", "\u0622"),
])
def test_composition(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("\uFE8E", ALEF),
    ("\uFE8D", ALEF),
    ("\uFB90", KEHEH),
    ("\uFBFD", YEH),
    ("\uFBAF", BARI_YEH),
    ("\uFEFB", LAM + ALEF),
    ("\uFE85", "\u0624"),
    ("\uFE91\uFE8E", BEH + ALEF),
    ("\uFDF2", ALEF + LAM + LAM + HEH_GOAL),
    (BEH + "\uFE70", BEH + "\u064B"),  # isolated fathatan: no stray space
    (BEH + "\uFE71", BEH + "\u064B"),  # medial fathatan: no stray tatweel
    ("\uFDFA", "\uFDFA"),
    ("\uFDFB", "\uFDFB"),
    ("\uFD3E" + BEH + "\uFD3F", "\uFD3E" + BEH + "\uFD3F"),  # ornate parentheses have no decomposition
])
def test_presentation_forms(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


def test_presentation_table_covers_letters_only_by_position() -> None:
    assert len(PRESENTATION) > 700
    assert not KEPT_LIGATURES & {chr(cp) for cp in PRESENTATION}
    for cp, value in PRESENTATION.items():
        assert value and not value.startswith((" ", "\u0640")), hex(cp)
        assert not any(0xFB50 <= ord(c) <= 0xFEFE for c in value), hex(cp)



def test_merged_table_is_equivalent_to_two_passes() -> None:
    assert not PRESENTATION.keys() & REMOVED.keys()
    assert not any(ord(c) in REMOVED for value in PRESENTATION.values() for c in value)
    assert PRESENTATION_AND_REMOVED == {**PRESENTATION, **REMOVED}


def test_tables_do_not_overlap() -> None:
    assert not LETTERS.keys() & PRESENTATION_AND_REMOVED.keys()
    assert not SPACES.keys() & PRESENTATION_AND_REMOVED.keys()
    assert not SPACES.keys() & LETTERS.keys()

@pytest.mark.parametrize("raw, expected", [
    ("\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9\u06F0", "1234567890"),
    ("\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669\u0660", "1234567890"),
    ("2020", "2020"),
    ("\u06F1\u06F2\u066B\u06F5", "12.5"),
    ("1\u066C000", "1,000"),
    ("\u066B", "\u066B"),  # not between digits: left alone
    (BEH + "\u066C" + BEH, BEH + "\u066C" + BEH),
    ("50\u066A", "50\u066A"),
])
def test_digits(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("a\u00A0b", "a b"),
    ("a\u00A0\u00A0 b", "a b"),
    ("a\u202Fb\u2009c\u3000d", "a b c d"),
    ("a\tb", "a b"),
    ("a\x0bb\x0cc", "a b c"),
    ("a   b", "a b"),
    ("a\r\nb\rc", "a\nb\nc"),
    ("a\u2028b\u2029c\x85d", "a\nb\nc\nd"),
    ("  a  \n  b  ", "a\nb"),
    ("a\n\n\n\nb", "a\nb"),
    ("\n\na\n\n", "a"),
    (" \n \t \n", ""),
    ("", ""),
    (BEH + " " + FULL_STOP, BEH + FULL_STOP),
    (BEH + "   " + COMMA + " " + BEH, BEH + COMMA + " " + BEH),
    (BEH + " " + QUESTION + " " + BEH + " " + SEMICOLON, BEH + QUESTION + " " + BEH + SEMICOLON),
    (BEH + "\n " + FULL_STOP, BEH + "\n" + FULL_STOP),  # never joins lines
    (BEH + " .", BEH + " ."),  # ASCII punctuation is left as typed
])
def test_whitespace(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize("text", [
    "Pakistan " + KEHEH + ALEF + " capital",
    "\U0001F44D\U0001F3FD " + BEH,
    "a<|endoftext|>b",
    "\u2018" + BEH + "\u2019 \u201C" + BEH + "\u201D",
    BEH + "\u0650 " + "\u0631\u064F",
    "\u0610\u0611\u0612\u0613\u0614",
    "\u0627\u0644\u0644\u0651\u0670\u06C1",
])
def test_left_alone(text: str) -> None:
    assert normalize(text) == text


ALPHABET = (
    "ab 1\n\t\r\x00\x85" + ALEF + BEH + TEH + LAM + HEH_GOAL + DO_CHASHMI + YEH + BARI_YEH + WAW + KEHEH
    + "\u064A\u0649\u0643\u0647\u06D5\u0629\u06C0\u06AA\u0640\u200C\u200D\u200B\u200E\u202B\uFEFF"
    + "\u0653\u0654\u0650\u0651\u064E\u0670\u06F1\u0662\u066B\u066C\u00A0\u2028\u3000"
    + FULL_STOP + COMMA + QUESTION + "\uFE8E\uFEFB\uFDF2\uFDFA\uFE70\uFE71\uFE85"
)


def random_strings(n: int, seed: int = 1337) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(ALPHABET) for _ in range(rng.randrange(0, 25))) for _ in range(n)]


@pytest.mark.parametrize("text", random_strings(5000))
def test_idempotent_and_invariants(text: str) -> None:
    out = normalize(text)
    assert normalize(out) == out
    assert unicodedata.is_normalized("NFC", out)
    assert not any(ord(c) in REMOVED for c in out)
    assert not any(ord(c) in LETTERS for c in out)
    assert not any(ord(c) in PRESENTATION for c in out)
    assert "\r" not in out and "  " not in out and "\n\n" not in out and "\t" not in out
    assert out == out.strip()
    for line in out.split("\n"):
        assert line == line.strip(" ")
