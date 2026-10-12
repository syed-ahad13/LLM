"""Tests for clean/linefilter.py (Step 5 line language filter). Inputs are normalized text."""

import pytest

from clean.linefilter import (FOREIGN_FUNCTION_WORDS, REASONS, URDU_FUNCTION_WORDS, classify_line, dropped_bytes,
                              filter_lines)
from normalize import normalize

FATHA, KASRA, SHADDA = "\u064E", "\u0650", "\u0651"
SWASH_KAF, TTEH_SINDHI, DDAL_SINDHI, BHEH = "\u06AA", "\u067D", "\u068F", "\u0680"


def vowelled(text: str) -> str:
    """Put a fatha on every Arabic-script letter, like vowelled Quranic text."""
    return "".join(c + FATHA if "\u0621" <= c <= "\u06FF" else c for c in text)


@pytest.mark.parametrize("line", ["2008", "(54)", "\u2026\u2026..", "\u2014\u2014\u2014", "\u06D4", "- 31/03/2021"])
def test_no_letters(line: str) -> None:
    assert classify_line(line) == "no_letters"


@pytest.mark.parametrize("line", [
    "So we have whole 18 hours to re enact movie The Terminal",
    "twitter.com",
    "Eretnids (1335\u20131390)",
    "Dar al Kuti سلطان (1830\u20131912)",
    "\u09AC\u09BE\u0982\u09B2\u09BE",  # Bengali
    "Abdul ka Amrika - Journey of a Pakistani student",  # Roman Urdu is not Urdu script
])
def test_mostly_not_urdu_script(line: str) -> None:
    assert classify_line(line) == "mostly_not_urdu_script"


@pytest.mark.parametrize("line", [
    "وہ Barrow A.F.C.، Birmingham City F.C.، Kidderminster Harriers F.C. اور Northampton Town F.C. کے لیے کھیل چکے ہیں۔",
    "Pakistan کا capital اسلام آباد ہے",
    "ویب سائٹ: www.portmoody.ca",
    "یہ ایک عام اردو جملہ ہے۔",
    "محسنِ پاکستان ڈاکٹر عبد القدیر خان",  # headline: no function word, but Urdu-only letters
    "امریکہ: جنگی طیارہ رہائشی عمارات پہ گرکر تباہ",  # no function word and no Urdu-only letter
    "طلبہ و طالبات پریشان، شدید احتجاج",
    "سی این این بزنس",  # not Persian: CNN
    "استاد حامد علی خان بمقابلہ استاد پرویز مشرف علی خان",  # not Arabic: the name Ali
    "پا" + SWASH_KAF + "ستان ایک ملک ہے",  # one stray Sindhi letter is not Sindhi
    "اعلیٰ",
])
def test_kept(line: str) -> None:
    assert classify_line(line) is None


@pytest.mark.parametrize("line", [
    "سوشل میڈیا تی پوسٹ ٿیڻ وارین وڊیوز " + "\u06FE" + " یو" + SWASH_KAF + "رین جی مختلف شہرن",
    "پا" + SWASH_KAF + "ستان جی پر" + DDAL_SINDHI + "یہی وزیر",
    BHEH + "ائر " + TTEH_SINDHI + "ی",
])
def test_sindhi(line: str) -> None:
    assert classify_line(line) == "sindhi"


@pytest.mark.parametrize("line", [
    vowelled("ان مثل عیسی عند اللہ کمثل آدم خلقہ من تراب"),
    "عن عائشۃ قالت کان رسول اللہ جالسا فسمعنا لغطا",
    "فی الخندق و فی الخیبر فی البدر واحد",  # repeats count
    "ما کار خویش را بخداوند کار ساز بسپردہ ایم تا کرم او چہ ہا کند",  # Persian
])
def test_arabic_or_persian(line: str) -> None:
    assert classify_line(line) == "arabic_or_persian"


@pytest.mark.parametrize("line", [
    vowelled("قال ہو"),  # under MIN_LETTERS_FOR_FOREIGN letters
    vowelled("ان مثل عیسی عند اللہ") + " کے",  # an Urdu function word
    vowelled("ان مثل عیسی عند اللہ کمثل") + " \u06D2",  # an Urdu-only letter
    "اور" + KASRA + SHADDA + " " + "مثل عیسی عند اللہ کمثل آدم",
])
def test_not_arabic_without_evidence_or_with_urdu_evidence(line: str) -> None:
    assert classify_line(line) != "arabic_or_persian"


def test_word_lists_are_normalized_and_disjoint() -> None:
    for word in URDU_FUNCTION_WORDS | FOREIGN_FUNCTION_WORDS:
        assert normalize(word) == word, word
    assert not URDU_FUNCTION_WORDS & FOREIGN_FUNCTION_WORDS


def test_function_words_ignore_edge_punctuation() -> None:
    assert classify_line("Hello world, this is a test کے، ہے۔") is None


def test_filter_lines_and_dropped_bytes() -> None:
    text = "یہ اردو ہے۔\n2008\nEnglish only here\nوہ بھی اردو ہے"
    kept, dropped = filter_lines(text)
    assert kept == ["یہ اردو ہے۔", "وہ بھی اردو ہے"]
    assert dropped == [("no_letters", "2008"), ("mostly_not_urdu_script", "English only here")]
    assert dropped_bytes(dropped) == {"no_letters": 4, "mostly_not_urdu_script": 17}
    assert set(r for r, _ in dropped) <= set(REASONS)
