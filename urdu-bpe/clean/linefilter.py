"""Line language filter (Step 5): keep Urdu lines, including natural code-mixing.

classify_line(line) returns None to keep a line, or the reason to drop it.
The line must already be normalized (normalize.py): the word lists below are
in normalized spelling, and normalization keeps the Sindhi letters this
filter looks for. The rules, in order:

- no_letters: no letter of any script ("2008", "(54)", "……..", "———").
- mostly_not_urdu_script: fewer than half of the words that hold a letter
  hold an Arabic-script letter, and fewer than MIN_FUNCTION_WORDS_TO_RESCUE
  Urdu function words. English and other-script lines go; an Urdu sentence
  with English names in it ("وہ Barrow A.F.C.، … کے لیے کھیل چکے ہیں") stays,
  because its function words show Urdu grammar.
- sindhi: at least MIN_SINDHI_CHARS characters used by Sindhi but not by
  Urdu (ڪ ٽ ڏ ڀ ٻ ڄ ڃ ڇ ڻ ٿ ڦ …, and the signs ۾ ۽), making up at least
  MIN_SINDHI_SHARE of the line's Arabic-script letters.
- arabic_or_persian: at least MIN_LETTERS_FOR_FOREIGN Arabic-script letters,
  none of the letters only Urdu has (ٹ ڈ ڑ ں ے ۓ), no Urdu function word, and
  positive evidence: at least MIN_FOREIGN_WORDS Arabic or Persian function
  words (repeats count), or harakat on at least MIN_HARAKAT_SHARE of the letters
  (vowelled Quranic Arabic). Missing Urdu evidence alone is not enough: Urdu
  headlines and name lines often have neither function words nor ٹ ڈ ڑ ں ے.

Words are whitespace-separated (str.split), with leading and trailing
punctuation stripped before a function-word lookup. Thresholds and their
evidence: docs/decisions.md ("Step 5: line language filter").
"""

from collections import Counter

import regex

MIN_FUNCTION_WORDS_TO_RESCUE = 2
MIN_SINDHI_CHARS = 2
MIN_SINDHI_SHARE = 0.01
MIN_LETTERS_FOR_FOREIGN = 12
MIN_FOREIGN_WORDS = 2
MIN_HARAKAT_SHARE = 0.25

REASONS = ("no_letters", "mostly_not_urdu_script", "sindhi", "arabic_or_persian")


def _char_class(code_points: list[int]) -> regex.Pattern[str]:
    return regex.compile("[" + "".join(regex.escape(chr(cp)) for cp in code_points) + "]")


LETTER = regex.compile(r"\p{L}")
ARABIC_LETTER = regex.compile(r"[\p{Script=Arabic}&&\p{L}]", regex.V1)
EDGE_PUNCT = regex.compile(r"^\p{P}+|\p{P}+$")

# Letters Sindhi has and Urdu does not, plus the Sindhi postposition and ampersand signs.
SINDHI_CHARS = _char_class([
    0x067A, 0x067B, 0x067D, 0x067F, 0x0680, 0x0683, 0x0684, 0x0687, 0x068A, 0x068C,
    0x068D, 0x068F, 0x0699, 0x06A6, 0x06AA, 0x06B3, 0x06BB, 0x06FD, 0x06FE,
])
# Letters Urdu has and Arabic and Persian do not: tteh, ddal, rreh, noon ghunna, bari yeh (and with hamza).
URDU_ONLY_LETTERS = _char_class([0x0679, 0x0688, 0x0691, 0x06BA, 0x06D2, 0x06D3])
HARAKAT = _char_class([*range(0x064B, 0x0653), 0x0670])

URDU_FUNCTION_WORDS = frozenset("""
    کے کی کا ہے ہیں میں نے سے کو تھا تھی تھے ہو ہوں ہوا ہوئے رہا رہی رہے گیا گئی گئے
    نہیں اور بھی یہ وہ پر کر کہ جو ایک اس ان لیے لئے
""".split())
# Arabic and Persian function words (normalized spelling). Not علی (a common Urdu name), آن (Urdu "moment")
# or این (also the letter N in Urdu transliteration: سی این این).
FOREIGN_FUNCTION_WORDS = frozenset("""
    فی عن الی إلی قال قالت قالوا فقال کان کانت فلما الذی التی الذین ھذا ہذا ھذہ ہذہ اذا إذا لم لن قد ثم إن أن من
    است را از کند کرد شد بود نیست ہست برای
""".split())


def _bare(word: str) -> str:
    return EDGE_PUNCT.sub("", word)


def classify_line(line: str) -> str | None:
    """None to keep the line, or the drop reason (one of REASONS)."""
    if not LETTER.search(line):
        return "no_letters"
    words = line.split()
    script_words = [w for w in words if ARABIC_LETTER.search(w)]
    letter_words = sum(1 for w in words if LETTER.search(w))
    function_words = sum(_bare(w) in URDU_FUNCTION_WORDS for w in script_words)
    if len(script_words) < 0.5 * letter_words and function_words < MIN_FUNCTION_WORDS_TO_RESCUE:
        return "mostly_not_urdu_script"
    script_letters = len(ARABIC_LETTER.findall(line))
    sindhi = len(SINDHI_CHARS.findall(line))
    if sindhi >= MIN_SINDHI_CHARS and sindhi >= MIN_SINDHI_SHARE * script_letters:
        return "sindhi"
    if script_letters >= MIN_LETTERS_FOR_FOREIGN and function_words == 0 and not URDU_ONLY_LETTERS.search(line):
        foreign = sum(_bare(w) in FOREIGN_FUNCTION_WORDS for w in script_words)
        if foreign >= MIN_FOREIGN_WORDS or len(HARAKAT.findall(line)) >= MIN_HARAKAT_SHARE * script_letters:
            return "arabic_or_persian"
    return None


def filter_lines(text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Split normalized text into kept lines and (reason, line) pairs for the dropped ones."""
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for line in text.split("\n"):
        reason = classify_line(line)
        if reason is None:
            kept.append(line)
        else:
            dropped.append((reason, line))
    return kept, dropped


def dropped_bytes(dropped: list[tuple[str, str]]) -> Counter[str]:
    """UTF-8 bytes of the dropped lines, per reason (newlines not counted)."""
    out: Counter[str] = Counter()
    for reason, line in dropped:
        out[reason] += len(line.encode("utf-8"))
    return out
