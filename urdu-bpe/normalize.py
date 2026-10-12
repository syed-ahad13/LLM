"""Urdu text normalization: normalize(text) -> str.

Applied while cleaning the corpus (clean/ pipeline), never inside the tokenizer.
Every rule is our own code over the standard library's Unicode tables
(unicodedata) and `regex`; no Urdu NLP library is used. The evidence for each
rule is in data/stats/char_audit.json (clean/audit_chars.py) and the decision
in docs/decisions.md ("Step 4: Urdu normalization rules").

Steps, in this order (the order matters where noted):

1. Line breaks: CR LF, CR, NEL, U+2028 and U+2029 become LF.
2. Arabic presentation forms (U+FB50-U+FDFF, U+FE70-U+FEFE) become the letters
   they display, from their Unicode compatibility decompositions; the Allah
   ligature U+FDF2 becomes alef lam lam heh (then gol heh in step 6). The
   phrase ligatures U+FDFA (salla Allahu alayhi wa sallam) and U+FDFB (jalla
   jalaluhu) stay as they are: they do not look like the typed phrase. A
   leading space or tatweel in a decomposition (isolated or medial harakat
   forms) is dropped, so the mark attaches to the letter before it.
3. Removed: tatweel, ZWJ, ZWSP, word joiner and invisible operators,
   LRM/RLM/ALM, bidi embeddings, overrides and isolates, BOM, soft hyphen,
   Mongolian vowel separator, and C0/C1 controls other than tab, LF, VT, FF.
   Before NFC, so a letter and a combining mark split by one of these compose.
4. ZWNJ (U+200C) is kept only between two Arabic-script letters (a run becomes
   one ZWNJ), where Urdu uses it on purpose, as in surat<ZWNJ>hal; anywhere
   else it is removed. Before NFC, for the same reason as step 3.
5. NFC: composes alef + madda into U+0622, waw + hamza above into U+0624, gol
   heh + hamza above into U+06C2, and so on, and puts stacked harakat in
   canonical order. Before step 6, so Arabic yeh + hamza above becomes U+0626
   instead of an Urdu yeh with a loose hamza.
6. Letters and digits:
   - Arabic yeh U+064A and alef maksura U+0649 -> Urdu yeh U+06CC;
     Arabic kaf U+0643 -> keheh U+06A9. Swash kaf U+06AA is a Sindhi letter
     and is left alone, so the language filter can still recognise Sindhi.
   - Arabic heh U+0647 and ae U+06D5 -> gol heh U+06C1; teh marbuta U+0629 ->
     teh marbuta goal U+06C3; heh with yeh above U+06C0 -> heh goal with hamza
     above U+06C2. Do-chashmi heh U+06BE is a different letter and is kept.
   - Arabic-Indic (U+0660-0669) and Urdu (U+06F0-06F9) digits -> ASCII 0-9.
   Then NFC again, so Arabic heh + hamza above, now gol heh + hamza, becomes
   U+06C2.
7. Between two digits, the Arabic decimal separator U+066B -> "." and the
   thousands separator U+066C -> ",".
8. Whitespace: no-break and other Unicode spaces, tab, VT and FF become a
   space; a space before the Urdu full stop, comma, question mark or
   semicolon (U+06D4, U+060C, U+061F, U+061B) is removed; runs of spaces
   become one; lines are trimmed; empty lines are dropped. The result has no
   leading or trailing whitespace.

normalize() is idempotent: normalize(normalize(s)) == normalize(s).
"""

import re
import unicodedata
from itertools import chain

import regex

ZWNJ = "\u200C"
TATWEEL = "\u0640"
KEPT_LIGATURES = frozenset("\uFDFA\uFDFB")
PRESENTATION_RANGES = (range(0xFB50, 0xFE00), range(0xFE70, 0xFEFF))
POSITIONAL_TAGS = frozenset({"<isolated>", "<initial>", "<medial>", "<final>"})


def _presentation_table() -> dict[int, str]:
    """Presentation form -> the characters it displays (its positional compatibility decomposition)."""
    table: dict[int, str] = {}
    for cp in chain(*PRESENTATION_RANGES):
        ch = chr(cp)
        decomposition = unicodedata.decomposition(ch).split()
        if ch in KEPT_LIGATURES or not decomposition or decomposition[0] not in POSITIONAL_TAGS:
            continue
        chars = [chr(int(h, 16)) for h in decomposition[1:]]
        if len(chars) > 1 and chars[0] in (" ", TATWEEL):
            chars = chars[1:]
        table[cp] = "".join(chars)
    return table


PRESENTATION = _presentation_table()

REMOVED: dict[int, None] = dict.fromkeys(chain(
    [0x0640],                                                  # tatweel
    [0x200D, 0x200B, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064],  # ZWJ, ZWSP, word joiner, invisible operators
    [0x200E, 0x200F, 0x061C],                                  # LRM, RLM, ALM
    range(0x202A, 0x202F),                                     # LRE, RLE, PDF, LRO, RLO
    range(0x2066, 0x206A),                                     # LRI, RLI, FSI, PDI
    [0xFEFF, 0x00AD, 0x180E],                                  # BOM, soft hyphen, Mongolian vowel separator
    (cp for cp in chain(range(0x00, 0x20), range(0x7F, 0xA0)) if cp not in (0x09, 0x0A, 0x0B, 0x0C)),
))

LETTERS: dict[int, str] = {
    0x064A: "\u06CC", 0x0649: "\u06CC",         # Arabic yeh, alef maksura -> Urdu yeh
    0x0643: "\u06A9",                           # Arabic kaf -> keheh
    0x0647: "\u06C1", 0x06D5: "\u06C1",         # Arabic heh, ae -> gol heh
    0x0629: "\u06C3",                           # teh marbuta -> teh marbuta goal
    0x06C0: "\u06C2",                           # heh with yeh above -> heh goal with hamza above
    **{0x0660 + d: str(d) for d in range(10)},  # Arabic-Indic digits
    **{0x06F0 + d: str(d) for d in range(10)},  # Extended Arabic-Indic (Urdu) digits
}

SPACES: dict[int, str] = dict.fromkeys(
    [0x09, 0x0B, 0x0C, 0x00A0, 0x1680, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000], " ")

# Steps 2 and 3 in one pass: the key sets are disjoint and no presentation form maps to a removed character.
PRESENTATION_AND_REMOVED: dict[int, str | None] = {**PRESENTATION, **REMOVED}


def _any_of(table: dict[int, object]) -> re.Pattern[str]:
    """A pattern that finds any key of a translate table, as a class of ranges.

    Searching first and translating only on a hit is much cheaper, since most
    documents hold no key at all. The stdlib `re` scans a class of ranges about
    15 times faster than `regex` (measured on Makhzan: 750 vs 50 MB/s).
    """
    cps = sorted(table)
    runs: list[tuple[int, int]] = []
    for cp in cps:
        if runs and cp == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], cp)
        else:
            runs.append((cp, cp))
    parts = (re.escape(chr(a)) if a == b else f"{re.escape(chr(a))}-{re.escape(chr(b))}" for a, b in runs)
    return re.compile("[" + "".join(parts) + "]")


_HAS_PRESENTATION_OR_REMOVED = _any_of(PRESENTATION_AND_REMOVED)
_HAS_LETTER = _any_of(LETTERS)
_HAS_SPACE = _any_of(SPACES)
_LINE_BREAKS = regex.compile(r"\r\n?|[\x85\u2028\u2029]")
_ZWNJ_RUN = regex.compile(
    r"(?<=[\p{Script=Arabic}&&\p{L}]\p{M}*)(?P<keep>\u200C+)(?=[\p{Script=Arabic}&&\p{L}])|\u200C+", regex.V1)
_DECIMAL_SEP = regex.compile(r"(?<=[0-9])[\u066B\u066C](?=[0-9])")
_SPACE_BEFORE_PUNCT = regex.compile(r"(?<=\S) +(?=[\u06D4\u060C\u061F\u061B])")
_SPACE_RUN = regex.compile(r" {2,}")


def _nfc(text: str) -> str:
    return text if unicodedata.is_normalized("NFC", text) else unicodedata.normalize("NFC", text)


def normalize(text: str) -> str:
    """Normalize Urdu text as described in the module docstring."""
    text = _LINE_BREAKS.sub("\n", text)
    if _HAS_PRESENTATION_OR_REMOVED.search(text):
        text = text.translate(PRESENTATION_AND_REMOVED)
    if ZWNJ in text:
        text = _ZWNJ_RUN.sub(lambda m: ZWNJ if m.group("keep") else "", text)
    text = _nfc(text)
    if _HAS_LETTER.search(text):
        text = _nfc(text.translate(LETTERS))
    text = _DECIMAL_SEP.sub(lambda m: "." if m.group() == "\u066B" else ",", text)
    if _HAS_SPACE.search(text):
        text = text.translate(SPACES)
    text = _SPACE_RUN.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub("", text)
    return "\n".join(line for line in (line.strip(" ") for line in text.split("\n")) if line)
