"""Count the code points that Urdu normalization (Step 4) acts on, per source.

    python clean/audit_chars.py                             # every data/samples/<source>.sample.jsonl
    python clean/audit_chars.py --source ufal_umc --contexts 10
    python clean/audit_chars.py --interim --limit 20000     # first N docs of data/interim/<source>.jsonl.gz
    python clean/audit_chars.py --normalized                # what is left after normalize.normalize

For each source it counts every category in CATEGORIES (and each distinct
match inside it), the documents that NFC would change, the characters on each
side of a ZWNJ, and the most frequent code points that are neither ASCII, nor
core Urdu, nor in a category. Up to --contexts examples per category are kept,
with the match in ⟦…⟧ and invisible characters spelled out as <U+XXXX>.

With --normalized, each text goes through normalize.normalize first, which
shows what normalization leaves behind. Writes data/stats/char_audit.json (or
char_audit_normalized.json) and prints occurrences per million
characters, one row per category and one column per source. Read-only: the
corpus is not changed.
"""

import argparse
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import INTERIM_DIR, SAMPLES_DIR  # noqa: E402
from common import STATS_DIR, iter_jsonl, setup_logging, write_json  # noqa: E402
from normalize import normalize  # noqa: E402

CONTEXT_CHARS = 20
TOP_OTHERS = 40
TOP_NEIGHBOURS = 25

# name -> (pattern, what it is). Order is the order of the printed table.
CATEGORIES: dict[str, tuple[str, str]] = {
    "arabic_yeh": (r"[\u064A\u0649]", "Arabic yeh / alef maksura (handout: map to U+06CC)"),
    "arabic_kaf": (r"[\u0643\u06AA]", "Arabic kaf / swash kaf (handout: map to U+06A9)"),
    "arabic_heh": (r"\u0647", "Arabic heh (handout: map to U+06C1)"),
    "heh_like_other": (r"[\u0629\u06C0\u06C3\u06D5]", "teh marbuta, heh with yeh above, teh marbuta goal, ae"),
    "tatweel": (r"\u0640", "tatweel (handout: remove)"),
    "zwnj": (r"\u200C", "zero-width non-joiner (handout: decide)"),
    "zwj": (r"\u200D", "zero-width joiner"),
    "invisible_other": (r"[\u00AD\u061C\u180E\u200B\u200E\u200F\u202A-\u202E\u2060-\u2064\u2066-\u2069\uFEFF]",
                        "soft hyphen, ALM, ZWSP, LRM/RLM, bidi embeddings and isolates, word joiner, BOM"),
    "control": (r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "C0/C1 control characters other than \\t \\n \\r"),
    "replacement_char": (r"\uFFFD", "U+FFFD, left by a bad decode upstream"),
    "digits_ascii": (r"[0-9]", "ASCII digits"),
    "digits_arabic_indic": (r"[\u0660-\u0669]", "Arabic-Indic digits"),
    "digits_extended": (r"[\u06F0-\u06F9]", "Extended Arabic-Indic (Urdu/Persian) digits"),
    "space_unusual": (r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", "no-break and other non-ASCII spaces"),
    "tab": (r"\t", "tab"),
    "carriage_return": (r"\r", "carriage return"),
    "line_separator_other": (r"[\x85\u2028\u2029]", "NEL, line separator, paragraph separator"),
    "space_run": (r" {2,}", "two or more ASCII spaces in a row"),
    "space_at_line_edge": (r"(?m)^[ \t]+|[ \t]+$", "leading or trailing spaces on a line"),
    "blank_line_run": (r"\n{3,}", "two or more empty lines in a row"),
    "presentation_form": (r"[\uFB50-\uFDFF\uFE70-\uFEFE]", "Arabic presentation forms (NFKC territory)"),
    "combining_madda_hamza": (r".[\u0653-\u0655]", "letter + combining madda/hamza above/below (NFC may compose)"),
    "harakat": (r"[\u064B-\u0652\u0656-\u065F\u0670]", "diacritics: zabar, zer, pesh, tashdeed, jazm, ..."),
    "space_before_urdu_punct": (r" [۔،؟؛]", "a space before ۔ ، ؟ ؛ (UFAL is pre-tokenized)"),
    "latin_punct_after_urdu": (r"(?<=[\u0600-\u06FF] ?)[.,?;]", "ASCII . , ? ; right after an Urdu letter"),
    "endoftext_literal": (r"<\|endoftext\|>", "the literal separator token (split step removes it)"),
}
COMPILED = {name: regex.compile(pattern) for name, (pattern, _) in CATEGORIES.items()}

# Standard Urdu letters, hamza forms and punctuation: never reported as "other".
URDU_CORE = frozenset(
    "ءآأؤئابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہۂھیےۓ"
    "۔،؟؛٪"
)
# Characters already counted by a single-character category (all in the BMP).
_SINGLE_CHAR = regex.compile("|".join(CATEGORIES[name][0] for name in (
    "arabic_yeh", "arabic_kaf", "arabic_heh", "heh_like_other", "tatweel", "zwnj", "zwj", "invisible_other",
    "control", "replacement_char", "digits_arabic_indic", "digits_extended", "space_unusual",
    "line_separator_other", "presentation_form", "harakat")))
CATEGORY_CHARS = frozenset(
    chr(cp) for cp in range(0x10000) if _SINGLE_CHAR.fullmatch(chr(cp))
) | frozenset("\u0653\u0654\u0655")


def visible(s: str) -> str:
    """Spell out characters that do not show on screen (format, control, space separators other than ' ')."""
    out = []
    for ch in s:
        cat = unicodedata.category(ch)
        if ch == "\n":
            out.append("⏎")
        elif cat in ("Cf", "Cc", "Zl", "Zp") or (cat == "Zs" and ch != " "):
            out.append(f"<U+{ord(ch):04X}>")
        else:
            out.append(ch)
    return "".join(out)


def char_label(ch: str) -> str:
    return f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"


def match_label(s: str) -> str:
    """A match as code points, e.g. 'U+0627 U+0653' (single characters also get their name)."""
    if len(s) == 1:
        return char_label(s)
    return " ".join(f"U+{ord(ch):04X}" for ch in s)


class Audit:
    """Running counts for one source."""

    def __init__(self, source: str, contexts: int) -> None:
        self.source = source
        self.max_contexts = contexts
        self.docs = 0
        self.chars = 0
        self.nfc_changed_docs = 0
        self.counts: Counter[str] = Counter()
        self.docs_with: Counter[str] = Counter()
        self.matches: dict[str, Counter[str]] = {name: Counter() for name in CATEGORIES}
        self.contexts: dict[str, list[str]] = {name: [] for name in CATEGORIES}
        self.zwnj_neighbours: Counter[str] = Counter()
        self.others: Counter[str] = Counter()

    def add(self, text: str) -> None:
        self.docs += 1
        self.chars += len(text)
        if not unicodedata.is_normalized("NFC", text):
            self.nfc_changed_docs += 1
        for name, pattern in COMPILED.items():
            n = 0
            for m in pattern.finditer(text):
                n += 1
                self.matches[name][match_label(m.group())] += 1
                if len(self.contexts[name]) < self.max_contexts:
                    a, b = max(0, m.start() - CONTEXT_CHARS), m.end() + CONTEXT_CHARS
                    self.contexts[name].append(
                        visible(text[a:m.start()]) + "⟦" + visible(m.group()) + "⟧" + visible(text[m.end():b]))
                if name == "zwnj":
                    before = text[m.start() - 1] if m.start() else "^"
                    after = text[m.end()] if m.end() < len(text) else "$"
                    self.zwnj_neighbours[f"{visible(before)} + ZWNJ + {visible(after)}"] += 1
            if n:
                self.counts[name] += n
                self.docs_with[name] += 1
        for ch in text:
            if ch > "\x7e" and ch not in URDU_CORE and ch not in CATEGORY_CHARS:
                self.others[ch] += 1

    def per_million(self, name: str) -> float:
        return self.counts[name] * 1e6 / self.chars if self.chars else 0.0

    def report(self) -> dict[str, Any]:
        return {
            "docs": self.docs,
            "chars": self.chars,
            "nfc_changed_docs": self.nfc_changed_docs,
            "categories": {
                name: {
                    "count": self.counts[name],
                    "per_million_chars": round(self.per_million(name), 2),
                    "docs_with": self.docs_with[name],
                    "matches": dict(self.matches[name].most_common(TOP_OTHERS)),
                    "contexts": self.contexts[name],
                }
                for name in CATEGORIES
            },
            "zwnj_neighbours": dict(self.zwnj_neighbours.most_common(TOP_NEIGHBOURS)),
            "top_other_chars": {char_label(ch): n for ch, n in self.others.most_common(TOP_OTHERS)},
        }


def audit_docs(source: str, docs: Iterable[dict[str, Any]], contexts: int) -> Audit:
    audit = Audit(source, contexts)
    for doc in docs:
        audit.add(doc["text"])
    return audit


def find_inputs(directory: Path, suffix: str, sources: Sequence[str] | None) -> dict[str, Path]:
    found = {p.name.removesuffix(suffix): p for p in sorted(directory.glob(f"*{suffix}"))}
    if sources:
        missing = [s for s in sources if s not in found]
        if missing:
            raise SystemExit(f"no {suffix} file in {directory} for: {', '.join(missing)}")
        found = {s: found[s] for s in sources}
    if not found:
        raise SystemExit(f"no *{suffix} files in {directory}")
    return found


def format_table(audits: Sequence[Audit]) -> str:
    """Occurrences per million characters: one row per category, one column per source."""
    width = max(len(name) for name in CATEGORIES)
    cols = [max(len(a.source), 10) for a in audits]
    lines = ["per million chars".ljust(width) + "".join(f"  {a.source:>{w}}" for a, w in zip(audits, cols))]
    lines.append("-" * width + "".join("  " + "-" * w for w in cols))
    for name in CATEGORIES:
        lines.append(name.ljust(width) + "".join(f"  {a.per_million(name):>{w},.1f}" for a, w in zip(audits, cols)))
    lines.append("nfc_changed_docs".ljust(width)
                 + "".join(f"  {a.nfc_changed_docs:>{w},}" for a, w in zip(audits, cols)))
    lines.append("docs".ljust(width) + "".join(f"  {a.docs:>{w},}" for a, w in zip(audits, cols)))
    lines.append("chars".ljust(width) + "".join(f"  {a.chars:>{w},}" for a, w in zip(audits, cols)))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", help="audit only this source (repeatable; default: all found)")
    p.add_argument("--interim", action="store_true",
                   help="read data/interim/<source>.jsonl.gz instead of data/samples/<source>.sample.jsonl")
    p.add_argument("--normalized", action="store_true", help="audit normalize(text) instead of the text")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="audit only the first N documents per source")
    p.add_argument("--contexts", type=int, default=5, metavar="N", help="examples kept per category (default 5)")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="default data/stats/char_audit.json, or char_audit_normalized.json with --normalized")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out is None:
        args.out = STATS_DIR / ("char_audit_normalized.json" if args.normalized else "char_audit.json")
    log = setup_logging("audit_chars")
    if args.interim:
        inputs = find_inputs(args.interim_dir, ".jsonl.gz", args.source)
    else:
        inputs = find_inputs(args.samples_dir, ".sample.jsonl", args.source)
    audits = []
    for source, path in inputs.items():
        docs: Iterator[dict[str, Any]] = iter_jsonl(path)
        if args.limit is not None:
            docs = islice(docs, args.limit)
        if args.normalized:
            docs = ({**doc, "text": normalize(doc["text"])} for doc in docs)
        audit = audit_docs(source, docs, args.contexts)
        log.info("%s: %d docs, %d chars from %s", source, audit.docs, audit.chars, path)
        audits.append(audit)
    write_json(args.out, {
        "input": "interim" if args.interim else "samples",
        "normalized": args.normalized,
        "limit": args.limit,
        "categories": {name: desc for name, (_, desc) in CATEGORIES.items()},
        "sources": {a.source: a.report() for a in audits},
    })
    print(format_table(audits))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
