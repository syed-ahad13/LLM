"""Byte-level BPE tokenizer written from scratch: class BPETokenizer.

Spec (CLAUDE.md, "Tokenizer spec"; docs/assignment.md, Part B):
- IDs 0-255 are the byte values. <|endoftext|> is ID 256: never split, never
  part of a merge. Merges get IDs 257, 258, ... in the order learned.
- Text is pre-tokenized with URDU_PATTERN (docs/pretokenizer.md), so merges
  never cross a word boundary. A word is a run of letters, combining marks and
  ZWNJ with an optional leading space, so اُردو stays whole.
- Training picks the most frequent adjacent pair, overlaps included ("aaa"
  holds (a, a) twice). Ties go to the GREATER pair, compared as
  (vocab[a], vocab[b]), a tuple of bytes, never by ID. Two different pairs can
  only have equal bytes if two merges built the same string (ab + c and
  a + bc); then, and only then, the greater pair of IDs wins, and a warning is
  logged. A merge replaces non-overlapping occurrences left to right.
- encode() merges, inside each word, the present pair with the lowest merge ID
  until none is left, which is the same as replaying the merges in order.
  Words are cached.
- decode(encode(s)) == s for every Python string, lone surrogates included
  (UTF-8 with surrogatepass). The tokenizer never normalizes.

train_reference() is the reference trainer: every occurrence of every word is
kept as its own list of byte IDs and all pairs are recounted after every merge.
It is slow on purpose; the fast trainer (Step 8) must learn identical merges
on the 1 MB sample.

Standard library and regex only (tests/test_forbidden_imports.py).
"""

import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Any, Literal

import regex

log = logging.getLogger("bpe")

ENDOFTEXT = "<|endoftext|>"
FORMAT = "urdu-bpe-v1"

# docs/pretokenizer.md explains each alternative and the patterns considered instead.
URDU_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"| ?[\p{L}\p{M}‌]+"
    r"| ?\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{M}\p{N}‌]+"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
# The handout's GPT-2 pattern, for the tests and the report only.
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

Pair = tuple[int, int]


def count_pairs(words: Iterable[Sequence[int]]) -> Counter[Pair]:
    """Adjacent pairs over all words, overlaps included: (a, a, a) holds (a, a) twice."""
    counts: Counter[Pair] = Counter()
    for w in words:
        counts.update(zip(w, w[1:]))
    return counts


def merge_word(word: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace the non-overlapping occurrences of pair in word, left to right: (a, a, a) -> (aa, a)."""
    a, b = pair
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def token_display(token: bytes) -> str:
    """A token as readable text: spaces as ␣, bytes that are not valid UTF-8 as \\xNN, invisible or control
    characters as \\n, \\t, \\r or \\uXXXX, and a backslash as \\\\."""
    out = []
    for ch in token.decode("utf-8", errors="surrogateescape"):
        cp = ord(ch)
        if 0xDC80 <= cp <= 0xDCFF:  # a byte that is not part of a valid UTF-8 character
            out.append(f"\\x{cp - 0xDC00:02x}")
        elif ch == " ":
            out.append("␣")
        elif ch == "\\":
            out.append("\\\\")
        elif ch in "\n\t\r":
            out.append({"\n": "\\n", "\t": "\\t", "\r": "\\r"}[ch])
        elif unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp", "Zs"):
            out.append(f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}")
        else:
            out.append(ch)
    return "".join(out)


class BPETokenizer:
    """Byte-level BPE with one special token, <|endoftext|> = 256, by default."""

    def __init__(self, pattern: str = URDU_PATTERN, special_tokens: dict[str, int] | None = None) -> None:
        self.pattern = pattern
        self.compiled_pattern = regex.compile(pattern)
        self.special_tokens: dict[str, int] = dict(special_tokens) if special_tokens is not None else {ENDOFTEXT: 256}
        ids = sorted(self.special_tokens.values())
        if ids != list(range(256, 256 + len(ids))):
            raise ValueError(f"special token IDs must be 256, 257, ... with no gaps, got {ids}")
        self.first_merge_id = 256 + len(self.special_tokens)
        self.meta: dict[str, Any] = {}
        self._reset()

    def _reset(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for token, i in self.special_tokens.items():
            self.vocab[i] = token.encode("utf-8")
        self.merges: list[Pair] = []
        self.merge_ranks: dict[Pair, int] = {}
        self.merge_counts: list[int] = []
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return self.first_merge_id + len(self.merges)

    def _add_merge(self, a: int, b: int) -> int:
        new_id = self.first_merge_id + len(self.merges)
        for x in (a, b):
            if x not in self.vocab or 256 <= x < self.first_merge_id:
                raise ValueError(f"merge {len(self.merges) + 1} uses {x}, which is not a byte or an earlier merge")
        if (a, b) in self.merge_ranks:
            raise ValueError(f"pair {(a, b)} is merged twice")
        self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.merges.append((a, b))
        self.merge_ranks[(a, b)] = new_id
        return new_id

    # ------------------------------------------------------------------
    # Pre-tokenization and special tokens
    # ------------------------------------------------------------------

    def pretokenize(self, text: str) -> list[str]:
        """The pattern's matches; together they are the whole text."""
        return self.compiled_pattern.findall(text)

    def _split_special(self, text: str, allowed: AbstractSet[str]) -> list[str]:
        """text split around the allowed special tokens: [text, special, text, special, ..., text]."""
        if not allowed:
            return [text]
        alternatives = "|".join(re.escape(t) for t in sorted(allowed, key=len, reverse=True))
        return re.split(f"({alternatives})", text)

    # ------------------------------------------------------------------
    # Reference trainer
    # ------------------------------------------------------------------

    def _choose(self, counts: Counter[Pair]) -> tuple[Pair, int]:
        """The pair with the highest count; ties go to the greater (vocab[a], vocab[b])."""
        best = max(counts.values())
        tied = [p for p, c in counts.items() if c == best]
        if len(tied) == 1:
            return tied[0], best
        vocab = self.vocab
        ranked = sorted(tied, key=lambda p: (vocab[p[0]], vocab[p[1]], p), reverse=True)
        top, second = ranked[0], ranked[1]
        if (vocab[top[0]], vocab[top[1]]) == (vocab[second[0]], vocab[second[1]]):
            log.warning("merge %d: pairs %s and %s have the same bytes; the greater IDs win",
                        len(self.merges) + 1, top, second)
        return top, best

    def train_reference(self, text: str, vocab_size: int, verbose: bool = False) -> dict[str, Any]:
        """Learn vocab_size - 257 merges from text by recounting every pair after every merge.

        Starts from the 257-token base vocabulary. Stops early, with a warning, when no pair is left.
        Returns a short summary (merges learned, word occurrences, seconds).
        """
        n_merges = vocab_size - self.first_merge_id
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {self.first_merge_id}, got {vocab_size}")
        start = time.perf_counter()
        self._reset()
        words: list[list[int]] = []
        for i, chunk in enumerate(self._split_special(text, set(self.special_tokens))):
            if i % 2 == 0:  # odd positions are the special tokens themselves
                words.extend(list(w.encode("utf-8", "surrogatepass")) for w in self.pretokenize(chunk))
        occurrences = len(words)
        for _ in range(n_merges):
            counts = count_pairs(words)
            if not counts:
                log.warning("no pair left after %d merges; %d were asked for", len(self.merges), n_merges)
                break
            pair, count = self._choose(counts)
            new_id = self._add_merge(*pair)
            self.merge_counts.append(count)
            a = pair[0]
            for j, w in enumerate(words):
                if a in w and len(w) > 1:
                    merged = merge_word(w, pair, new_id)
                    if len(merged) != len(w):
                        words[j] = merged
            if verbose:
                log.info("merge %d/%d: %s -> %d %s (count %d)", len(self.merges), n_merges, pair, new_id,
                         token_display(self.vocab[new_id]), count)
        self._cache.clear()
        return {"trainer": "reference", "merges": len(self.merges), "word_occurrences": occurrences,
                "seconds": round(time.perf_counter() - start, 3)}

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------

    def _encode_word(self, word: str) -> tuple[int, ...]:
        ids = list(word.encode("utf-8", "surrogatepass"))
        ranks = self.merge_ranks
        while len(ids) > 1:
            best_id = None
            best_pair: Pair = (0, 0)
            for pair in zip(ids, ids[1:]):
                r = ranks.get(pair)
                if r is not None and (best_id is None or r < best_id):
                    best_id, best_pair = r, pair
            if best_id is None:
                break
            ids = merge_word(ids, best_pair, best_id)
        return tuple(ids)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text with no special tokens: a literal <|endoftext|> is ordinary text."""
        ids: list[int] = []
        cache = self._cache
        for word in self.pretokenize(text):
            encoded = cache.get(word)
            if encoded is None:
                encoded = cache[word] = self._encode_word(word)
            ids.extend(encoded)
        return ids

    def encode(self, text: str, allowed_special: Literal["all", "none"] | AbstractSet[str] = "all") -> list[int]:
        """Encode text. The allowed special tokens become their IDs; any other special string is ordinary text."""
        if isinstance(allowed_special, str):
            if allowed_special not in ("all", "none"):
                raise ValueError(f'allowed_special must be "all", "none" or a set of special tokens, '
                                 f"got {allowed_special!r}")
            allowed = set(self.special_tokens) if allowed_special == "all" else set()
        else:
            allowed = set(allowed_special)
            if unknown := allowed - set(self.special_tokens):
                raise ValueError(f"not special tokens of this tokenizer: {sorted(unknown)}")
        ids: list[int] = []
        for i, part in enumerate(self._split_special(text, allowed)):
            if i % 2:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        try:
            return b"".join(self.vocab[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"unknown token id {e.args[0]}") from None

    def decode(self, ids: Iterable[int]) -> str:
        """The text of ids. A sequence that cuts a character decodes it as U+FFFD."""
        data = self.decode_bytes(ids)
        try:
            return data.decode("utf-8", errors="surrogatepass")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str], meta: dict[str, Any] | None = None) -> Path:
        """Write <name>.json (format urdu-bpe-v1) and a readable <name>.merges.txt next to it.

        meta is stored as given, plus merge_counts when the merges were trained in this process.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta | (meta or {}))
        if self.merge_counts and len(self.merge_counts) == len(self.merges):
            meta["merge_counts"] = self.merge_counts
        head = {"format": FORMAT, "pattern": self.pattern, "special_tokens": self.special_tokens,
                "vocab_size": self.vocab_size, "num_merges": len(self.merges)}
        lines = ["{"]
        lines += [f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()]
        lines.append('  "merges": [')
        lines.append(",\n".join(f"    [{a}, {b}]" for a, b in self.merges))
        lines.append("  ],")
        lines.append(f'  "meta": {json.dumps(meta, ensure_ascii=False, sort_keys=True)}')
        lines.append("}")
        _write_atomic(path, "\n".join(lines) + "\n")
        _write_atomic(merges_path(path), self.merges_text())
        return path

    def merges_text(self) -> str:
        rows = ["# rank\tnew_id\tleft_id\tright_id\ttoken_hex\ttoken  (rank 1 = first merge; ␣ = space, \\xNN = "
                "byte that is not valid UTF-8 on its own)"]
        for rank, (a, b) in enumerate(self.merges, start=1):
            new_id = self.first_merge_id + rank - 1
            token = self.vocab[new_id]
            rows.append(f"{rank}\t{new_id}\t{a}\t{b}\t{token.hex()}\t{token_display(token)}")
        return "\n".join(rows) + "\n"

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "BPETokenizer":
        """Load a saved tokenizer and rebuild its vocabulary by replaying the merges in order."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format") != FORMAT:
            raise ValueError(f"{path}: format {data.get('format')!r}, expected {FORMAT!r}")
        tok = cls(pattern=data["pattern"], special_tokens={k: int(v) for k, v in data["special_tokens"].items()})
        merges = data["merges"]
        if len(merges) != data["num_merges"]:
            raise ValueError(f"{path}: {len(merges)} merges, but num_merges is {data['num_merges']}")
        for a, b in merges:
            tok._add_merge(int(a), int(b))
        if tok.vocab_size != data["vocab_size"]:
            raise ValueError(f"{path}: vocab_size {data['vocab_size']}, but the merges give {tok.vocab_size}")
        tok.meta = dict(data.get("meta") or {})
        counts = tok.meta.pop("merge_counts", None)
        if isinstance(counts, list) and len(counts) == len(tok.merges):
            tok.merge_counts = [int(c) for c in counts]
        return tok


def merges_path(path: str | os.PathLike[str]) -> Path:
    """tokenizer/<name>.json -> tokenizer/<name>.merges.txt"""
    path = Path(path)
    return path.with_name(path.stem + ".merges.txt")


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
