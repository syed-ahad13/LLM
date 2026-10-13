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
It is slow on purpose.

train() is the fast trainer and learns exactly the same merges (Step 8,
tests/test_bpe_fast.py, docs/equivalence_1mb.md). It uses the handout's three
optimizations:
1. count_words() counts each distinct word once (in a process pool, cached on
   disk), and its pairs are weighted by its frequency;
2. after a merge only the words that contained the pair are updated, found
   through an index from each pair to the words that contain it;
3. encode() caches the token IDs of every word already encoded.

The fast trainer's state, and what stays true after every merge:
- words[i]: the token IDs of distinct word i; freqs[i]: its count.
- pair_counts[pk], pk = (a << 16) | b (every ID is below 65,536): the sum,
  over the words, of freq times the number of adjacent (a, b) positions,
  overlaps counted. That is exactly the reference trainer's recount. A pair
  with count 0 is deleted, and the pair just merged always reaches 0.
- where[pk]: the IDs of the words that contain (a, b) or once did. A new
  adjacency always involves the token just created, which is created once, so
  each word is appended to a pair's list at most once. Entries go stale when
  later merges destroy the pair, and are skipped when the list is used.
- heap: entries (-count, rk[a], rk[b], -pk). rk reverses byte order (see
  reverse_key), so the smallest entry is the highest count, then the greatest
  (bytes of a, bytes of b): the spec's tie-break, compared as C-level bytes.
  The last element, -pk, settles a tie between pairs with identical bytes the
  way train_reference does: greater IDs win. Every live pair has an entry
  whose count is at least its current count; an entry is used only when its
  count equals the current one. A higher stored count is pushed again with
  the current count; a lower one is dropped (a fresher entry exists). Only
  increases push; decreases are repaired lazily when popped.

Standard library and regex only (tests/test_forbidden_imports.py).
"""

import hashlib
import heapq
import json
import logging
import multiprocessing
import os
import pickle
import re
import resource
import sys
import time
import unicodedata
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
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


def reverse_key(token: bytes) -> bytes:
    """A key that sorts byte strings in reverse: for x != y, x < y exactly when reverse_key(x) > reverse_key(y).

    Each byte c becomes (0, 255 - c), which reverses the order at the first differing byte, and the end of
    the string becomes (1, 0), which sorts after every byte, so a proper prefix (the smaller string) gets
    the greater key.
    """
    return b"".join(bytes((0, 255 - c)) for c in token) + b"\x01\x00"


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6  # bytes on macOS, KiB on Linux


# ---------------------------------------------------------------------------
# Word counting (optimization 1)
# ---------------------------------------------------------------------------

COUNT_BLOCK_BYTES = 16 * 1024 * 1024
HEAP_SLACK = 1_000_000  # rebuild the heap when it holds more than 3 x live pairs + this many entries
WORD_COUNTS_VERSION = 1
_COMPILED: dict[str, Any] = {}


def _count_text(text: str, pattern: str, special_tokens: Sequence[str]) -> Counter[str]:
    """Pre-tokenized words of text and their counts; the special tokens are separators, not words."""
    compiled = _COMPILED.get(pattern)
    if compiled is None:
        compiled = _COMPILED[pattern] = regex.compile(pattern)
    counts: Counter[str] = Counter()
    pieces = (re.split("|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)), text)
              if special_tokens else [text])
    for piece in pieces:
        if piece:
            counts.update(compiled.findall(piece))
    return counts


def _count_chunk(item: tuple[bytes, str, tuple[str, ...]]) -> Counter[str]:
    chunk, pattern, special_tokens = item
    return _count_text(chunk.decode("utf-8"), pattern, special_tokens)


def _iter_chunks(path: Path, block_bytes: int, separators: Sequence[bytes]) -> Iterator[bytes]:
    """Blocks of the file, each cut just after its last separator; the rest is carried into the next block."""
    carry = b""
    with open(path, "rb") as f:
        while block := f.read(block_bytes):
            data = carry + block
            ends = [i + len(s) for s in separators if (i := data.rfind(s)) != -1]
            cut = max(ends, default=0)
            if cut:
                yield data[:cut]
                carry = data[cut:]
            else:
                carry = data  # no separator yet: a long document; keep reading
    if carry:
        yield carry


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(COUNT_BLOCK_BYTES):
            h.update(block)
    return h.hexdigest()


def count_words(source: str | os.PathLike[str], workers: int | None = None, *, pattern: str = URDU_PATTERN,
                special_tokens: Sequence[str] = (ENDOFTEXT,), cache_dir: str | os.PathLike[str] | None = None,
                block_bytes: int = COUNT_BLOCK_BYTES) -> dict[str, int]:
    """Count the pre-tokenized words of a text (str) or of a file (a path object): {word: occurrences}.

    The special tokens split the text and are not counted. A file is read in blocks of block_bytes, each cut
    after its last special token so no document is split, and the blocks are counted in a pool of workers
    (default: all cores but one). With cache_dir, the counts of a file are pickled under a key made of the
    file's sha256, the pattern and the special tokens, and a rerun loads them instead of counting again.
    """
    specials = tuple(special_tokens)
    if not isinstance(source, os.PathLike):
        return dict(_count_text(source, pattern, specials))
    path = Path(source)
    cache_path = None
    if cache_dir is not None:
        config = hashlib.sha256(json.dumps([pattern, sorted(specials)]).encode("utf-8")).hexdigest()
        file_sha = _file_sha256(path)
        cache_path = Path(cache_dir) / f"word_counts_{file_sha[:16]}_{config[:12]}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if (cached.get("version"), cached.get("file_sha256"), cached.get("pattern"),
                    cached.get("special_tokens")) == (WORD_COUNTS_VERSION, file_sha, pattern, list(specials)):
                log.info("word counts of %s loaded from %s", path, cache_path)
                return cached["counts"]
    start = time.perf_counter()
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    total: Counter[str] = Counter()
    separators = [t.encode("utf-8") for t in specials]
    chunks = ((chunk, pattern, specials) for chunk in _iter_chunks(path, block_bytes, separators))
    n_chunks = 0
    if workers == 1:
        for item in chunks:
            total.update(_count_chunk(item))
            n_chunks += 1
    else:
        with multiprocessing.Pool(workers) as pool:
            pending: deque[Any] = deque()
            for item in chunks:  # at most 2 x workers blocks in flight, so the file is never all in memory
                pending.append(pool.apply_async(_count_chunk, (item,)))
                if len(pending) >= 2 * workers:
                    total.update(pending.popleft().get())
                    n_chunks += 1
            while pending:
                total.update(pending.popleft().get())
                n_chunks += 1
    counts = dict(total)
    log.info("counted %s: %d blocks, %d distinct words, %d words in %.1f s", path, n_chunks, len(counts),
             sum(counts.values()), time.perf_counter() - start)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "wb") as f:
                pickle.dump({"version": WORD_COUNTS_VERSION, "file_sha256": file_sha, "pattern": pattern,
                             "special_tokens": list(specials), "counts": counts}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
        finally:
            tmp.unlink(missing_ok=True)
    return counts


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
        self.train_log: list[dict[str, Any]] = []
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
    # Fast trainer (optimizations 1 and 2; state and invariants in the module docstring)
    # ------------------------------------------------------------------

    def train(self, word_counts: Mapping[str, int], vocab_size: int, *, log_every: int = 1000,
              checkpoint_every: int = 5000, checkpoint_path: str | os.PathLike[str] | None = None,
              resume: bool = False, debug: int = 0) -> dict[str, Any]:
        """Learn vocab_size - 257 merges from {word: count} (see count_words), exactly as train_reference would
        from the text those words came from.

        log_every: log progress (and append to self.train_log) every this many merges; 0 turns it off.
        checkpoint_every / checkpoint_path: save the merges so far every this many merges.
        resume: start from the merges in checkpoint_path; each word is re-encoded by replaying them.
        debug: if > 0, check the invariants: the merged pair is gone after every merge, no word is appended
            twice to a pair's list, and every debug merges pair_counts equals a full recount (small inputs).
        """
        n_target = vocab_size - self.first_merge_id
        if n_target < 0:
            raise ValueError(f"vocab_size must be at least {self.first_merge_id}, got {vocab_size}")
        if vocab_size > 1 << 16:
            raise ValueError(f"vocab_size {vocab_size} does not fit the 16-bit IDs of the pair keys")
        start = time.perf_counter()
        self._reset()
        if resume:
            if checkpoint_path is None or not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"no checkpoint to resume from: {checkpoint_path}")
            done = BPETokenizer.load(checkpoint_path)
            if (done.pattern, done.special_tokens) != (self.pattern, self.special_tokens):
                raise ValueError(f"{checkpoint_path} was trained with another pattern or other special tokens")
            if len(done.merge_counts) != len(done.merges) or len(done.merges) > n_target:
                raise ValueError(f"{checkpoint_path}: {len(done.merges)} merges with {len(done.merge_counts)} "
                                 f"counts; cannot resume towards {n_target} merges")
            for a, b in done.merges:
                self._add_merge(a, b)
            self.merge_counts = list(done.merge_counts)
            log.info("resuming from %s after %d merges", checkpoint_path, len(self.merges))
        resumed = len(self.merges)
        vocab, merges, merge_counts = self.vocab, self.merges, self.merge_counts
        rk = [reverse_key(vocab[i]) for i in range(self.first_merge_id + resumed)]

        # Distinct words as token IDs (replaying the checkpoint's merges when resuming), then the pair counts,
        # the pair -> words index and the heap.
        words: list[list[int]] = []
        freqs: list[int] = []
        encode_bytes = self._encode_bytes
        for word, f in word_counts.items():
            if f <= 0:
                continue
            data = word.encode("utf-8", "surrogatepass")
            ids = list(encode_bytes(data)) if resumed and len(data) > 1 else list(data)
            if len(ids) > 1:
                words.append(ids)
                freqs.append(f)
        pair_counts: dict[int, int] = {}
        where: dict[int, list[int]] = {}
        for wid, w in enumerate(words):
            f = freqs[wid]
            for i in range(len(w) - 1):
                k = (w[i] << 16) | w[i + 1]
                pair_counts[k] = pair_counts.get(k, 0) + f
                lst = where.get(k)
                if lst is None:
                    where[k] = [wid]
                elif lst[-1] != wid:
                    lst.append(wid)
        heap = [(-c, rk[k >> 16], rk[k & 0xFFFF], -k) for k, c in pair_counts.items()]
        heapq.heapify(heap)
        setup_seconds = time.perf_counter() - start
        log.info("fast trainer: %d distinct words, %d pairs, set up in %.1f s", len(words), len(pair_counts),
                 setup_seconds)

        heappush, heappop, add_merge = heapq.heappush, heapq.heappop, self._add_merge
        self.train_log = []
        compactions = 0
        t_last, m_last = time.perf_counter(), len(merges)
        while len(merges) < n_target:
            # Pop until an entry's count is the pair's current count.
            pk = -1
            while heap:
                negc, rka, rkb, negk = heappop(heap)
                cur = pair_counts.get(-negk)
                if cur is None:
                    continue  # the pair is gone
                if -negc == cur:
                    pk, count = -negk, cur
                    break
                if -negc > cur:
                    heappush(heap, (-cur, rka, rkb, negk))  # the count went down since this entry was pushed
                # a lower stored count: a fresher entry is in the heap
            if pk < 0:
                log.warning("no pair left after %d merges; %d were asked for", len(merges), n_target)
                break
            if heap and heap[0][1] == rka and heap[0][2] == rkb and -heap[0][0] == count \
                    and pair_counts.get(-heap[0][3]) == count:
                log.warning("merge %d: pairs %s and %s have the same bytes; the greater IDs win", len(merges) + 1,
                            (pk >> 16, pk & 0xFFFF), (-heap[0][3] >> 16, -heap[0][3] & 0xFFFF))
            a, b = pk >> 16, pk & 0xFFFF
            z = add_merge(a, b)
            rk.append(reverse_key(vocab[z]))
            merge_counts.append(count)

            # Rewrite only the words that contained (a, b); collect the pair count changes.
            delta: dict[int, int] = {}
            created: set[int] = set()
            for wid in where.pop(pk, ()):
                w = words[wid]
                if a not in w:
                    continue  # stale: an earlier merge took a
                n = len(w)
                out: list[int] = []
                i = 0
                hit = False
                while i < n:
                    t = w[i]
                    if t == a and i + 1 < n and w[i + 1] == b:
                        out.append(z)
                        i += 2
                        hit = True
                    else:
                        out.append(t)
                        i += 1
                if not hit:
                    continue  # stale: (a, b) is no longer adjacent here
                f = freqs[wid]
                for i in range(n - 1):
                    k = (w[i] << 16) | w[i + 1]
                    delta[k] = delta.get(k, 0) - f
                prev = out[0]
                for i in range(1, len(out)):
                    t = out[i]
                    k = (prev << 16) | t
                    delta[k] = delta.get(k, 0) + f
                    if prev == z or t == z:
                        lst = where.get(k)
                        if lst is None:
                            where[k] = [wid]
                            if debug:
                                created.add(k)
                        elif lst[-1] != wid:
                            if debug and k not in created:
                                raise AssertionError(f"merge {len(merges)}: pair {(prev, t)} indexed before z={z}")
                            lst.append(wid)
                    prev = t
                words[wid] = out

            for k, d in delta.items():
                if d == 0:
                    continue
                c = pair_counts.get(k, 0) + d
                if c > 0:
                    pair_counts[k] = c
                    if d > 0:
                        heappush(heap, (-c, rk[k >> 16], rk[k & 0xFFFF], -k))
                elif c == 0:
                    del pair_counts[k]
                    where.pop(k, None)  # no new adjacency can recreate it: every new pair holds a new token
                else:
                    raise AssertionError(f"merge {len(merges)}: pair {(k >> 16, k & 0xFFFF)} count went to {c}")

            m = len(merges)
            if debug:
                if pk in pair_counts:
                    raise AssertionError(f"merge {m}: merged pair {(a, b)} still has count {pair_counts[pk]}")
                if m % debug == 0:
                    recount: dict[int, int] = {}
                    for wid, w in enumerate(words):
                        for i in range(len(w) - 1):
                            k = (w[i] << 16) | w[i + 1]
                            recount[k] = recount.get(k, 0) + freqs[wid]
                    if recount != pair_counts:
                        raise AssertionError(f"merge {m}: pair_counts differ from a full recount")
            if len(heap) > 3 * len(pair_counts) + HEAP_SLACK:
                heap = [(-c, rk[k >> 16], rk[k & 0xFFFF], -k) for k, c in pair_counts.items()]
                heapq.heapify(heap)
                compactions += 1
            if log_every and m % log_every == 0:
                now = time.perf_counter()
                entry = {"merge": m, "count": count, "token": token_display(vocab[z]),
                         "seconds": round(now - start, 2),
                         "merges_per_s": round((m - m_last) / (now - t_last), 1) if now > t_last else None,
                         "heap": len(heap), "pairs": len(pair_counts), "peak_rss_mb": round(_peak_rss_mb(), 1)}
                self.train_log.append(entry)
                log.info("merge %d/%d count %d %s | %s merges/s | heap %d | pairs %d | peak RSS %.0f MB", m,
                         n_target, count, entry["token"], entry["merges_per_s"], len(heap), len(pair_counts),
                         entry["peak_rss_mb"])
                t_last, m_last = now, m
            if checkpoint_path and checkpoint_every and m % checkpoint_every == 0 and m < n_target:
                self.save(checkpoint_path, meta={"checkpoint": True, "merges_done": m})

        self._cache.clear()
        seconds = time.perf_counter() - start
        return {"trainer": "fast", "merges": len(merges), "resumed_from": resumed, "distinct_words": len(words),
                "setup_seconds": round(setup_seconds, 2), "merge_seconds": round(seconds - setup_seconds, 2),
                "seconds": round(seconds, 2), "heap_compactions": compactions, "peak_rss_mb": round(_peak_rss_mb(), 1)}

    # ------------------------------------------------------------------
    # Encoding and decoding
    # ------------------------------------------------------------------

    def _encode_word(self, word: str) -> tuple[int, ...]:
        return self._encode_bytes(word.encode("utf-8", "surrogatepass"))

    def _encode_bytes(self, data: bytes) -> tuple[int, ...]:
        """Merge the present pair with the lowest merge ID until none is left (= replaying the merges in order)."""
        ids = list(data)
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
