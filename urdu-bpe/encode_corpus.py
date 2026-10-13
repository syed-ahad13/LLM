"""CLI: encode train.txt and val.txt with the trained tokenizer into train.bin and val.bin (Part B, Step 10).

    python encode_corpus.py --tokenizer tokenizer/urdu_bpe_50k.json --input data/final/train.txt \\
        --out data/final/train.bin --workers 7
    python encode_corpus.py --input data/final/val.txt --out data/final/val.bin --workers 7
    python encode_corpus.py --input data/final/train.txt --sample-mb 20      # smoke run on the first 20 MB
    python encode_corpus.py --verify --input data/final/train.txt --out data/final/train.bin   # check only

A .bin holds the token IDs of the whole .txt in order as raw little-endian
uint16, 2 bytes per ID, with no header: np.memmap(path, dtype="<u2", mode="r")
or np.fromfile(path, dtype="<u2") reads it. Each document is followed by
<|endoftext|> = ID 256, as in the .txt. test.txt is refused: the test split
stays text, for Part C.

Encoding. The input is read in 16 MB blocks cut just after their last
<|endoftext|> (bpe._iter_chunks, the reader of count_words), so no document is
split. The blocks are encoded in a process pool in file order (common.
ordered_imap, at most 2 x workers blocks in flight). Each worker loads the
tokenizer once and keeps its own cache word -> token IDs (optimization 3), and
returns its block as a "<u2" array: a 16 MB block is about 2.3M IDs, 4.6 MB as
uint16 against about 80 MB as a list of Python ints in the main process. The
main process checks that every ID is below the vocabulary size and appends the
block with tofile(). The .bin is written under a temporary name and renamed
when complete, so it is never half-written.

Verification (--verify, and after every encode). The .bin is walked through
np.memmap in 10M-ID chunks; each chunk becomes BYTES (vocab[id] joined) fed
into one sha256, which must equal the sha256 of the .txt. Bytes, never decoded
strings, so a chunk edge inside a character changes nothing. The count of ID
256 must equal the count of <|endoftext|> in the .txt and, when the .txt is a
split listed in data/final/manifest.json (same sha256), its document count.

Writes <out> and <out dir>/meta.json: vocab size, EOT ID, dtype, byte order,
the tokenizer and its sha256, the pattern, <split>_tokens, and under "splits"
the report of each encode (<split> = the .bin's stem). The report: tokens,
documents, input bytes, bytes per token, tokens per str.split() word, cache hit
rate per worker, wall-clock time and MB/s.

Exits with status 1 if verification fails.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bpe import COUNT_BLOCK_BYTES, ENDOFTEXT, BPETokenizer, _iter_chunks
from common import REPO_ROOT, Timer, machine_info, ordered_imap, setup_logging, sha256_file, write_json
from train_tokenizer import git_commit, make_sample, utc

DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer" / "urdu_bpe_50k.json"
DEFAULT_INPUT = REPO_ROOT / "data" / "final" / "train.txt"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "final" / "manifest.json"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
DTYPE = np.dtype("<u2")  # explicit little-endian uint16, whatever the machine
VERIFY_CHUNK_IDS = 10_000_000
EOT = ENDOFTEXT.encode("utf-8")

log = logging.getLogger("encode_corpus")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

_TOK: BPETokenizer | None = None


def _init_worker(tokenizer_path: str, use_cache: bool) -> None:
    """Pool initializer: load the tokenizer once per worker; its word cache lives as long as the worker."""
    global _TOK
    _TOK = BPETokenizer.load(tokenizer_path)
    _TOK.use_cache = use_cache


def _encode_block(block: bytes) -> tuple[np.ndarray, dict[str, int]]:
    """One block -> its IDs as a "<u2" array, plus this worker's running counters."""
    tok = _TOK
    assert tok is not None, "_init_worker was not called"
    text = block.decode("utf-8")
    ids = tok.encode(text, allowed_special="all")
    top = max(ids, default=0)
    if top >= tok.vocab_size:
        raise ValueError(f"token ID {top} >= vocab size {tok.vocab_size}")
    words = sum(len(doc.split()) for doc in text.split(ENDOFTEXT))  # per document, as in clean/corpus_stats.py
    stats = {"pid": os.getpid(), "bytes": len(block), "words": words,
             "lookups": tok.cache_lookups, "misses": tok.cache_misses, "cached_words": len(tok._cache)}
    return np.asarray(ids, dtype=DTYPE), stats


def encode_file(tokenizer_path: Path, text_path: Path, out: Path, workers: int, *, use_cache: bool = True,
                block_bytes: int = COUNT_BLOCK_BYTES) -> dict[str, Any]:
    """Encode text_path into out (raw "<u2"), in file order. Returns the report (see the module docstring)."""
    tok = BPETokenizer.load(tokenizer_path)
    if tok.vocab_size > 2**16:
        raise SystemExit(f"{tokenizer_path}: vocab size {tok.vocab_size:,} does not fit in uint16")
    vocab_size, eot = tok.vocab_size, tok.special_tokens[ENDOFTEXT]
    input_bytes = text_path.stat().st_size
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    tokens = docs = words = done = blocks = 0
    per_worker: dict[int, dict[str, int]] = {}
    blocks_in = _iter_chunks(text_path, block_bytes, [EOT])
    start = time.perf_counter()
    if workers == 1:
        _init_worker(str(tokenizer_path), use_cache)
        pool_cm: Any = contextlib.nullcontext(None)
    else:
        pool_cm = multiprocessing.Pool(workers, initializer=_init_worker, initargs=(str(tokenizer_path), use_cache))
    try:
        with pool_cm as pool, open(tmp, "wb") as f:
            for arr, stats in ordered_imap(pool, _encode_block, blocks_in, 2 * workers):
                if arr.size and int(arr.max()) >= vocab_size:
                    raise ValueError(f"block {blocks + 1}: token ID {int(arr.max())} >= vocab size {vocab_size}")
                arr.tofile(f)
                tokens += arr.size
                docs += int(np.count_nonzero(arr == eot))
                words += stats["words"]
                done += stats["bytes"]
                blocks += 1
                per_worker[stats["pid"]] = stats
                elapsed = time.perf_counter() - start
                log.info("block %d: %.1f/%.1f MB (%.0f%%), %s tokens, %.1f MB/s, ETA %.0f s", blocks, done / 1e6,
                         input_bytes / 1e6, 100 * done / max(input_bytes, 1), f"{tokens:,}", done / 1e6 / elapsed,
                         (input_bytes - done) / (done / elapsed) if done else 0)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    seconds = time.perf_counter() - start
    if done != input_bytes:
        raise RuntimeError(f"encoded {done:,} bytes of {input_bytes:,}")
    workers_out = []
    for i, (_, s) in enumerate(sorted(per_worker.items()), start=1):
        workers_out.append({"worker": i, "words": s["lookups"], "misses": s["misses"],
                            "hit_rate": round(1 - s["misses"] / s["lookups"], 4) if s["lookups"] else None,
                            "cached_words": s["cached_words"]})
    lookups = sum(w["words"] for w in workers_out)
    misses = sum(w["misses"] for w in workers_out)
    return {
        "input": rel(text_path), "input_bytes": input_bytes, "out": rel(out), "out_bytes": out.stat().st_size,
        "tokens": tokens, "docs": docs, "words": words, "blocks": blocks, "workers": workers,
        "bytes_per_token": round(input_bytes / tokens, 4) if tokens else None,
        "tokens_per_word": round(tokens / words, 4) if words else None,
        "cache": {"enabled": use_cache, "hit_rate": round(1 - misses / lookups, 4) if lookups else None,
                  "per_worker": workers_out},
        "seconds": round(seconds, 2), "mb_per_s": round(input_bytes / 1e6 / seconds, 2) if seconds else None,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(tokenizer_path: Path, text_path: Path, bin_path: Path, manifest_path: Path | None = DEFAULT_MANIFEST, *,
           chunk_ids: int = VERIFY_CHUNK_IDS) -> dict[str, Any]:
    """Check that bin_path decodes to exactly the bytes of text_path. Returns the results; "failures" lists what failed."""
    tok = BPETokenizer.load(tokenizer_path)
    vocab = [tok.vocab[i] for i in range(tok.vocab_size)]
    eot = tok.special_tokens[ENDOFTEXT]
    failures: list[str] = []
    res: dict[str, Any] = {"txt": rel(text_path), "bin": rel(bin_path)}

    size = bin_path.stat().st_size
    if size % DTYPE.itemsize:
        failures.append(f"{bin_path} has {size:,} bytes, not a whole number of uint16 IDs")
        size -= size % DTYPE.itemsize
    decoded, raw = hashlib.sha256(), hashlib.sha256()
    eot_ids = 0
    ids = np.memmap(bin_path, dtype=DTYPE, mode="r", shape=(size // DTYPE.itemsize,)) if size else np.zeros(0, DTYPE)
    for start in range(0, len(ids), chunk_ids):
        chunk = ids[start:start + chunk_ids]
        top = int(chunk.max())
        if top >= len(vocab):
            failures.append(f"ID {top:,} (in IDs {start:,}-{start + len(chunk) - 1:,}) is not in the vocabulary")
            break
        raw.update(chunk.tobytes())
        eot_ids += int(np.count_nonzero(chunk == eot))
        decoded.update(b"".join(map(vocab.__getitem__, chunk.tolist())))
    del ids
    res["tokens"] = size // DTYPE.itemsize
    res["bin_sha256"] = raw.hexdigest()

    txt_sha = hashlib.sha256()
    eot_txt = 0
    for block in _iter_chunks(text_path, COUNT_BLOCK_BYTES, [EOT]):  # whole separators, so count() cannot miss one
        txt_sha.update(block)
        eot_txt += block.count(EOT)
    res["txt_sha256"] = txt_sha.hexdigest()
    res["decoded_sha256"] = decoded.hexdigest()
    if res["decoded_sha256"] != res["txt_sha256"]:
        failures.append("the decoded bytes of the .bin differ from the .txt (sha256)")

    res["eot_ids"] = eot_ids
    res["eot_in_txt"] = eot_txt
    if eot_ids != eot_txt:
        failures.append(f"{eot_ids:,} IDs {eot} but {eot_txt:,} {ENDOFTEXT} in the .txt")
    res["manifest"] = None
    if manifest_path is not None and manifest_path.exists():
        splits = json.loads(manifest_path.read_text(encoding="utf-8")).get("splits", {})
        for name, split in splits.items():
            if split.get("sha256") == res["txt_sha256"]:
                res["manifest"] = {"path": rel(manifest_path), "split": name, "docs": split["docs"]}
                if split["docs"] != eot_ids:
                    failures.append(f"{eot_ids:,} IDs {eot} but {split['docs']:,} documents in {name} "
                                    f"({manifest_path})")
    res["failures"] = failures
    res["passed"] = not failures
    return res


def verify_text(res: dict[str, Any]) -> str:
    match = "MATCH" if res["decoded_sha256"] == res["txt_sha256"] else "DIFFERENT"
    m = res["manifest"]
    manifest = (f" = {m['docs']:,} documents of {m['split']} in {m['path']}" if m else
                " (manifest: no split with this sha256, not compared)")
    lines = [
        f"verify {res['bin']} against {res['txt']}:",
        f"  tokens                  {res['tokens']:,}",
        f"  sha256(decoded .bin)    {res['decoded_sha256']}",
        f"  sha256(.txt)            {res['txt_sha256']}  {match}",
        f"  ID 256 count            {res['eot_ids']:,} = {res['eot_in_txt']:,} {ENDOFTEXT} in the .txt{manifest}",
        f"  sha256(.bin)            {res['bin_sha256']}",
        "  VERIFIED: the .bin decodes to exactly the bytes of the .txt" if res["passed"] else
        "  FAILED: " + "; ".join(res["failures"]),
    ]
    return "\n".join(lines)


def report_text(rep: dict[str, Any]) -> str:
    rates = ", ".join(f"{w['hit_rate']:.2%}" if w["hit_rate"] is not None else "-" for w in rep["cache"]["per_worker"])
    cached = sum(w["cached_words"] for w in rep["cache"]["per_worker"])
    hit = rep["cache"]["hit_rate"]
    return "\n".join([
        f"encoded {rep['input']} -> {rep['out']}:",
        f"  tokens                  {rep['tokens']:,} ({rep['out_bytes']:,} bytes)",
        f"  documents (ID 256)      {rep['docs']:,}",
        f"  input bytes             {rep['input_bytes']:,}",
        f"  bytes per token         {rep['bytes_per_token']}",
        f"  tokens per word         {rep['tokens_per_word']} ({rep['words']:,} str.split() words)",
        f"  cache hit rate          {f'{hit:.2%}' if hit is not None else '-'} overall"
        + (f"; per worker {rates}; {cached:,} words cached in all" if rep["cache"]["enabled"] else " (cache off)"),
        f"  wall clock              {rep['seconds']:,.1f} s, {rep['mb_per_s']} MB/s "
        f"({rep['workers']} workers, {rep['blocks']} blocks)",
    ])


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------

def update_meta(meta_path: Path, tokenizer_path: Path, split: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Record this split in meta.json; entries made with another tokenizer are dropped."""
    tok = BPETokenizer.load(tokenizer_path)
    tok_sha = sha256_file(tokenizer_path)
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if old and old.get("tokenizer_sha256") != tok_sha:
        log.warning("%s was written with another tokenizer; starting it again", meta_path)
        old = {}
    splits = {**old.get("splits", {}), split: entry}
    meta: dict[str, Any] = {"vocab_size": tok.vocab_size, "eot_id": tok.special_tokens[ENDOFTEXT],
                            "dtype": "uint16", "byteorder": "little", "tokenizer": rel(tokenizer_path),
                            "tokenizer_sha256": tok_sha}
    for name in sorted(splits, key=lambda s: ({"train": 0, "val": 1}.get(s, 2), s)):
        meta[f"{name}_tokens"] = splits[name]["tokens"]
    meta["pattern"] = tok.pattern
    meta["read_with"] = 'np.memmap(path, dtype="<u2", mode="r")'
    meta["splits"] = splits
    write_json(meta_path, meta)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="train.txt or val.txt (default train.txt)")
    p.add_argument("--out", type=Path, default=None, help="default: the input (or sample) with .bin")
    p.add_argument("--workers", type=int, default=None, help="encoding processes (default: cores - 1)")
    p.add_argument("--verify", action="store_true", help="only verify an existing --out against --input")
    p.add_argument("--no-verify", action="store_true", help="skip the verification after encoding")
    p.add_argument("--sample-mb", type=float, default=None,
                   help="encode the first X MB of the input (whole documents), saved as data/samples/train_<X>mb.txt")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--meta", type=Path, default=None, help="default: meta.json next to --out")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="split document counts (Step 6)")
    p.add_argument("--block-mb", type=float, default=COUNT_BLOCK_BYTES / 2**20, help="read block size, MiB")
    p.add_argument("--verify-chunk", type=int, default=VERIFY_CHUNK_IDS, help="IDs per verification chunk")
    p.add_argument("--no-cache", action="store_true", help="encode every word afresh (for testing)")
    args = p.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        p.error("--workers must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging("encode_corpus")
    if not args.input.exists():
        raise SystemExit(f"no {args.input}")
    if not args.tokenizer.exists():
        raise SystemExit(f"no {args.tokenizer}")
    if args.input.name.lower().startswith("test"):
        raise SystemExit(f"refusing {args.input}: test.txt is not encoded (the test split stays text, for Part C)")
    text_path = make_sample(args.input, args.sample_mb, args.samples_dir) if args.sample_mb else args.input
    out = args.out or text_path.with_suffix(".bin")
    if out.suffix != ".bin" or out.resolve() == text_path.resolve():
        raise SystemExit(f"--out must be a .bin file, got {out}")

    if args.verify:
        if not out.exists():
            raise SystemExit(f"no {out} to verify")
        res = verify(args.tokenizer, text_path, out, args.manifest, chunk_ids=args.verify_chunk)
        log.info("\n%s", verify_text(res))
        return 0 if res["passed"] else 1

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    started = utc()
    log.info("encoding %s (%s bytes) with %s -> %s, %d workers", text_path, f"{text_path.stat().st_size:,}",
             args.tokenizer, out, workers)
    timings: dict[str, float] = {}
    with Timer("encode", timings):
        rep = encode_file(args.tokenizer, text_path, out, workers, use_cache=not args.no_cache,
                          block_bytes=int(args.block_mb * 2**20))
    log.info("\n%s", report_text(rep))

    res = None
    if not args.no_verify:
        with Timer("verify", timings):
            res = verify(args.tokenizer, text_path, out, args.manifest, chunk_ids=args.verify_chunk)
        log.info("verification in %.1f s\n%s", timings["verify"], verify_text(res))

    entry: dict[str, Any] = {
        "txt": rep["input"], "txt_bytes": rep["input_bytes"], "txt_sha256": res["txt_sha256"] if res else None,
        "bin": rep["out"], "bin_bytes": rep["out_bytes"], "bin_sha256": res["bin_sha256"] if res else None,
        "tokens": rep["tokens"], "docs": rep["docs"], "words": rep["words"],
        "bytes_per_token": rep["bytes_per_token"], "tokens_per_word": rep["tokens_per_word"],
        "cache": rep["cache"], "workers": rep["workers"], "blocks": rep["blocks"],
        "encode_seconds": rep["seconds"], "mb_per_s": rep["mb_per_s"],
        "verify_seconds": round(timings["verify"], 2) if res else None,
        "verified": res["passed"] if res else None, "verify_failures": res["failures"] if res else None,
        "started_utc": started, "finished_utc": utc(), "git": git_commit(), "machine": machine_info(),
    }
    meta_path = args.meta or out.parent / "meta.json"
    update_meta(meta_path, args.tokenizer, out.stem, entry)
    log.info("updated %s", meta_path)
    return 0 if res is None or res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
