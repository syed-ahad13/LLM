"""CLI: train a BPETokenizer and save it to tokenizer/<name>.json (+ <name>.merges.txt).

    python train_tokenizer.py --name urdu_bpe_50k                     # fast trainer, train.txt, 50,000 tokens
    python train_tokenizer.py --trainer reference --input data/samples/train_1mb.txt --vocab-size 3257 \\
        --name reference_1mb_3000                                     # Step 7: the reference trainer

fast (default): count_words reads the input in 16 MB blocks in a process pool
and caches the word counts in data/cache/ (keyed by the file's sha256, the
pattern and the special tokens), then BPETokenizer.train learns the merges.
Progress goes to logs/train_tokenizer.log every --log-every merges, and the
merges so far are saved to tokenizer/<name>.checkpoint.json every
--checkpoint-every merges; --resume continues from that checkpoint.

reference: reads the input whole and recounts every pair after every merge.
It is refused on inputs over --max-reference-mb unless --force is given.

Records the wall-clock time per phase, the input's size and sha256, and
common.machine_info() in the tokenizer's meta, as the handout asks for the
report, and prints the first --show merges in readable form.
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bpe import BPETokenizer, count_words, merges_path, token_display
from common import REPO_ROOT, machine_info, setup_logging, sha256_file

TOKENIZER_DIR = REPO_ROOT / "tokenizer"
DEFAULT_INPUT = REPO_ROOT / "data" / "final" / "train.txt"
CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_VOCAB_SIZE = 50_000


def first_merges(tok: BPETokenizer, n: int) -> str:
    rows = [f"{'rank':>5}  {'id':>6}  {'pair':>12}  {'count':>11}  token"]
    for rank, (a, b) in enumerate(tok.merges[:n], start=1):
        new_id = tok.first_merge_id + rank - 1
        count = f"{tok.merge_counts[rank - 1]:,}" if rank <= len(tok.merge_counts) else "-"
        rows.append(f"{rank:>5}  {new_id:>6}  {f'({a}, {b})':>12}  {count:>11}  {token_display(tok.vocab[new_id])}")
    return "\n".join(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainer", choices=["fast", "reference"], default="fast")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="257 + number of merges")
    p.add_argument("--name", required=True, help="saved as tokenizer/<name>.json")
    p.add_argument("--out-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--show", type=int, default=30, help="merges to print at the end")
    p.add_argument("--workers", type=int, default=None, help="count_words processes (default: cores - 1)")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="word-count cache (default data/cache)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--resume", action="store_true", help="continue from tokenizer/<name>.checkpoint.json")
    p.add_argument("--debug", type=int, default=0, metavar="N",
                   help="fast trainer: check the invariants, with a full recount every N merges (small inputs)")
    p.add_argument("--verbose", action="store_true", help="reference trainer: log every merge")
    p.add_argument("--max-reference-mb", type=float, default=5.0)
    p.add_argument("--force", action="store_true", help="run the reference trainer on a larger input anyway")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("train_tokenizer")
    bpe_log = logging.getLogger("bpe")
    bpe_log.handlers = log.handlers  # the trainer's progress and warnings go to the same log
    bpe_log.setLevel(logging.INFO)
    if not args.input.exists():
        raise SystemExit(f"no {args.input}")
    size = args.input.stat().st_size
    if args.trainer == "reference" and size > args.max_reference_mb * 1e6 and not args.force:
        raise SystemExit(f"{args.input} is {size / 1e6:,.1f} MB; the reference trainer is meant for the 1 MB "
                         f"sample (--max-reference-mb, --force)")
    log.info("training %s on %s (%d bytes) to vocab size %d", args.trainer, args.input, size, args.vocab_size)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tok = BPETokenizer()
    meta: dict[str, object] = {"trainer": args.trainer, "input": str(args.input), "input_bytes": size,
                               "input_sha256": sha256_file(args.input), "requested_vocab_size": args.vocab_size,
                               "started_utc": started}
    t0 = time.perf_counter()
    if args.trainer == "reference":
        summary = tok.train_reference(args.input.read_text(encoding="utf-8"), args.vocab_size, verbose=args.verbose)
        meta["word_occurrences"] = summary["word_occurrences"]
    else:
        counts = count_words(args.input, args.workers, cache_dir=None if args.no_cache else args.cache_dir)
        meta["count_seconds"] = round(time.perf_counter() - t0, 2)
        meta["distinct_words"] = len(counts)
        meta["word_occurrences"] = sum(counts.values())
        checkpoint = args.out_dir / f"{args.name}.checkpoint.json"
        summary = tok.train(counts, args.vocab_size, log_every=args.log_every,
                            checkpoint_every=args.checkpoint_every, checkpoint_path=checkpoint,
                            resume=args.resume, debug=args.debug)
        del counts
        meta.update({k: summary[k] for k in ("setup_seconds", "merge_seconds", "resumed_from", "heap_compactions",
                                             "peak_rss_mb")})
        meta["train_log"] = tok.train_log
    seconds = time.perf_counter() - t0
    meta["train_seconds"] = round(seconds, 2)
    meta["machine"] = machine_info()
    path = tok.save(args.out_dir / f"{args.name}.json", meta=meta)
    log.info("%d merges in %.1f s (%.2f ms per merge); saved %s and %s", len(tok.merges), seconds,
             1000 * seconds / max(1, len(tok.merges)), path, merges_path(path))
    print(f"\nFirst {min(args.show, len(tok.merges))} merges:")
    print(first_merges(tok, args.show))
    print(f"\n{args.trainer} trainer: {len(tok.merges):,} merges on {size:,} bytes in {seconds:,.1f} s "
          f"wall clock ({machine_info()['cpu_model']}, Python {machine_info()['python_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
