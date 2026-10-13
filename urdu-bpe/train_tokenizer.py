"""CLI: train a BPETokenizer and save it to tokenizer/<name>.json (+ <name>.merges.txt).

    python train_tokenizer.py --trainer reference --input data/samples/train_1mb.txt --vocab-size 3257 \\
        --name reference_1mb_3000                  # Step 7: 3,000 merges with the reference trainer

The text is read whole into memory. The reference trainer recounts every pair
after every merge, so it is refused on inputs over --max-reference-mb unless
--force is given; the full train.txt needs the fast trainer (Step 8).

Records the wall-clock time, the input's size and sha256, and
common.machine_info() in the tokenizer's meta, as the handout asks for the
report, and prints the first --show merges in readable form.
"""

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bpe import BPETokenizer, merges_path, token_display
from common import REPO_ROOT, machine_info, setup_logging, sha256_file

TOKENIZER_DIR = REPO_ROOT / "tokenizer"
DEFAULT_INPUT = REPO_ROOT / "data" / "final" / "train.txt"
DEFAULT_VOCAB_SIZE = 50_000


def first_merges(tok: BPETokenizer, n: int) -> str:
    rows = [f"{'rank':>5}  {'id':>6}  {'pair':>12}  {'count':>9}  token"]
    for rank, (a, b) in enumerate(tok.merges[:n], start=1):
        new_id = tok.first_merge_id + rank - 1
        count = f"{tok.merge_counts[rank - 1]:,}" if rank <= len(tok.merge_counts) else "-"
        rows.append(f"{rank:>5}  {new_id:>6}  {f'({a}, {b})':>12}  {count:>9}  {token_display(tok.vocab[new_id])}")
    return "\n".join(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainer", choices=["reference"], default="reference",
                   help="reference: full recount after every merge (the fast trainer comes in Step 8)")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="257 + number of merges")
    p.add_argument("--name", required=True, help="saved as tokenizer/<name>.json")
    p.add_argument("--out-dir", type=Path, default=TOKENIZER_DIR)
    p.add_argument("--show", type=int, default=30, help="merges to print at the end")
    p.add_argument("--verbose", action="store_true", help="log every merge")
    p.add_argument("--max-reference-mb", type=float, default=5.0)
    p.add_argument("--force", action="store_true", help="run the reference trainer on a larger input anyway")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("train_tokenizer")
    logging.getLogger("bpe").handlers = log.handlers  # the trainer's warnings and --verbose go to the same log
    logging.getLogger("bpe").setLevel(logging.INFO)
    if not args.input.exists():
        raise SystemExit(f"no {args.input}")
    size = args.input.stat().st_size
    if args.trainer == "reference" and size > args.max_reference_mb * 1e6 and not args.force:
        raise SystemExit(f"{args.input} is {size / 1e6:,.1f} MB; the reference trainer is meant for the 1 MB "
                         f"sample (--max-reference-mb, --force)")
    text = args.input.read_text(encoding="utf-8")
    log.info("training %s on %s (%d bytes) to vocab size %d", args.trainer, args.input, size, args.vocab_size)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tok = BPETokenizer()
    t0 = time.perf_counter()
    summary = tok.train_reference(text, args.vocab_size, verbose=args.verbose)
    seconds = time.perf_counter() - t0
    meta = {
        "trainer": args.trainer,
        "input": str(args.input),
        "input_bytes": size,
        "input_sha256": sha256_file(args.input),
        "requested_vocab_size": args.vocab_size,
        "word_occurrences": summary["word_occurrences"],
        "train_seconds": round(seconds, 2),
        "started_utc": started,
        "machine": machine_info(),
    }
    path = tok.save(args.out_dir / f"{args.name}.json", meta=meta)
    log.info("%d merges in %.1f s (%.1f ms per merge); saved %s and %s", len(tok.merges), seconds,
             1000 * seconds / max(1, len(tok.merges)), path, merges_path(path))
    print(f"\nFirst {min(args.show, len(tok.merges))} merges:")
    print(first_merges(tok, args.show))
    print(f"\n{args.trainer} trainer: {len(tok.merges):,} merges on {size:,} bytes in {seconds:,.1f} s "
          f"wall clock ({machine_info()['cpu_model']}, Python {machine_info()['python_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
