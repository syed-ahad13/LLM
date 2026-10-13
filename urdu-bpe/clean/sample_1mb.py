"""Make the 1 MB training sample for the reference trainer (Step 7) and the fast-vs-reference check (Step 8).

    python clean/sample_1mb.py                     # data/final/train.txt -> data/samples/train_1mb.txt
    python clean/sample_1mb.py --bytes 200000 --out /tmp/train_200kb.txt

Takes whole documents from the start of train.txt, each followed by its
<|endoftext|>, and stops at the last document boundary at or below --bytes.
train.txt is already shuffled across sources (clean/split.py), so its start
is a fair sample of the training text.

Fallback when Step 6 has not run: --from-clean-samples builds the file from
the Step 5b samples (data/samples/clean/<source>.sample.jsonl), one document
per source in turn. Rebuild from train.txt once it exists, so the reported
1 MB test uses real training text.
"""

import argparse
import itertools
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.budget import FINAL_DIR  # noqa: E402
from clean.dedup import ENDOFTEXT, PRIORITY  # noqa: E402
from clean.extract_common import SAMPLES_DIR  # noqa: E402
from common import iter_jsonl, setup_logging, sha256_file  # noqa: E402

EOT = ENDOFTEXT.encode("utf-8")
DEFAULT_BYTES = 1_000_000
OUT = SAMPLES_DIR / "train_1mb.txt"


def prefix_at_boundary(path: Path, limit: int) -> bytes:
    """The longest prefix of a split file that ends just after a separator and holds at most limit bytes."""
    with open(path, "rb") as f:
        head = f.read(limit)
    end = head.rfind(EOT)
    if end == -1:
        raise SystemExit(f"{path}: no complete document in the first {limit:,} bytes")
    return head[:end + len(EOT)]


def from_clean_samples(samples_dir: Path, limit: int) -> bytes:
    """Documents from the Step 5b samples, one source at a time in turn, until limit bytes."""
    paths = sorted(samples_dir.glob("*.sample.jsonl"),
                   key=lambda p: PRIORITY.index(p.name.split(".")[0]) if p.name.split(".")[0] in PRIORITY else 99)
    if not paths:
        raise SystemExit(f"no *.sample.jsonl in {samples_dir}")
    streams: list[Iterator[str]] = [(d["text"] for d in iter_jsonl(p)) for p in paths]
    out: list[bytes] = []
    size = 0
    for docs in itertools.zip_longest(*streams):
        for text in docs:
            if text is None:
                continue
            piece = text.encode("utf-8") + EOT
            if size + len(piece) > limit:
                return b"".join(out)
            out.append(piece)
            size += len(piece)
    return b"".join(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=FINAL_DIR / "train.txt")
    p.add_argument("--bytes", type=int, default=DEFAULT_BYTES, help="largest size (default 1,000,000)")
    p.add_argument("--out", type=Path, default=OUT, help="default data/samples/train_1mb.txt")
    p.add_argument("--from-clean-samples", action="store_true",
                   help="build from data/samples/clean/*.sample.jsonl instead (Step 6 not run yet)")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR / "clean")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("sample_1mb")
    if args.from_clean_samples:
        data, source = from_clean_samples(args.samples_dir, args.bytes), f"{args.samples_dir} (fallback)"
    else:
        if not args.input.exists():
            raise SystemExit(f"no {args.input}; run Step 6 (clean/split.py) or use --from-clean-samples")
        data, source = prefix_at_boundary(args.input, args.bytes), str(args.input)
    data.decode("utf-8")  # strict: the sample must be valid UTF-8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    docs = data.count(EOT)
    log.info("%s: %d documents, %d bytes from %s; sha256 %s", args.out, docs, len(data), source,
             sha256_file(args.out))
    if args.from_clean_samples:
        log.warning("built from the Step 5b samples; rebuild from train.txt once Step 6 has run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
