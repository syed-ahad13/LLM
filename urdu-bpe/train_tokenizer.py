"""CLI: train the byte-level BPE tokenizer on train.txt, save it, and check it (Part B, Steps 7-9).

    # the full run (Step 9): fast trainer, 50,000 tokens
    python train_tokenizer.py --input data/final/train.txt --vocab-size 50000 --out tokenizer/urdu_bpe_50k.json \\
        --workers 7 --checkpoint-every 5000
    python train_tokenizer.py ... --resume          # continue from tokenizer/urdu_bpe_50k.checkpoint.json
    python train_tokenizer.py --sample-mb 50 --out tokenizer/smoke_50mb.json        # smoke run, first 50 MB
    python train_tokenizer.py --trainer reference --sample-mb 1 --vocab-size 3257 \\
        --out tokenizer/reference_1mb_3000.json     # Step 7: reference trainer on the 1 MB sample
    python train_tokenizer.py --check-only          # rerun the post-training checks on --out

Only the training split may shape the merges. The input must be named
train.txt; another name is accepted only with --sample-mb, and a file named
val* or test*, or the --val file itself, is always refused. --sample-mb X
trains on the first X MB (10^6 bytes) of the input, cut at a document
boundary, and saves that text as data/samples/train_<X>mb.txt (--sample-mb 1
writes exactly the Step 7 sample).

Phases, each timed with common.Timer: (1) count_words, i.e. pre-tokenization
in a process pool, cached in data/cache/; (2) init, i.e. words, pair counts,
pair -> words index and heap; (3) merges; (4) save. The reference trainer
does (2) and (3) in one call, timed as "train_reference". Then the
post-training checks (post_training_checks below) run on the saved file and
--val.

Writes:
- <out> and <out stem>.merges.txt (format in bpe.py);
- <out stem>.checkpoint.json every --checkpoint-every merges;
- tokenizer/train_log.json for a full run, or <out stem>.train_log.json for a
  --sample-mb run, so a smoke run never overwrites the full run's log;
- after a full run, the paragraph in docs/training_run.md.

Exits with status 1 if a check fails.
"""

import argparse
import json
import logging
import os
import random
import resource
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bpe import ENDOFTEXT, BPETokenizer, count_words, merges_path, token_display, word_counts_cache_path
from clean.sample_1mb import prefix_at_boundary
from common import REPO_ROOT, Timer, machine_info, setup_logging, sha256_file, write_json

DEFAULT_INPUT = REPO_ROOT / "data" / "final" / "train.txt"
DEFAULT_VAL = REPO_ROOT / "data" / "final" / "val.txt"
DEFAULT_OUT = REPO_ROOT / "tokenizer" / "urdu_bpe_50k.json"
SAMPLES_DIR = REPO_ROOT / "data" / "samples"
CACHE_DIR = REPO_ROOT / "data" / "cache"
TRAINING_RUN_DOC = REPO_ROOT / "docs" / "training_run.md"
DEFAULT_VOCAB_SIZE = 50_000
WINDOW = 5000
SHOW_RANKS = (*range(1, 21), 1000, 10_000, 20_000)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def check_input(path: Path, sample_mb: float | None, val: Path) -> None:
    """Only the training split may shape the merges (see the module docstring)."""
    if not path.exists():
        raise SystemExit(f"no {path}")
    name = path.name.lower()
    if name.startswith(("val", "test")) or (val.exists() and path.resolve().samefile(val.resolve())):
        raise SystemExit(f"refusing {path}: validation and test text must never influence the merges")
    if name != "train.txt" and sample_mb is None:
        raise SystemExit(f"refusing {path}: the tokenizer is trained on train.txt only (use --sample-mb for a "
                         "sample run)")


def sample_name(sample_mb: float) -> str:
    return f"train_{int(sample_mb) if float(sample_mb).is_integer() else sample_mb}mb.txt"


def make_sample(path: Path, sample_mb: float, samples_dir: Path) -> Path:
    """The first sample_mb MB of path, ending at a document boundary, saved as samples_dir/train_<X>mb.txt."""
    data = prefix_at_boundary(path, int(sample_mb * 1_000_000))
    out = samples_dir / sample_name(sample_mb)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_bytes() != data:
        out.write_bytes(data)
    return out


def git_commit() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        return {"hash": None, "dirty": None}
    return {"hash": head, "dirty": dirty}


def rss_mb(who: int) -> float:
    peak = resource.getrusage(who).ru_maxrss
    return round(peak / 1e6 if sys.platform == "darwin" else peak * 1024 / 1e6, 1)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Post-training checks
# ---------------------------------------------------------------------------

def val_documents(val: Path, limit_bytes: int | None = None) -> list[str]:
    """The documents of val.txt, or of its first limit_bytes cut at a document boundary."""
    data = prefix_at_boundary(val, limit_bytes) if limit_bytes else val.read_bytes()
    docs = data.decode("utf-8").split(ENDOFTEXT)
    return docs[:-1] if docs and docs[-1] == "" else docs


def post_training_checks(tok: BPETokenizer, saved: Path, val: Path, expected_merges: int, *,
                         val_bytes: int = 5_000_000, round_trip_docs: int = 1000, seed: int = 1337) -> dict[str, Any]:
    """The Step 9 checks. Returns the results; results["failures"] lists what failed."""
    failures: list[str] = []
    n = len(tok.merges)
    first = tok.first_merge_id
    res: dict[str, Any] = {"merges": n, "vocab_size": tok.vocab_size, "val": str(val)}
    if n != expected_merges:
        failures.append(f"{n} merges, expected {expected_merges}")
    if sorted(tok.vocab) != list(range(first + n)) or tok.vocab_size != first + expected_merges:
        failures.append(f"token IDs are not 0-{first + expected_merges - 1}")
    res["merge_ids"] = [first, first + n - 1] if n else []
    touching = [i + 1 for i, (a, b) in enumerate(tok.merges) if 256 in (a, b)]
    res["merges_touching_256"] = len(touching)
    if touching:
        failures.append(f"merges {touching[:5]} use <|endoftext|> (ID 256)")

    seen = {tok.vocab[i] for i in range(first)}
    repeats = []
    for z in range(first, first + n):
        if tok.vocab[z] in seen:
            repeats.append(z)
        seen.add(tok.vocab[z])
    res["tokens_repeating_earlier_bytes"] = len(repeats)  # expected 0; reported, not an error
    res["repeat_examples"] = [{"id": z, "token": token_display(tok.vocab[z])} for z in repeats[:10]]

    loaded = BPETokenizer.load(saved)
    docs = val_documents(val, val_bytes)
    encoded = [tok.encode_ordinary(d) for d in docs]
    same_merges = loaded.merges == tok.merges and loaded.vocab == tok.vocab
    same_encodings = all(loaded.encode_ordinary(d) == e for d, e in zip(docs, encoded))
    res["save_load"] = {"identical_merges": same_merges, "identical_encodings": same_encodings, "docs": len(docs),
                        "bytes": sum(len(d.encode("utf-8")) for d in docs)}
    if not (same_merges and same_encodings):
        failures.append("load(save(tok)) differs from tok")

    every = val_documents(val)
    picked = random.Random(seed).sample(every, min(round_trip_docs, len(every)))
    bad = [i for i, d in enumerate(picked) if loaded.decode(loaded.encode(d)) != d]
    res["round_trip"] = {"docs": len(picked), "failures": len(bad)}
    if bad or len(picked) < round_trip_docs:
        failures.append(f"round trip: {len(bad)} of {len(picked)} val documents changed"
                        + ("" if len(picked) == round_trip_docs else f" (val.txt has only {len(every)})"))

    tokens = sum(len(e) for e in encoded)
    words = [w for d in docs for w in d.split()]
    size = res["save_load"]["bytes"]
    single: dict[str, bool] = {}
    for w in set(words):
        single[w] = len(loaded.encode_ordinary(" " + w)) == 1
    res["metrics"] = {
        "docs": len(docs), "bytes": size, "words": len(words), "tokens": tokens,
        "tokens_per_word": round(tokens / len(words), 4) if words else None,
        "bytes_per_token": round(size / tokens, 4) if tokens else None,
        "single_token_word_share": round(sum(single[w] for w in words) / len(words), 4) if words else None,
        "definitions": "words: str.split(); a word is single-token if ' ' + word encodes to one token",
    }
    res["failures"] = failures
    res["passed"] = not failures
    return res


def checks_text(res: dict[str, Any]) -> str:
    m = res["metrics"]
    lines = [
        f"merges: {res['merges']:,}; IDs {res['merge_ids'][0] if res['merge_ids'] else '-'}-"
        f"{res['merge_ids'][1] if res['merge_ids'] else '-'}; vocab size {res['vocab_size']:,}; merges using ID 256: "
        f"{res['merges_touching_256']}",
        f"load(save(tok)): merges identical {res['save_load']['identical_merges']}, encodings identical on "
        f"{res['save_load']['bytes'] / 1e6:.2f} MB of val.txt {res['save_load']['identical_encodings']}",
        f"round trip: {res['round_trip']['docs'] - res['round_trip']['failures']:,}/{res['round_trip']['docs']:,} "
        "val documents unchanged",
        f"merged tokens repeating earlier bytes: {res['tokens_repeating_earlier_bytes']}"
        + (f" {res['repeat_examples']}" if res["repeat_examples"] else ""),
        f"val metrics ({m['bytes'] / 1e6:.2f} MB): {m['tokens_per_word']} tokens per word, {m['bytes_per_token']} "
        f"bytes per token, {m['single_token_word_share']:.1%} of words are one token",
        "ALL CHECKS PASSED" if res["passed"] else "FAILED: " + "; ".join(res["failures"]),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def window_rates(train_log: list[dict[str, Any]], resumed: int, window: int = WINDOW) -> list[dict[str, Any]]:
    """Merges per second in each window of `window` merges (the last window may be shorter)."""
    marks = [(resumed, 0.0)] + [(e["merge"], e["seconds"]) for e in train_log
                                if e["merge"] % window == 0 or e is train_log[-1]]
    out = []
    for (m0, t0), (m1, t1) in zip(marks, marks[1:]):
        if m1 > m0:
            out.append({"from": m0 + 1, "to": m1, "seconds": round(t1 - t0, 2),
                        "merges_per_s": round((m1 - m0) / (t1 - t0), 1) if t1 > t0 else None})
    return out


def show_merges(tok: BPETokenizer, ranks: Sequence[int]) -> str:
    rows = [f"{'rank':>6}  {'id':>6}  {'pair':>14}  {'count':>12}  token"]
    n = len(tok.merges)
    for rank in [r for r in ranks if r <= n] + ([n] if n and n not in ranks else []):
        a, b = tok.merges[rank - 1]
        new_id = tok.first_merge_id + rank - 1
        count = f"{tok.merge_counts[rank - 1]:,}" if rank <= len(tok.merge_counts) else "-"
        rows.append(f"{rank:>6}  {new_id:>6}  {f'({a}, {b})':>14}  {count:>12}  {token_display(tok.vocab[new_id])}")
    return "\n".join(rows)


def minutes(seconds: float) -> str:
    return f"{seconds / 60:,.1f} min" if seconds >= 90 else f"{seconds:,.1f} s"


def training_run_paragraph(run: dict[str, Any]) -> str:
    """One paragraph for docs/training_run.md, from a full run's train_log.json."""
    ph, mach, inp, chk = run["phases"], run["machine"], run["train_txt"], run.get("checks") or {}
    merges = run["merges"]
    count = (f"word counting {minutes(ph['count_words'])} ({run['workers']} workers"
             + (", loaded from the cache)" if run.get("count_words_cached") else ")"))
    resumed = (f" The run resumed after {run['resumed_from']:,} merges; the merge time includes "
               f"{minutes(run['merge_seconds_before_resume'])} before the restart." if run.get("resumed_from") else "")
    m = chk.get("metrics") or {}
    checks = ("All post-training checks passed" if chk.get("passed") else
              "Post-training checks FAILED (" + "; ".join(chk.get("failures", [])) + ")" if chk else
              "The post-training checks have not been run")
    if chk:
        checks += (f": {chk['merges']:,} merges with IDs {chk['merge_ids'][0]}-{chk['merge_ids'][1]}, none using "
                   f"<|endoftext|>; load(save) identical on {chk['save_load']['bytes'] / 1e6:.1f} MB of val.txt; "
                   f"{chk['round_trip']['docs'] - chk['round_trip']['failures']:,} of {chk['round_trip']['docs']:,} "
                   f"random val documents round-trip; {chk['tokens_repeating_earlier_bytes']} merged tokens repeat "
                   f"an earlier token's bytes. On {m['bytes'] / 1e6:.1f} MB of val.txt the tokenizer gives "
                   f"{m['tokens_per_word']:.3f} tokens per word and {m['bytes_per_token']:.2f} bytes per token, and "
                   f"{m['single_token_word_share']:.1%} of words are a single token")
    return (
        f"The {run['vocab_size']:,}-token tokenizer `{run['out']}` ({merges:,} merges) was trained with the "
        f"{run['trainer']} trainer on `{inp['path']}` ({inp['bytes'] / 1e6:,.2f} MB, sha256 `{inp['sha256'][:16]}…`; "
        f"{run['word_occurrences']:,} pre-tokens, {run['distinct_words']:,} distinct) from {run['started_utc']} to "
        f"{run['finished_utc']} at commit `{(run['git']['hash'] or '?')[:10]}`"
        + (" (with uncommitted changes)" if run["git"].get("dirty") else "") + ". "
        f"Wall clock by phase: {count}, initialization {minutes(ph['init'])}, merges "
        f"{minutes(ph['merges'])} ({run['merges_per_s_overall']:,.0f} merges per second), saving "
        f"{minutes(ph['save'])}; {minutes(run['total_seconds'])} in total.{resumed} Peak memory: "
        f"{run['peak_rss_mb']['trainer'] / 1000:,.2f} GB in the trainer process, "
        f"{run['peak_rss_mb']['count_workers'] / 1000:,.2f} GB in a counting worker. Machine: {mach['cpu_model']}, "
        f"{mach['physical_cores']} physical / {mach['logical_cores']} logical cores, "
        f"{mach['ram_bytes'] / 2**30:,.0f} GiB RAM, {mach['os_version']}, Python {mach['python_version']}, regex "
        f"{mach['regex_version']}, numpy {mach['numpy_version']}. {checks}."
    )


def write_training_run(run: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Training run\n\n<!-- Generated by train_tokenizer.py from tokenizer/train_log.json; "
                    "`python train_tokenizer.py --check-only` reruns the checks and rewrites it. -->\n\n"
                    + training_run_paragraph(run) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="the training split (default data/final/train.txt)")
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="257 + number of merges")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="default tokenizer/urdu_bpe_50k.json")
    p.add_argument("--trainer", choices=["fast", "reference"], default="fast")
    p.add_argument("--workers", type=int, default=None, help="count_words processes (default: cores - 1)")
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--resume", action="store_true", help="continue from <out stem>.checkpoint.json")
    p.add_argument("--sample-mb", type=float, default=None, help="train on the first X MB of the input only")
    p.add_argument("--val", type=Path, default=DEFAULT_VAL, help="for the post-training checks only")
    p.add_argument("--check-only", action="store_true", help="only rerun the post-training checks on --out")
    p.add_argument("--no-checks", action="store_true", help="skip the post-training checks")
    p.add_argument("--val-mb", type=float, default=5.0, help="val.txt prefix for the save/load and metric checks")
    p.add_argument("--round-trip-docs", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=1000, help=f"progress interval; must divide {WINDOW}")
    p.add_argument("--debug", type=int, default=0, metavar="N", help="check the trainer's invariants (small inputs)")
    p.add_argument("--max-reference-mb", type=float, default=5.0)
    p.add_argument("--force", action="store_true", help="run the reference trainer on a larger input anyway")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="word-count cache (default data/cache)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--log-json", type=Path, default=None,
                   help="default tokenizer/train_log.json (full run) or <out stem>.train_log.json (sample run)")
    p.add_argument("--training-run-doc", type=Path, default=TRAINING_RUN_DOC)
    args = p.parse_args(argv)
    if args.log_every <= 0 or WINDOW % args.log_every:
        p.error(f"--log-every must divide {WINDOW}")
    return args


def log_path(args: argparse.Namespace) -> Path:
    if args.log_json:
        return args.log_json
    return args.out.parent / ("train_log.json" if args.sample_mb is None else f"{args.out.stem}.train_log.json")


def run_checks(args: argparse.Namespace, tok: BPETokenizer, log: logging.Logger) -> dict[str, Any]:
    if not args.val.exists():
        raise SystemExit(f"no {args.val} for the post-training checks (--val, or --no-checks)")
    timings: dict[str, float] = {}
    with Timer("checks", timings):
        res = post_training_checks(tok, args.out, args.val, args.vocab_size - tok.first_merge_id,
                                   val_bytes=int(args.val_mb * 1_000_000), round_trip_docs=args.round_trip_docs)
    res["seconds"] = round(timings["checks"], 1)
    log.info("post-training checks in %.1f s:\n%s", timings["checks"], checks_text(res))
    return res


def check_only(args: argparse.Namespace, log: logging.Logger) -> int:
    if not args.out.exists():
        raise SystemExit(f"no {args.out}")
    tok = BPETokenizer.load(args.out)
    res = run_checks(args, tok, log)
    print(checks_text(res))
    path = log_path(args)
    if path.exists():
        run = json.loads(path.read_text(encoding="utf-8"))
        run["checks"] = res
        write_json(path, run)
        if run.get("sample_mb") is None and run.get("trainer") == "fast":
            write_training_run(run, args.training_run_doc)
            print(f"updated {path} and {args.training_run_doc}")
    return 0 if res["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("train_tokenizer")
    bpe_log = logging.getLogger("bpe")
    bpe_log.handlers = log.handlers  # the trainer's progress and warnings go to the same log
    bpe_log.setLevel(logging.INFO)
    if args.check_only:
        return check_only(args, log)

    started = utc()
    check_input(args.input, args.sample_mb, args.val)
    train_bytes, train_sha = args.input.stat().st_size, sha256_file(args.input)
    text_path = make_sample(args.input, args.sample_mb, args.samples_dir) if args.sample_mb else args.input
    text_bytes = text_path.stat().st_size
    text_sha = sha256_file(text_path) if text_path != args.input else train_sha
    if args.trainer == "reference" and text_bytes > args.max_reference_mb * 1e6 and not args.force:
        raise SystemExit(f"{text_path} is {text_bytes / 1e6:,.1f} MB; the reference trainer is meant for the 1 MB "
                         "sample (--sample-mb 1, or --max-reference-mb / --force)")
    log.info("training %s on %s (%d bytes) to vocab size %d -> %s", args.trainer, text_path, text_bytes,
             args.vocab_size, args.out)

    timings: dict[str, float] = {}
    tok = BPETokenizer()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    command = " ".join(["python", "train_tokenizer.py", *(shlex.quote(a) for a in raw_args)])
    run: dict[str, Any] = {"trainer": args.trainer, "command": command,
                           "out": str(args.out), "vocab_size": args.vocab_size,
                           "train_txt": {"path": str(args.input), "bytes": train_bytes, "sha256": train_sha},
                           "sample_mb": args.sample_mb,
                           "trained_on": {"path": str(text_path), "bytes": text_bytes, "sha256": text_sha},
                           "started_utc": started, "git": git_commit()}
    if args.trainer == "reference":
        with Timer("read", timings):
            text = text_path.read_text(encoding="utf-8")
        with Timer("train_reference", timings):
            summary = tok.train_reference(text, args.vocab_size)
        run.update(word_occurrences=summary["word_occurrences"], distinct_words=None, workers=1, resumed_from=0)
        merge_seconds = timings["train_reference"]
    else:
        cache_dir = None if args.no_cache else args.cache_dir
        cached = cache_dir is not None and word_counts_cache_path(cache_dir, text_sha).exists()
        with Timer("count_words", timings):
            counts = count_words(text_path, args.workers, cache_dir=cache_dir, file_sha256=text_sha)
        run.update(distinct_words=len(counts), word_occurrences=sum(counts.values()), count_words_cached=cached,
                   workers=args.workers or max(1, (os.cpu_count() or 2) - 1))
        checkpoint = args.out.with_name(f"{args.out.stem}.checkpoint.json")
        with Timer("init", timings):
            state = tok.prepare_training(counts, args.vocab_size, checkpoint_path=checkpoint, resume=args.resume)
        del counts
        with Timer("merges", timings):
            summary = tok.run_merges(state, log_every=args.log_every, checkpoint_every=args.checkpoint_every,
                                     checkpoint_path=checkpoint, debug=args.debug)
        del state
        run.update(resumed_from=summary["resumed_from"], heap_compactions=summary["heap_compactions"],
                   merge_seconds_before_resume=round(summary["merge_seconds_total"] - summary["merge_seconds"], 2),
                   progress=tok.train_log, merges_per_s_windows=window_rates(tok.train_log, summary["resumed_from"]))
        merge_seconds = timings["merges"]
    meta = {"trainer": args.trainer, "input": str(text_path), "input_bytes": text_bytes, "input_sha256": text_sha,
            "train_txt_sha256": train_sha, "requested_vocab_size": args.vocab_size,
            "phase_seconds": {k: round(v, 2) for k, v in timings.items()}, "started_utc": started,
            "git": run["git"], "machine": machine_info()}
    meta["train_seconds"] = round(sum(timings.values()), 2)
    with Timer("save", timings):
        tok.save(args.out, meta=meta)

    done_here = len(tok.merges) - run.get("resumed_from", 0)
    run.update(merges=len(tok.merges), phases={k: round(v, 2) for k, v in timings.items()},
               total_seconds=round(sum(timings.values()), 2), finished_utc=utc(),
               merges_per_s_overall=round(done_here / merge_seconds, 1) if merge_seconds > 0 else None,
               peak_rss_mb={"trainer": rss_mb(resource.RUSAGE_SELF),
                            "count_workers": rss_mb(resource.RUSAGE_CHILDREN)},
               machine=machine_info())
    log.info("%d merges; phases %s; total %.1f s; saved %s and %s", len(tok.merges), run["phases"],
             run["total_seconds"], args.out, merges_path(args.out))
    print(f"\nMerges at ranks 1-20, 1,000, 10,000, 20,000 and the last ({len(tok.merges):,}):")
    print(show_merges(tok, SHOW_RANKS))

    passed = True
    if not args.no_checks:
        run["checks"] = run_checks(args, tok, log)
        passed = run["checks"]["passed"]
    path = log_path(args)
    write_json(path, run)
    if args.sample_mb is None and args.trainer == "fast":
        write_training_run(run, args.training_run_doc)
    summary_keys = ("trainer", "merges", "distinct_words", "word_occurrences", "phases", "total_seconds",
                    "merges_per_s_overall", "peak_rss_mb", "resumed_from")
    print(f"\n{path}:")
    print(json.dumps({k: run.get(k) for k in summary_keys}, indent=2, ensure_ascii=False))
    if "checks" in run:
        print("\n" + checks_text(run["checks"]))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
