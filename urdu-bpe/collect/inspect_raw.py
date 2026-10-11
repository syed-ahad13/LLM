"""Print what each downloaded source in data/raw/ looks like.

    python collect/inspect_raw.py                       # every source folder in data/raw/
    python collect/inspect_raw.py --source ufal_umc

For every source: its files and sizes, then the first 3 records decoded as UTF-8.
Extra checks:
- urdu_news_1m: column names, row count, and the share of 1,000 sampled rows
  containing Urdu punctuation (U+06D4, U+060C, U+061F). The dataset's methods
  text says only "Urdu text and numbers" were kept, so the report also checks
  letters and marks that such a filter can drop (alef madda, hamza forms, diacritics).
- ufal_umc: the first 50 lines, and a line-order test: mean word-set Jaccard of
  consecutive line pairs vs random line pairs over the first 20,000 lines.
  Shuffled sentences give a ratio near 1; preserved document order gives clearly more.
- fineweb2_urd: parquet schema and 3 rows (text, url, date, language_score).

Unfinished downloads (.part) are read as far as they go and marked PARTIAL;
counts from them cover only the downloaded prefix.
"""

import argparse
import bz2
import csv
import gzip
import lzma
import random
import sys
import tarfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any, TextIO

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

URDU_PUNCTUATION = ["۔", "،", "؟"]  # ۔ ، ؟
ASCII_PUNCTUATION = [".", ",", "?"]
# Characters normal Urdu text contains that an "Urdu letters and digits only" filter may have removed.
FILTER_CASUALTIES = {
    "آ": "alef madda (آپ, آج)",
    "ئ": "yeh with hamza (گئی, لئے)",
    "أ": "alef with hamza above",
    "ۂ": "heh goal with hamza (خانۂ)",
    "ۓ": "yeh barree with hamza",
}
DIACRITICS = {chr(c) for c in range(0x064B, 0x0653)} | {"ٰ"}
TRUNCATED = (EOFError, OSError, lzma.LZMAError)


# ---------------------------------------------------------------------------
# Files and readers (all tolerate a file that stops mid-stream)
# ---------------------------------------------------------------------------

def is_partial(path: Path) -> bool:
    return path.name.endswith(".part")


def logical_name(path: Path) -> str:
    return path.name.removesuffix(".part")


def data_files(source_dir: Path) -> list[Path]:
    """Data files in a source folder: no manifest, no resume metadata, no stale .part next to a finished file."""
    files = [
        p for p in sorted(source_dir.iterdir())
        if p.is_file() and not p.name.startswith(".") and p.name != "MANIFEST.json" and not p.name.endswith(".part.json")
    ]
    finished = {p.name for p in files if not is_partial(p)}
    return [p for p in files if not (is_partial(p) and logical_name(p) in finished)]


def open_text(path: Path) -> TextIO:
    name = logical_name(path)
    opener = {".gz": gzip.open, ".xz": lzma.open, ".bz2": bz2.open}.get(Path(name).suffix)
    if opener:
        return opener(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def iter_lines(path: Path) -> Iterator[str]:
    """Lines without their newline. Stops quietly where a partial or truncated file ends."""
    try:
        with open_text(path) as f:
            for line in f:
                yield line.rstrip("\n")
    except TRUNCATED:
        return


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def iter_wiki_pages(path: Path) -> Iterator[dict[str, str]]:
    """<page> records from a MediaWiki XML dump (.xml.bz2): title, ns, id and wikitext."""
    try:
        with bz2.open(path, "rb") as f:
            for _, elem in ET.iterparse(f, events=("end",)):
                if _local(elem.tag) != "page":
                    continue
                page = {_local(child.tag): child for child in elem}
                text = ""
                if (revision := page.get("revision")) is not None:
                    text = next((c.text or "" for c in revision if _local(c.tag) == "text"), "")
                yield {
                    "title": page["title"].text or "",
                    "ns": page["ns"].text or "",
                    "id": page["id"].text or "",
                    "text": text,
                }
                elem.clear()
    except (*TRUNCATED, ET.ParseError):
        return


def read_csv_header(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        return next(csv.reader(f), [])


def iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    """Rows as dicts. In a partial file the last row may be cut off, so it is dropped."""
    csv.field_size_limit(2**31 - 1)
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return
        pending: list[str] | None = None
        try:
            for row in reader:
                if pending is not None:
                    yield dict(zip(header, pending))
                pending = row
        except csv.Error:  # a partial file can end inside a quoted field
            return
        if pending is not None and not is_partial(path):
            yield dict(zip(header, pending))


def iter_tar_texts(path: Path) -> Iterator[tuple[str, str]]:
    """(member name, text) for each regular file in a .tar.gz, read as a stream."""
    try:
        with tarfile.open(path, "r|gz") as tf:
            for member in tf:
                if member.isfile() and (f := tf.extractfile(member)) is not None:
                    yield member.name, f.read().decode("utf-8", errors="replace")
    except (*TRUNCATED, tarfile.TarError):
        return


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def order_test(lines: Sequence[str], seed: int = 1337) -> tuple[float, float]:
    """(mean Jaccard of consecutive line pairs, mean Jaccard of as many random pairs)."""
    sets = [set(line.split()) for line in lines if line.strip()]
    if len(sets) < 3:
        raise ValueError("need at least 3 non-empty lines")
    consecutive = [jaccard(a, b) for a, b in zip(sets, sets[1:])]
    rng = random.Random(seed)
    shuffled = []
    for _ in consecutive:
        i, j = rng.sample(range(len(sets)), 2)
        shuffled.append(jaccard(sets[i], sets[j]))
    return fmean(consecutive), fmean(shuffled)


def order_verdict(consecutive: float, random_pairs: float) -> str:
    ratio = consecutive / random_pairs if random_pairs else float("inf")
    if ratio >= 1.5:
        return f"{ratio:.2f}x: consecutive lines are related, so document order looks preserved"
    if ratio <= 1.1:
        return f"{ratio:.2f}x: no sign of document order (lines look shuffled)"
    return f"{ratio:.2f}x: weak signal; inspect the lines before relying on order"


def reservoir_sample(rows: Iterator[dict[str, str]], k: int, seed: int) -> tuple[list[dict[str, str]], int]:
    """Uniform sample of k rows in one pass, plus the total row count."""
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    n = 0
    for n, row in enumerate(rows, start=1):
        if len(sample) < k:
            sample.append(row)
        elif (j := rng.randrange(n)) < k:
            sample[j] = row
    return sample, n


def share_containing(texts: Sequence[str], chars: Sequence[str]) -> float:
    return sum(any(c in t for c in chars) for t in texts) / len(texts) if texts else 0.0


def char_label(c: str) -> str:
    return f"U+{ord(c):04X} {c} {unicodedata.name(c, '?').title()}"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def clip(text: str, max_chars: int) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def print_records(path: Path, n: int, max_chars: int) -> None:
    name = logical_name(path)
    if name.endswith(".parquet"):
        return  # shown by report_parquet
    if name.endswith(".xml.bz2"):
        records: Iterator[Any] = iter_wiki_pages(path)
    elif name.endswith(".csv"):
        records = iter_csv_rows(path)
    elif name.endswith(".tar.gz"):
        records = (f"{member}: {text}" for member, text in iter_tar_texts(path))
    else:
        records = iter_lines(path)
    print(f"  first {n} records:")
    shown = 0
    for record in records:
        if shown == n:
            break
        if isinstance(record, dict):
            body = "; ".join(f"{k}={clip(str(v), max_chars)}" for k, v in record.items())
        else:
            body = clip(str(record), max_chars)
        print(f"    [{shown + 1}] {body}")
        shown += 1
    if shown == 0:
        print("    (none readable yet)")


def report_news(path: Path, sample_rows: int, seed: int) -> None:
    columns = read_csv_header(path)
    print(f"  columns: {columns}")
    sample, n = reservoir_sample(iter_csv_rows(path), sample_rows, seed)
    print(f"  rows: {n:,}" + ("  (downloaded prefix only)" if is_partial(path) else ""))
    text_columns = [c for c in columns if c.lower() in ("news text", "headline")] or columns
    for column in text_columns:
        texts = [row.get(column, "") for row in sample]
        print(f"  share of {len(texts):,} sampled rows whose {column!r} contains:")
        for c in URDU_PUNCTUATION:
            print(f"    {char_label(c):40s} {share_containing(texts, [c]):6.1%}")
        print(f"    {'any of the three':40s} {share_containing(texts, URDU_PUNCTUATION):6.1%}")
        print(f"    {'ASCII . , ? (for comparison)':40s} {share_containing(texts, ASCII_PUNCTUATION):6.1%}")
        for c, why in FILTER_CASUALTIES.items():
            print(f"    {char_label(c) + ' — ' + why:40s} {share_containing(texts, [c]):6.1%}")
        print(f"    {'any diacritic (U+064B-0652, U+0670)':40s} {share_containing(texts, sorted(DIACRITICS)):6.1%}")
        inventory = Counter(ch for t in texts for ch in t)
        print(f"    distinct characters: {len(inventory)}")


def report_ufal(path: Path, head_lines: int, order_lines: int, seed: int, max_chars: int) -> None:
    lines = []
    for line in iter_lines(path):
        lines.append(line)
        if len(lines) >= max(head_lines, order_lines):
            break
    if is_partial(path) and lines:
        lines.pop()  # the last line of a partial file may be cut off
    print(f"  first {min(head_lines, len(lines))} lines:")
    for i, line in enumerate(lines[:head_lines], start=1):
        print(f"    {i:3d}  {clip(line, max_chars)}")
    window = lines[:order_lines]
    blank = sum(1 for line in window if not line.strip())
    try:
        consecutive, random_pairs = order_test(window, seed)
    except ValueError as e:
        print(f"  order test: skipped ({e})")
        return
    print(f"  order test over {len(window):,} lines ({blank:,} blank):")
    print(f"    mean Jaccard, consecutive pairs: {consecutive:.4f}")
    print(f"    mean Jaccard, random pairs:      {random_pairs:.4f}")
    print(f"    ratio {order_verdict(consecutive, random_pairs)}")


def report_parquet(path: Path, n: int, max_chars: int) -> None:
    if is_partial(path):
        print("  PARTIAL parquet: schema and rows live in the footer at the end of the file, so nothing is readable yet")
        return
    pf = pq.ParquetFile(path)
    md = pf.metadata
    print("  schema:")
    for line in str(pf.schema_arrow).splitlines():
        print(f"    {line}")
    names = pf.schema_arrow.names
    print(f"  rows: {md.num_rows:,} in {md.num_row_groups:,} row groups")
    if "text" in names:
        ti = names.index("text")
        unc = sum(md.row_group(i).column(ti).total_uncompressed_size for i in range(md.num_row_groups))
        print(f"  text column, uncompressed (from metadata): about {unc / 1e9:.2f} GB")
    columns = [c for c in ("text", "url", "date", "language_score") if c in names]
    table = pf.read_row_group(0, columns=columns).slice(0, n)
    print(f"  first {table.num_rows} rows ({', '.join(columns)}):")
    for i, row in enumerate(table.to_pylist(), start=1):
        print(f"    [{i}] " + "; ".join(f"{k}={clip(str(v), max_chars)}" for k, v in row.items()))


def inspect_source(source_dir: Path, args: argparse.Namespace) -> None:
    sid = source_dir.name
    files = data_files(source_dir)
    print(f"=== {sid} ===")
    if not files:
        print("  no data files yet")
        return
    for p in files:
        flag = "  PARTIAL" if is_partial(p) else ""
        print(f"  {p.name}  {p.stat().st_size:,} bytes{flag}")
    for p in files:
        print(f"-- {p.name}")
        name = logical_name(p)
        if name.endswith(".parquet"):
            report_parquet(p, args.records, args.max_chars)
            continue
        print_records(p, args.records, args.max_chars)
        if sid == "urdu_news_1m" and name.endswith(".csv"):
            report_news(p, args.sample_rows, args.seed)
        elif sid == "ufal_umc":
            report_ufal(p, args.head_lines, args.order_lines, args.seed, args.max_chars)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="all", help="source id (folder in data/raw/), or 'all' (default)")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--records", type=int, default=3, help="records to show per file (default 3)")
    p.add_argument("--max-chars", type=int, default=300, help="truncate each shown field (default 300)")
    p.add_argument("--sample-rows", type=int, default=1000, help="urdu_news_1m rows to sample (default 1000)")
    p.add_argument("--head-lines", type=int, default=50, help="ufal_umc lines to print (default 50)")
    p.add_argument("--order-lines", type=int, default=20000, help="ufal_umc lines for the order test (default 20000)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.raw_dir.is_dir():
        print(f"no {args.raw_dir}; download something first", file=sys.stderr)
        return 1
    dirs = sorted(d for d in args.raw_dir.iterdir() if d.is_dir())
    if args.source != "all":
        dirs = [d for d in dirs if d.name == args.source]
        if not dirs:
            print(f"no folder {args.raw_dir / args.source}", file=sys.stderr)
            return 1
    for d in dirs:
        inspect_source(d, args)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
