"""Pre-flight checks, the submission archive and the shared-drive bundle (Step 12).

    python make_submission.py                 # every check, then submission/urdu_bpe_456094.zip and SHA256SUMS
    python make_submission.py --skip-drive    # a machine without data/final: the drive bundle is NOT checked
    make submission                           # the same as the first line

Pre-flight checks, all run and all reported; any failure stops the script
before anything is written:
- every handout deliverable is present: normalize.py, bpe.py with class
  BPETokenizer, train_tokenizer.py, encode_corpus.py, the collection and
  cleaning scripts, the tokenizer file, DATA_CARD.md, report/report.pdf, a
  README with the "Rebuild the tokenizer from train.txt" section, and the
  drive files (data/final/train.txt, val.txt, test.txt, train.bin, val.bin,
  meta.json, manifest.json);
- the tokenizer has exactly 49,743 merges (vocabulary 50,000);
- report/report.pdf has at most 4 pages;
- DATA_CARD.md has no <!-- REVIEW --> marker left;
- no PASTE placeholder (an opening square bracket followed by PASTE, as the
  briefs write them) is left in any file of the archive or in report/report.tex;
- pytest -q is green, and so is tests/test_forbidden_imports.py;
- the drive files match their recorded sha256: train/val/test.txt against
  data/final/manifest.json, train.bin and val.bin against data/final/meta.json, and
  meta.json names the delivered tokenizer (its sha256).

Then:
- submission/urdu_bpe_<regno>.zip with the code, tests, tokenizer, data card,
  report, README, evidence/ and docs/ (no data/, logs/, .venv/, __pycache__/
  or LaTeX build files), members in sorted order with fixed timestamps, so the
  same tree gives the same archive;
- data/final/SHA256SUMS (sha256sum format) over the drive files, and the list
  of files to upload with their sizes. The link must stay private to course
  staff (Makhzan, docs/decisions.md 2026-10-11).
"""

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from report.checks import pdf_pages  # noqa: E402

REGNO = "456094"
EXPECTED_MERGES = 49_743
EXPECTED_VOCAB = 50_000
MAX_PAGES = 4
REVIEW = "<!-- REVIEW -->"
PLACEHOLDER = "[" + "PASTE"  # built here so this file does not match itself
ARCHIVE_ITEMS = (
    "normalize.py", "bpe.py", "train_tokenizer.py", "encode_corpus.py", "common.py", "make_submission.py",
    "collect", "clean", "analysis", "tests", "requirements.txt", "requirements-analysis.txt",
    "requirements-dev.txt", "Makefile", "pytest.ini", "tokenizer/urdu_bpe_50k.json",
    "tokenizer/urdu_bpe_50k.merges.txt", "tokenizer/train_log.json", "DATA_CARD.md", "report/report.pdf",
    "README.md", "evidence", "docs",
)
EXCLUDE_DIRS = {"data", "logs", ".venv", "__pycache__", ".pytest_cache", ".git", "submission"}
EXCLUDE_SUFFIXES = (".pyc", ".aux", ".log", ".fls", ".fdb_latexmk", ".synctex.gz", ".out", ".toc", ".bbl",
                    ".blg", ".xdv", ".bin", ".pkl")
EXCLUDE_NAMES = {".DS_Store"}
DRIVE_FILES = ("train.txt", "val.txt", "test.txt", "train.bin", "val.bin", "meta.json", "manifest.json")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".tex", ".ini", ".cfg", ".toml", ".yml", ".yaml", "", ".html"}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, ok: bool, what: str, detail: str = "") -> bool:
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failures.append(what + (f": {detail}" if detail else ""))
        return ok


# ---------------------------------------------------------------------------
# Archive members
# ---------------------------------------------------------------------------

def excluded(rel: Path) -> bool:
    return (any(part in EXCLUDE_DIRS for part in rel.parts) or rel.name in EXCLUDE_NAMES
            or rel.name.endswith(EXCLUDE_SUFFIXES) or ".checkpoint." in rel.name)


def archive_members(root: Path, items: Iterable[str] = ARCHIVE_ITEMS) -> list[Path]:
    """Files to archive, relative to root, sorted. Missing items are left out (the checks report them)."""
    out: set[Path] = set()
    for item in items:
        path = root / item
        if path.is_file():
            out.add(Path(item))
        elif path.is_dir():
            out |= {p.relative_to(root) for p in path.rglob("*") if p.is_file()
                    and not excluded(p.relative_to(root))}
    return sorted(p for p in out if not excluded(p) or str(p) in items)


def write_archive(root: Path, members: Sequence[Path], out: Path, regno: str = REGNO) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for rel in members:
            info = zipfile.ZipInfo(f"urdu_bpe_{regno}/{rel.as_posix()}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (root / rel).read_bytes())
    tmp.replace(out)
    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def check_deliverables(root: Path, rep: Report, skip_drive: bool) -> None:
    for name in ("normalize.py", "train_tokenizer.py", "encode_corpus.py", "DATA_CARD.md", "README.md",
                 "tokenizer/urdu_bpe_50k.json", "tokenizer/urdu_bpe_50k.merges.txt", "tokenizer/train_log.json",
                 "report/report.pdf"):
        rep.check((root / name).is_file(), f"deliverable {name}")
    bpe = root / "bpe.py"
    rep.check(bpe.is_file() and "class BPETokenizer" in bpe.read_text(encoding="utf-8"),
              "deliverable bpe.py with class BPETokenizer")
    for d in ("collect", "clean"):
        rep.check(any((root / d).rglob("*.py")), f"collection/cleaning scripts in {d}/")
    readme = root / "README.md"
    rep.check(readme.is_file() and "Rebuild the tokenizer from train.txt" in readme.read_text(encoding="utf-8"),
              "README has the section 'Rebuild the tokenizer from train.txt'")
    if not skip_drive:
        for name in DRIVE_FILES:
            rep.check((root / "data" / "final" / name).is_file(), f"drive file data/final/{name}")


def check_tokenizer(root: Path, rep: Report) -> None:
    path = root / "tokenizer" / "urdu_bpe_50k.json"
    if not path.is_file():
        return
    from bpe import BPETokenizer
    tok = BPETokenizer.load(path)
    rep.check(len(tok.merges) == EXPECTED_MERGES and tok.vocab_size == EXPECTED_VOCAB,
              f"tokenizer has exactly {EXPECTED_MERGES:,} merges",
              f"{len(tok.merges):,} merges, vocabulary {tok.vocab_size:,}")


def check_report(root: Path, rep: Report) -> None:
    pdf = root / "report" / "report.pdf"
    if not pdf.is_file():
        return
    count, how = pdf_pages(pdf)
    rep.check(count <= MAX_PAGES, f"report.pdf has at most {MAX_PAGES} pages", f"{count} pages ({how})")


def check_markers(root: Path, members: Sequence[Path], rep: Report) -> None:
    card = root / "DATA_CARD.md"
    if card.is_file():
        left = card.read_text(encoding="utf-8").count(REVIEW)
        rep.check(left == 0, "DATA_CARD.md has no REVIEW markers",
                  f"{left} left; edit docs/data_card_gaps.md and rerun clean/data_card.py" if left else "")
    found = []
    for rel in sorted(set(members) | {Path("report/report.tex")}):
        path = root / rel
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if PLACEHOLDER in line:
                found.append(f"{rel}:{lineno}")
    rep.check(not found, f"no {PLACEHOLDER} ...] placeholders left", ", ".join(found[:10]))


def run_pytest(root: Path, args: Sequence[str]) -> tuple[bool, str]:
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", *args], cwd=root, capture_output=True, text=True)
    last = (out.stdout.strip().splitlines() or ["(no output)"])[-1]
    return out.returncode == 0, last


def check_tests(root: Path, rep: Report) -> None:
    ok, last = run_pytest(root, ["tests/test_forbidden_imports.py"])
    rep.check(ok, "forbidden-imports test green", last)
    ok, last = run_pytest(root, [])
    rep.check(ok, "pytest -q green", last)


def drive_bundle(root: Path, rep: Report) -> list[tuple[str, int, str]]:
    """(name, bytes, sha256) of each drive file, checked against manifest.json and meta.json."""
    final = root / "data" / "final"
    files = [(name, (final / name).stat().st_size, sha256(final / name)) for name in DRIVE_FILES
             if (final / name).is_file()]
    digest = {name: h for name, _, h in files}
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8")) if "manifest.json" in digest else {}
    for split, entry in manifest.get("splits", {}).items():
        name = f"{split}.txt"
        if name in digest:
            rep.check(entry.get("sha256") == digest[name], f"{name} sha256 = manifest.json")
    meta = json.loads((final / "meta.json").read_text(encoding="utf-8")) if "meta.json" in digest else {}
    tok_path = root / "tokenizer" / "urdu_bpe_50k.json"
    if meta and tok_path.is_file():
        rep.check(meta.get("tokenizer_sha256") == sha256(tok_path),
                  "train.bin and val.bin were encoded with the delivered tokenizer (meta.json tokenizer_sha256)")
    for split, entry in meta.get("splits", {}).items():
        name = f"{split}.bin"
        if name in digest and entry.get("bin_sha256"):
            rep.check(entry["bin_sha256"] == digest[name], f"{name} sha256 = meta.json")
        if name in digest:
            rep.check(bool(entry.get("verified")), f"{name} verified by encode_corpus.py (decodes to its .txt)")
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regno", default=REGNO)
    p.add_argument("--out", type=Path, default=None, help="default submission/urdu_bpe_<regno>.zip")
    p.add_argument("--skip-drive", action="store_true", help="no data/final here; the drive bundle is NOT checked")
    p.add_argument("--skip-tests", action="store_true", help="do not run pytest (not a full pre-flight)")
    p.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root
    out = args.out or root / "submission" / f"urdu_bpe_{args.regno}.zip"
    rep = Report()
    members = archive_members(root)
    print("Pre-flight checks:")
    check_deliverables(root, rep, args.skip_drive)
    check_tokenizer(root, rep)
    check_report(root, rep)
    check_markers(root, members, rep)
    files = [] if args.skip_drive else drive_bundle(root, rep)
    if args.skip_tests:
        print("  [SKIP] pytest (--skip-tests): this is not a full pre-flight")
    else:
        check_tests(root, rep)
    if rep.failures:
        print(f"\n{len(rep.failures)} pre-flight check(s) FAILED; nothing was written:")
        for f in rep.failures:
            print(f"  - {f}")
        return 1

    write_archive(root, members, out, args.regno)
    size = out.stat().st_size
    print(f"\nwrote {out.relative_to(root) if out.is_relative_to(root) else out}: {len(members)} files, "
          f"{size / 1e6:.2f} MB, sha256 {sha256(out)}")
    if args.skip_drive:
        print("\nDrive bundle NOT checked (--skip-drive): run this on the machine with data/final/.")
        return 0
    sums = root / "data" / "final" / "SHA256SUMS"
    sums.write_text("".join(f"{h}  {name}\n" for name, _, h in files), encoding="utf-8")
    print("\nUpload to the shared drive (private to course staff):")
    width = max(len(name) for name, _, _ in files)
    for name, size, h in [*files, ("SHA256SUMS", sums.stat().st_size, sha256(sums))]:
        print(f"  data/final/{name.ljust(width)}  {size / 1e6:>10,.2f} MB  {h[:16]}…")
    print(f"  total {sum(s for _, s, _ in files) / 1e6:,.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
