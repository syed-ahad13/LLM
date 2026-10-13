"""Tests for make_submission.py and report/checks.py (Step 12), on a small fake repository tree."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pytest

import make_submission as ms
from bpe import BPETokenizer
from report import checks

FAKE_PDF = b"%PDF-1.5\n1 0 obj << /Type /Pages /Kids [2 0 R 3 0 R 4 0 R] /Count 3 >> endobj\n%%EOF\n"
LINK = "https://" + "drive.google.com/drive/folders/test-folder"  # split so this file is not a leak itself
README = ("# x\n## Rebuild the tokenizer from train.txt\n## Data access\n<!-- DRIVE LINK -->\n"
          "Link: in the submitted copy.\n<!-- /DRIVE LINK -->\n")


@pytest.fixture(autouse=True)
def no_pdfinfo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count pages from the fake PDF's page tree, never with a real pdfinfo."""
    real = shutil.which
    monkeypatch.setattr(checks.shutil, "which", lambda name: None if name == "pdfinfo" else real(name))


def tree(root: Path, merges: int = 5) -> Path:
    files = {
        "normalize.py": "def normalize(t):\n    return t\n", "bpe.py": "class BPETokenizer:\n    pass\n",
        "train_tokenizer.py": "", "encode_corpus.py": "", "common.py": "", "make_submission.py": "",
        "collect/download.py": "", "clean/split.py": "", "clean/__pycache__/split.cpython-312.pyc": "x",
        "analysis/merges_report.py": "", "analysis/samples/news.txt": "خبر", "analysis/samples/SOURCES.md": "",
        "tests/test_x.py": "", "requirements.txt": "", "requirements-analysis.txt": "", "requirements-dev.txt": "",
        "Makefile": "", "pytest.ini": "", "DATA_CARD.md": "# Data card\n",
        "README.md": README, "drive_link.local": LINK + "\n", "evidence/robots/a.txt": "",
        "docs/decisions.md": "", "report/report.tex": "x", "report/report.aux": "x", "logs/run.log": "x",
        "data/final/train.txt": "secret", "tokenizer/urdu_bpe_50k.merges.txt": "", "tokenizer/train_log.json": "{}",
        "tokenizer/urdu_bpe_50k.checkpoint.json": "{}", ".DS_Store": "",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "report" / "report.pdf").write_bytes(FAKE_PDF)
    tok = BPETokenizer()
    for a, b in [(97, 98), (99, 100), (257, 258), (101, 102), (103, 104)][:merges]:
        tok._add_merge(a, b)
    tok.save(root / "tokenizer" / "urdu_bpe_50k.json")
    return root


def drive(root: Path) -> None:
    final = root / "data" / "final"
    for name in ("train", "val", "test"):
        (final / f"{name}.txt").write_text(f"{name} text<|endoftext|>", encoding="utf-8")
    for name in ("train", "val"):
        (final / f"{name}.bin").write_bytes(b"\x01\x00\x00\x01")
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in final.iterdir()}
    (final / "manifest.json").write_text(json.dumps({"splits": {s: {"sha256": sha[f"{s}.txt"]}
                                                                for s in ("train", "val", "test")}}))
    tok_sha = hashlib.sha256((root / "tokenizer" / "urdu_bpe_50k.json").read_bytes()).hexdigest()
    (final / "meta.json").write_text(json.dumps({"tokenizer_sha256": tok_sha,
                                                 "splits": {s: {"bin_sha256": sha[f"{s}.bin"], "verified": True}
                                                            for s in ("train", "val")}}))


def run(root: Path, *extra: str, monkeypatch: pytest.MonkeyPatch, merges: int = 5) -> int:
    monkeypatch.setattr(ms, "EXPECTED_MERGES", merges)
    monkeypatch.setattr(ms, "EXPECTED_VOCAB", 257 + merges)
    return ms.main(["--root", str(root), "--skip-tests", *extra])


def test_passing_tree_writes_archive_and_sums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    root = tree(tmp_path)
    drive(root)
    assert run(root, monkeypatch=monkeypatch) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "Upload to the shared drive" in out
    names = zipfile.ZipFile(root / "submission" / "urdu_bpe_456094.zip").namelist()
    rel = {n.split("/", 1)[1] for n in names}
    assert {"bpe.py", "analysis/samples/news.txt", "report/report.pdf", "tokenizer/urdu_bpe_50k.json",
            "evidence/robots/a.txt", "docs/decisions.md"} <= rel
    assert not any(r.startswith(("data/", "logs/")) or "__pycache__" in r or r.endswith((".aux", ".pyc"))
                   or ".checkpoint." in r or r.endswith(".DS_Store") for r in rel)
    assert all(n.startswith("urdu_bpe_456094/") for n in names)
    sums = (root / "data" / "final" / "SHA256SUMS").read_text().splitlines()
    assert len(sums) == len(ms.DRIVE_FILES) and sums[0].endswith("  train.txt")


def test_archive_is_reproducible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tree(tmp_path)
    assert run(root, "--skip-drive", monkeypatch=monkeypatch) == 0
    first = (root / "submission" / "urdu_bpe_456094.zip").read_bytes()
    assert run(root, "--skip-drive", monkeypatch=monkeypatch) == 0
    assert (root / "submission" / "urdu_bpe_456094.zip").read_bytes() == first


@pytest.mark.parametrize("breakage, message", [
    (lambda r: (r / "DATA_CARD.md").write_text("- <!-- REVIEW --> draft\n"), "REVIEW markers"),
    (lambda r: (r / "README.md").write_text(README + "link: [" + "PASTE link]\n"), "placeholders"),
    (lambda r: (r / "drive_link.local").unlink(), "shared-drive link set"),
    (lambda r: (r / "docs" / "decisions.md").write_text("folder: " + LINK), "no Google Drive URL"),
    (lambda r: (r / "README.md").write_text("## Rebuild the tokenizer from train.txt\n"), "DRIVE LINK markers"),
    (lambda r: (r / "report" / "report.tex").write_text("[" + "PASTE course]"), "placeholders"),
    (lambda r: (r / "report" / "report.pdf").write_bytes(FAKE_PDF.replace(b"/Count 3", b"/Count 5")), "4 pages"),
    (lambda r: (r / "encode_corpus.py").unlink(), "encode_corpus.py"),
    (lambda r: (r / "bpe.py").write_text("x = 1\n"), "BPETokenizer"),
    (lambda r: (r / "README.md").write_text("# no section\n"), "Rebuild the tokenizer"),
])
def test_failed_checks_stop_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str], breakage, message: str) -> None:
    root = tree(tmp_path)
    breakage(root)
    assert run(root, "--skip-drive", monkeypatch=monkeypatch) == 1
    out = capsys.readouterr().out
    assert "FAILED; nothing was written" in out and message in out
    assert not (root / "submission").exists()


def test_wrong_merge_count_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    root = tree(tmp_path, merges=4)
    assert run(root, "--skip-drive", monkeypatch=monkeypatch, merges=5) == 1
    assert "exactly 5 merges" in capsys.readouterr().out


def test_drive_files_are_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    root = tree(tmp_path)
    assert run(root, monkeypatch=monkeypatch) == 1  # no data/final files
    assert "drive file data/final/val.txt" in capsys.readouterr().out
    drive(root)
    (root / "data" / "final" / "val.bin").write_bytes(b"\x02\x00")  # changed after encoding
    assert run(root, monkeypatch=monkeypatch) == 1
    assert "val.bin sha256 = meta.json" in capsys.readouterr().out


def test_pdf_pages_fallbacks(tmp_path: Path) -> None:
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(FAKE_PDF)
    assert checks.pdf_pages(pdf) == (3, "page tree")
    pdf.write_bytes(b"%PDF compressed, no readable page tree")
    (tmp_path / "r.log").write_text("Output written on r.pdf (4 pages, 123 bytes).")
    assert checks.pdf_pages(pdf) == (4, "LaTeX log")
    assert checks.pages(pdf, 4) == 0 and checks.pages(pdf, 3) == 1


def test_link_goes_only_into_the_archived_readme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tree(tmp_path)
    assert run(root, "--skip-drive", monkeypatch=monkeypatch) == 0
    with zipfile.ZipFile(root / "submission" / "urdu_bpe_456094.zip") as z:
        readme = z.read("urdu_bpe_456094/README.md").decode("utf-8")
        assert not any(n.endswith("drive_link.local") for n in z.namelist())
    assert f"Link (private to course staff): <{LINK}>" in readme and "<!-- DRIVE LINK -->" not in readme
    assert LINK not in (root / "README.md").read_text(encoding="utf-8")
