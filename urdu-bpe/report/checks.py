"""Build checks for the report: the TeX tools and fonts it needs, and its page count.

    python report/checks.py tools                       # lualatex, latexmk, pdfinfo, Noto Naskh Arabic
    python report/checks.py pages report/report.pdf --max 4

tools prints what is missing and how to install it on macOS (Homebrew) or
Debian/Ubuntu, and exits 1 if the build cannot run. pdfinfo (poppler) is
optional: without it the page count is read from the PDF's page tree
(report.tex writes uncompressed objects), else from the LaTeX log.
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

FONT = "Noto Naskh Arabic"
INSTALL = {
    "lualatex": ("brew install --cask mactex-no-gui   # or BasicTeX: brew install --cask basictex, then\n"
                 "    sudo tlmgr update --self && sudo tlmgr install latexmk babel fontspec luaotfload booktabs "
                 "geometry hyperref xcolor\n"
                 "    (Debian/Ubuntu: sudo apt install texlive-luatex texlive-latex-extra texlive-lang-arabic latexmk)"),
    "latexmk": "comes with MacTeX; with BasicTeX: sudo tlmgr install latexmk (Debian/Ubuntu: sudo apt install latexmk)",
    "pdfinfo": "brew install poppler   (Debian/Ubuntu: sudo apt install poppler-utils); optional",
    "font": (f"brew install --cask font-noto-naskh-arabic   (or download {FONT} from fonts.google.com/noto and\n"
             "    install it; Debian/Ubuntu: sudo apt install fonts-noto-core); then: luaotfload-tool --update"),
}


def find_font(name: str = FONT) -> str | None:
    """Path of the font file luaotfload resolves for name, or None."""
    tool = shutil.which("luaotfload-tool")
    if tool is None:
        return None
    out = subprocess.run([tool, f"--find={name}"], capture_output=True, text=True)
    match = re.search(r'Resolved file name "([^"]+)"', out.stdout + out.stderr)
    return match.group(1) if match else None


def tools() -> int:
    missing = []
    for tool in ("lualatex", "latexmk", "pdfinfo"):
        path = shutil.which(tool)
        print(f"{tool:10s} {path or 'MISSING'}")
        if path is None:
            missing.append(tool)
    font = find_font() if "lualatex" not in missing else None
    print(f"{'font':10s} {FONT}: {font or 'NOT FOUND (or lualatex missing)'}")
    if font is None:
        missing.append("font")
    for item in missing:
        print(f"\ninstall {item}:\n    {INSTALL[item]}")
    blocking = [m for m in missing if m != "pdfinfo"]
    return 1 if blocking else 0


def pdf_pages(pdf: Path, log: Path | None = None) -> tuple[int, str]:
    """(pages, how they were counted): pdfinfo, else the page tree's /Count, else the LaTeX log."""
    tool = shutil.which("pdfinfo")
    if tool:
        out = subprocess.run([tool, str(pdf)], capture_output=True, text=True, check=True).stdout
        match = re.search(r"^Pages:\s+(\d+)", out, re.M)
        if match:
            return int(match.group(1)), "pdfinfo"
    data = pdf.read_bytes()
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", data, re.S)]
    counts += [int(c) for c in re.findall(rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages\b", data, re.S)]
    if counts:
        return max(counts), "page tree"
    log = log or pdf.with_suffix(".log")
    if log.exists():
        match = re.search(r"Output written on .*?\((\d+) pages?", log.read_text(errors="replace"), re.S)
        if match:
            return int(match.group(1)), "LaTeX log"
    raise SystemExit(f"cannot count the pages of {pdf}: install pdfinfo ({INSTALL['pdfinfo']})")


def pages(pdf: Path, max_pages: int) -> int:
    if not pdf.exists():
        raise SystemExit(f"no {pdf}")
    count, how = pdf_pages(pdf)
    ok = count <= max_pages
    print(f"{pdf}: {count} pages ({how}); limit {max_pages}: {'OK' if ok else 'TOO LONG'}")
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tools")
    pg = sub.add_parser("pages")
    pg.add_argument("pdf", type=Path)
    pg.add_argument("--max", type=int, default=4)
    args = p.parse_args(argv)
    return tools() if args.cmd == "tools" else pages(args.pdf, args.max)


if __name__ == "__main__":
    sys.exit(main())
