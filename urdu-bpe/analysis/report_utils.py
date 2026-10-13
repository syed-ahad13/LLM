"""Shared helpers for the Part C scripts (Step 11): paths, inputs, the token display convention, LaTeX tables,
figures, data/stats/analysis.json and the generated drafts in docs/analysis_notes.md.

Display convention (bpe.token_display): valid UTF-8 as text with every space as ␣, bytes that are not valid
UTF-8 on their own as \\xNN, invisible characters as \\n, \\uXXXX and so on; always with the byte length.

In LaTeX, runs of Arabic-script text are wrapped in \\ur{...}, which the report must define, e.g. with
polyglossia: \\newcommand{\\ur}[1]{\\texturdu{#1}}. ␣ becomes \\textvisiblespace and escapes are typeset in
\\texttt. Plots use English labels only (matplotlib cannot shape Urdu).
"""

import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bpe import ENDOFTEXT, token_display  # noqa: E402
from common import write_json  # noqa: E402
from normalize import normalize  # noqa: E402

TOKENIZER = ROOT / "tokenizer" / "urdu_bpe_50k.json"
TEST_TXT = ROOT / "data" / "final" / "test.txt"
TRAIN_TXT = ROOT / "data" / "final" / "train.txt"
SAMPLES_DIR = ROOT / "analysis" / "samples"
CACHE_DIR = ROOT / "data" / "cache"
FIGURES_DIR = ROOT / "report" / "figures"
TABLES_DIR = ROOT / "report" / "tables"
ANALYSIS_JSON = ROOT / "data" / "stats" / "analysis.json"
WORDS_JSON = ROOT / "data" / "stats" / "analysis_words.json"
NOTES = ROOT / "docs" / "analysis_notes.md"

# The seven hand-picked inputs (analysis/samples/SOURCES.md), in the handout's order, with plot labels.
SAMPLES = {"news": "News", "poetry": "Poetry", "urdu_english": "Urdu + English", "roman_urdu": "Roman Urdu",
           "numbers_dates": "Numbers, dates", "diacritics": "Diacritics", "names": "Names"}
INPUT_LABELS = {"test": "test.txt", **SAMPLES}
UNNORMALIZED = "unnormalized.txt"

# Categorical slots 1-8 of the dataviz reference palette (light mode), assigned in this order, never cycled.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def test_documents(path: Path = TEST_TXT) -> list[str]:
    """The documents of test.txt, separators removed, normalized (a no-op: the split is already normalized)."""
    docs = path.read_text(encoding="utf-8").split(ENDOFTEXT)
    return [normalize(d) for d in docs if d]


def sample_text(name: str, samples_dir: Path = SAMPLES_DIR) -> str:
    return normalize((samples_dir / f"{name}.txt").read_text(encoding="utf-8"))


def load_inputs(test_path: Path = TEST_TXT, samples_dir: Path = SAMPLES_DIR) -> dict[str, list[str]]:
    """{input: its documents}: test.txt and the seven samples, all normalized, in INPUT_LABELS order."""
    missing = [n for n in SAMPLES if not (samples_dir / f"{n}.txt").exists()]
    if missing:
        raise SystemExit(f"missing samples in {samples_dir}: {', '.join(m + '.txt' for m in missing)}")
    return {"test": test_documents(test_path), **{n: [sample_text(n, samples_dir)] for n in SAMPLES}}


def excerpt(text: str, max_words: int = 12) -> str:
    """The first line of a text, cut to at most max_words words: the passage shown with token boundaries."""
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    return " ".join(line.split()[:max_words])


# ---------------------------------------------------------------------------
# Display and LaTeX
# ---------------------------------------------------------------------------

def show(token: bytes) -> str:
    return token_display(token)


def boundaries(tokens: Iterable[bytes]) -> str:
    """Tokens joined with " | ", each in the display convention (fragments as \\xNN)."""
    return " | ".join(show(t) for t in tokens)


_LATEX_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
                  "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\~{}"}
_DISPLAY_PIECES = regex.compile(r"\\x[0-9a-f]{2}|\\u[0-9a-f]{4}|\\U[0-9a-f]{8}|\\[ntr\\]|␣|\|"
                                r"|[\p{Script=Arabic}\p{M}]+|.", regex.S)


def latex_escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(c, c) for c in text)


def latex_display(display: str) -> str:
    """A display string (show(), boundaries()) as LaTeX: Urdu runs in \\ur{}, ␣ visible, escapes in \\texttt."""
    out = []
    for piece in _DISPLAY_PIECES.findall(display):
        if piece == "␣":
            out.append(r"\textvisiblespace{}")
        elif piece == "|":
            out.append(r"\textbar{}")
        elif piece.startswith("\\") and len(piece) > 1:
            out.append(r"\texttt{\textbackslash{}" + latex_escape(piece[1:]) + "}")
        elif regex.match(r"[\p{Script=Arabic}\p{M}]", piece):
            out.append(r"\ur{" + piece + "}")
        else:
            out.append(latex_escape(piece))
    return "".join(out)


def latex_token(token: bytes) -> str:
    return latex_display(show(token))


def write_table(name: str, colspec: str, header: Sequence[str], rows: Iterable[Sequence[str]], script: str,
                tables_dir: Path = TABLES_DIR) -> Path:
    """report/tables/<name>.tex: a booktabs tabular (no float), cells already LaTeX; \\input it in a table."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"% Generated by {script} (make analysis); do not edit by hand.",
             f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = tables_dir / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def pyplot() -> Any:
    """matplotlib.pyplot with a non-interactive backend and the report's quiet axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42,
    })
    return plt


def save_figure(fig: Any, name: str, figures_dir: Path = FIGURES_DIR) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    return path


# ---------------------------------------------------------------------------
# analysis.json and the notes
# ---------------------------------------------------------------------------

def read_analysis(path: Path = ANALYSIS_JSON) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_analysis(section: str, data: dict[str, Any], path: Path = ANALYSIS_JSON) -> dict[str, Any]:
    """Replace one section of data/stats/analysis.json, keeping the others."""
    analysis = read_analysis(path)
    analysis[section] = data
    write_json(path, analysis)
    return analysis


NOTES_HEADER = """# Part C analysis notes

Drafts for the report, one per question, generated by `make analysis` from `data/stats/analysis.json`, so
their numbers match it. The text between each BEGIN/END pair is rewritten on every run: write your own
version outside the markers (it is kept), or copy the draft into `report/` and edit it there.

Display convention: ␣ is a space, \\xNN a byte that is not valid UTF-8 on its own; byte lengths are given
in the merge tables. Tables: `report/tables/*.tex`; figures: `report/figures/*.pdf`.
"""


def begin_marker(section: str) -> str:
    return f"<!-- BEGIN generated:{section} -->"


def end_marker(section: str) -> str:
    return f"<!-- END generated:{section} -->"


def generated_block(notes: str, section: str) -> str | None:
    """The text between a section's markers, or None."""
    m = re.search(re.escape(begin_marker(section)) + r"\n(.*?)\n" + re.escape(end_marker(section)), notes, re.S)
    return m.group(1) if m else None


def update_notes(section: str, title: str, draft: str, path: Path = NOTES) -> None:
    """Rewrite one generated draft in docs/analysis_notes.md; everything outside the markers is kept."""
    notes = path.read_text(encoding="utf-8") if path.exists() else NOTES_HEADER
    block = f"{begin_marker(section)}\n{draft.strip()}\n{end_marker(section)}"
    if generated_block(notes, section) is not None:
        pattern = re.escape(begin_marker(section)) + r".*?" + re.escape(end_marker(section))
        notes = re.sub(pattern, lambda _: block, notes, count=1, flags=re.S)
    else:
        notes = notes.rstrip("\n") + f"\n\n## {title}\n\n{block}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(notes, encoding="utf-8")


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"
