"""Static check of every import in the project (CLAUDE.md, hard rules 1-3).

Each .py file is parsed with ast (never imported). The top-level module name is
collected from `import x`, `from x import y`, and importlib.import_module("x") /
__import__("x") calls with a literal string. Relative imports are local by
definition and skipped.

Rules:
- Every file outside analysis/ (repo-root *.py, collect/, clean/ and any other
  directory) may import only the standard library, IMPLEMENTATION_ALLOWED and
  our own local modules.
- analysis/ may also import matplotlib.
- tiktoken and tokenizers are allowed only in analysis/compare_tokenizers.py.
- pytest is allowed only in tests/.
- bpe.py may import only the standard library and regex (Step 7 brief): no
  numpy, and none of our own modules.
- Banned everywhere: BANNED, plus any third-party module with "bpe" in its
  name (our own bpe.py is a local module and fine).

Skipped: tests/, data/ and logs/ at the top level, plus hidden directories
(.venv, .git, caches) and __pycache__ at any depth. data/ and logs/ hold
downloads and output, not our code.
"""

import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

TOP_LEVEL_SKIP = frozenset({"tests", "data", "logs"})
STDLIB = frozenset(sys.stdlib_module_names)
IMPLEMENTATION_ALLOWED = frozenset({"regex", "numpy", "tqdm", "requests", "bs4", "lxml", "pyarrow"})
ANALYSIS_ONLY = frozenset({"matplotlib"})
COMPARISON_ONLY = frozenset({"tiktoken", "tokenizers"})
COMPARISON_FILE = Path("analysis/compare_tokenizers.py")
BPE_FILE = Path("bpe.py")
BPE_ALLOWED = frozenset({"regex"})
TESTS_ONLY = frozenset({"pytest"})
# Tokenizer/BPE libraries and Urdu NLP libraries. Third-party names containing
# "bpe" (fastBPE, bpemb, ...) are caught separately.
BANNED = frozenset({"sentencepiece", "minbpe", "transformers", "urduhack", "youtokentome", "subword_nmt"})
DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


class Violation(NamedTuple):
    path: Path
    line: int
    module: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: imports '{self.module}': {self.reason}"


def python_files(root: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        at_top = Path(dirpath) == root
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".") and d != "__pycache__" and not (at_top and d in TOP_LEVEL_SKIP)
        )
        files.extend(Path(dirpath) / f for f in sorted(filenames) if f.endswith(".py"))
    return files


def _is_dynamic_import(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in DYNAMIC_IMPORTERS
    if isinstance(func, ast.Attribute):
        return func.attr in DYNAMIC_IMPORTERS
    return False


def imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (line, top-level module name) for every absolute import in the tree."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.append((node.lineno, node.module.split(".")[0]))
        elif isinstance(node, ast.Call) and _is_dynamic_import(node.func) and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and not arg.value.startswith("."):
                found.append((node.lineno, arg.value.split(".")[0]))
    return sorted(found)


def local_modules(root: Path, path: Path) -> set[str]:
    """Names that resolve to our own code: root modules, top-level dirs, and siblings of `path`."""
    names = {p.stem for p in root.glob("*.py")}
    names |= {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    names |= {p.stem for p in path.parent.glob("*.py")}
    return names


def rule_for(module: str, rel: Path, local: set[str]) -> str | None:
    """Return why `module` may not be imported by the file at `rel`, or None if it may."""
    in_analysis = rel.parts[0] == "analysis"
    if module in BANNED:
        return "banned everywhere (tokenizer/BPE or Urdu NLP library)"
    if module in COMPARISON_ONLY:
        return None if rel == COMPARISON_FILE else f"allowed only in {COMPARISON_FILE.as_posix()}"
    if module in ANALYSIS_ONLY:
        return None if in_analysis else "allowed only in analysis/"
    if module in TESTS_ONLY:
        return "allowed only in tests/"
    if rel == BPE_FILE:
        return None if module in STDLIB or module in BPE_ALLOWED else "bpe.py may import only the stdlib and regex"
    if module in STDLIB or module in IMPLEMENTATION_ALLOWED or module in local:
        return None
    if "bpe" in module.lower():
        return "third-party BPE code is banned"
    allowed = ["stdlib", *sorted(IMPLEMENTATION_ALLOWED), *(sorted(ANALYSIS_ONLY) if in_analysis else [])]
    return f"not allowed here (allowed: {', '.join(allowed)}, local modules)"


def check_file(path: Path, root: Path) -> list[Violation]:
    rel = path.relative_to(root)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
    except SyntaxError as e:
        return [Violation(rel, e.lineno or 0, "?", f"could not parse: {e.msg}")]
    local = local_modules(root, path)
    return [
        Violation(rel, line, module, reason)
        for line, module in imported_modules(tree)
        if (reason := rule_for(module, rel, local)) is not None
    ]


def scan(root: Path) -> list[Violation]:
    return [v for path in python_files(root) for v in check_file(path, root)]


# ---------------------------------------------------------------------------
# The real check
# ---------------------------------------------------------------------------

def test_project_imports_are_allowed() -> None:
    violations = scan(REPO_ROOT)
    if violations:
        pytest.fail("Forbidden imports:\n" + "\n".join(f"  {v}" for v in violations), pytrace=False)


def test_scan_sees_project_files() -> None:
    seen = {p.relative_to(REPO_ROOT).as_posix() for p in python_files(REPO_ROOT)}
    assert {"common.py", "normalize.py", "bpe.py", "train_tokenizer.py", "encode_corpus.py"} <= seen
    assert not any(s.startswith(("tests/", ".venv/", "data/", "logs/")) for s in seen)


# ---------------------------------------------------------------------------
# The checker itself, on throwaway trees
# ---------------------------------------------------------------------------

def make_repo(root: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return root


def found(root: Path) -> list[tuple[str, int, str]]:
    return [(v.path.as_posix(), v.line, v.module) for v in scan(root)]


BASE = {"bpe.py": "", "common.py": "", "clean/extract.py": ""}


@pytest.mark.parametrize(
    "rel, src, expected",
    [
        ("normalize.py", "import os, json\nimport regex\nfrom collections import Counter\n", []),
        ("normalize.py", "import xml.etree.ElementTree\nimport numpy as np\nimport pyarrow.parquet\n", []),
        ("normalize.py", '"""doc."""\n\nimport tiktoken\n', [("normalize.py", 3, "tiktoken")]),
        ("bpe.py", "from tiktoken.core import Encoding\n", [("bpe.py", 1, "tiktoken")]),
        ("bpe.py", "import importlib\nimportlib.import_module('tokenizers.models')\n", [("bpe.py", 2, "tokenizers")]),
        ("bpe.py", "from importlib import import_module\nimport_module('tiktoken')\n", [("bpe.py", 2, "tiktoken")]),
        ("bpe.py", "__import__('sentencepiece')\n", [("bpe.py", 1, "sentencepiece")]),
        ("bpe.py", "import json, re\nimport regex\nfrom collections import Counter\n", []),
        ("bpe.py", "import numpy as np\nimport common\n", [("bpe.py", 1, "numpy"), ("bpe.py", 2, "common")]),
        ("clean/bpe.py", "import numpy\n", []),
        ("train_tokenizer.py", "import bpe\nimport common\n", []),
        ("train_tokenizer.py", "import fastbpe\n", [("train_tokenizer.py", 1, "fastbpe")]),
        ("train_tokenizer.py", "from transformers import AutoTokenizer\n", [("train_tokenizer.py", 1, "transformers")]),
        ("clean/pipeline.py", "import common\nimport extract\nfrom bs4 import BeautifulSoup\nimport lxml.html\n", []),
        ("clean/pipeline.py", "from . import extract\n", []),
        ("clean/plot.py", "import matplotlib.pyplot as plt\n", [("clean/plot.py", 1, "matplotlib")]),
        ("clean/audit.py", "import urduhack\n", [("clean/audit.py", 1, "urduhack")]),
        ("collect/scrape.py", "import requests\nimport pandas\n", [("collect/scrape.py", 2, "pandas")]),
        ("collect/scrape.py", "import pytest\n", [("collect/scrape.py", 1, "pytest")]),
        ("report/build.py", "import jinja2\n", [("report/build.py", 1, "jinja2")]),
        ("analysis/plots.py", "import matplotlib.pyplot as plt\nimport common\n", []),
        ("analysis/plots.py", "import tiktoken\n", [("analysis/plots.py", 1, "tiktoken")]),
        ("analysis/compare_tokenizers.py", "import tiktoken\nfrom tokenizers import Tokenizer\nimport matplotlib\n", []),
        ("analysis/compare_tokenizers.py", "import sentencepiece\nimport bpemb\n",
         [("analysis/compare_tokenizers.py", 1, "sentencepiece"), ("analysis/compare_tokenizers.py", 2, "bpemb")]),
    ],
)
def test_rules(tmp_path: Path, rel: str, src: str, expected: list[tuple[str, int, str]]) -> None:
    root = make_repo(tmp_path, BASE | {rel: src})
    assert found(root) == expected


def test_local_file_cannot_shadow_a_banned_module(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"tiktoken.py": "", "normalize.py": "import tiktoken\n"})
    assert found(root) == [("normalize.py", 1, "tiktoken")]


def test_skipped_directories(tmp_path: Path) -> None:
    bad = "import tiktoken\n"
    root = make_repo(tmp_path, {
        ".venv/lib/site.py": bad,
        "tests/test_x.py": bad,
        "data/raw/dataset/loader.py": bad,
        "logs/x.py": bad,
        "clean/__pycache__/x.py": bad,
        "clean/tests/x.py": bad,  # only the top-level tests/ is skipped
    })
    assert found(root) == [("clean/tests/x.py", 1, "tiktoken")]


def test_unparseable_file_is_reported(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"normalize.py": "def broken(:\n"})
    [v] = scan(root)
    assert v.path == Path("normalize.py") and v.line == 1 and "could not parse" in v.reason


def test_third_party_bpe_is_named_as_such(tmp_path: Path) -> None:
    # The allowlist would reject it anyway; the reason should still say why it matters.
    root = make_repo(tmp_path, {"analysis/plots.py": "import fastBPE\n"})
    [v] = scan(root)
    assert v.reason == "third-party BPE code is banned"


def test_message_names_file_line_and_module(tmp_path: Path) -> None:
    root = make_repo(tmp_path, {"normalize.py": "import os\nimport tiktoken\n"})
    [v] = scan(root)
    assert str(v) == "normalize.py:2: imports 'tiktoken': allowed only in analysis/compare_tokenizers.py"
