"""Tests for clean/audit_chars.py (Step 4 evidence: which code points normalization must handle)."""

import json
from pathlib import Path

import pytest

from clean import audit_chars
from clean.audit_chars import CATEGORIES, Audit, audit_docs, visible
from common import write_jsonl


def counts(text: str) -> dict[str, int]:
    audit = Audit("s", contexts=0)
    audit.add(text)
    return {name: n for name, n in audit.counts.items() if n}


@pytest.mark.parametrize("text, expected", [
    ("\u064A\u0649", {"arabic_yeh": 2}),
    ("\u0643\u06AA", {"arabic_kaf": 2}),
    ("\u0647", {"arabic_heh": 1}),
    ("\u0629\u06C0\u06C3\u06D5", {"heh_like_other": 4}),
    ("\u0640", {"tatweel": 1}),
    ("ب\u200Cب", {"zwnj": 1}),
    ("\u200D", {"zwj": 1}),
    ("\u200E\u200F\uFEFF\u00AD\u200B\u202B\u2066", {"invisible_other": 7}),
    ("\x01\x7f", {"control": 2}),
    ("\uFFFD", {"replacement_char": 1}),
    ("12", {"digits_ascii": 2}),
    ("\u0661", {"digits_arabic_indic": 1}),
    ("\u06F1\u06F2", {"digits_extended": 2}),
    ("a\u00A0b\u202Fc", {"space_unusual": 2}),
    ("a\tb", {"tab": 1}),
    ("a\r\nb", {"carriage_return": 1}),
    ("a\u2028b", {"line_separator_other": 1}),
    ("a  b   c", {"space_run": 2}),
    (" a\nb ", {"space_at_line_edge": 2}),
    ("a\n\n\nb", {"blank_line_run": 1}),
    ("\uFE8E\uFDFA", {"presentation_form": 2}),
    ("ا\u0653", {"combining_madda_hamza": 1}),
    ("\u064E\u0651", {"harakat": 2}),
    ("ہے ۔ اور ،", {"space_before_urdu_punct": 2}),
    ("ہے. اور ,", {"latin_punct_after_urdu": 2}),
    ("a<|endoftext|>b", {"endoftext_literal": 1}),
    ("یہ عام اردو ہے۔", {}),
])
def test_each_category(text: str, expected: dict[str, int]) -> None:
    assert counts(text) == expected


def test_categories_compile_and_have_descriptions() -> None:
    for name, (pattern, desc) in CATEGORIES.items():
        assert pattern and desc, name


def test_visible_spells_out_invisible_characters() -> None:
    assert visible("ت\u200Cح") == "ت<U+200C>ح"
    assert visible("a\u00A0b\nc") == "a<U+00A0>b⏎c"
    assert visible("a b") == "a b"
    assert visible("\x07") == "<U+0007>"


def test_matches_contexts_and_doc_counts() -> None:
    docs = [{"text": "او\u0651ل \u064A " * 5}, {"text": "کوئی عربی حرف نہیں"}, {"text": "\u0649"}]
    audit = audit_docs("s", docs, contexts=2)
    report = audit.report()
    yeh = report["categories"]["arabic_yeh"]
    assert yeh["count"] == 6
    assert yeh["docs_with"] == 2
    assert yeh["matches"] == {"U+064A ARABIC LETTER YEH": 5, "U+0649 ARABIC LETTER ALEF MAKSURA": 1}
    assert len(yeh["contexts"]) == 2  # capped at --contexts
    assert "⟦\u064A⟧" in yeh["contexts"][0]
    assert report["docs"] == 3


def test_multi_char_matches_are_labelled_by_code_points() -> None:
    report = audit_docs("s", [{"text": "ا\u0653"}], contexts=1).report()
    assert report["categories"]["combining_madda_hamza"]["matches"] == {"U+0627 U+0653": 1}


def test_nfc_changed_docs() -> None:
    docs = [{"text": "ا\u0653"}, {"text": "آ"}, {"text": "abc"}]
    assert audit_docs("s", docs, contexts=0).nfc_changed_docs == 1


def test_zwnj_neighbours() -> None:
    report = audit_docs("s", [{"text": "صورت\u200Cحال \u200C"}], contexts=0).report()
    assert report["zwnj_neighbours"] == {"ت + ZWNJ + ح": 1, "  + ZWNJ + $": 1}


def test_others_skip_ascii_core_urdu_and_category_chars() -> None:
    text = "abc ہم نے کہا۔ \u064A\u06F1\u00A0 \u06AA\u06AA ’"
    report = audit_docs("s", [{"text": text}], contexts=0).report()
    assert report["top_other_chars"] == {"U+2019 RIGHT SINGLE QUOTATION MARK": 1}


def test_main_writes_json_for_selected_sources(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    samples = tmp_path / "samples"
    write_jsonl(samples / "a.sample.jsonl", [{"text": "\u064A"}, {"text": "\u0643"}, {"text": "\u0647"}])
    write_jsonl(samples / "b.sample.jsonl", [{"text": "ہ"}])
    out = tmp_path / "audit.json"
    assert audit_chars.main(["--samples-dir", str(samples), "--out", str(out), "--source", "a", "--limit", "2"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data["sources"]) == ["a"]
    a = data["sources"]["a"]
    assert a["docs"] == 2
    assert a["categories"]["arabic_yeh"]["count"] == 1
    assert a["categories"]["arabic_heh"]["count"] == 0  # third doc is past --limit
    assert "arabic_yeh" in capsys.readouterr().out


def test_main_reads_interim(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "x.jsonl.gz", [{"text": "\u0640"}])
    out = tmp_path / "audit.json"
    assert audit_chars.main(["--interim", "--interim-dir", str(tmp_path), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["input"] == "interim"
    assert data["sources"]["x"]["categories"]["tatweel"]["count"] == 1


def test_main_missing_source(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "a.sample.jsonl", [{"text": "x"}])
    with pytest.raises(SystemExit, match="nope"):
        audit_chars.main(["--samples-dir", str(tmp_path), "--out", str(tmp_path / "o.json"), "--source", "nope"])
