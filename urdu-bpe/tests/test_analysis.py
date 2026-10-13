"""Tests for the Part C scripts (Step 11): analysis/report_utils.py, merges_report.py, compare_tokenizers.py and
failure_cases.py.

Unit tests run on small made-up inputs. The last part checks the real outputs of `make analysis` when they
exist: every generated draft in docs/analysis_notes.md is exactly what the scripts produce from
data/stats/analysis.json now, so the notes' numbers match the JSON, and every table and figure is there.
"""

from pathlib import Path

import pytest

from analysis import compare_tokenizers, failure_cases, merges_report, report_utils
from analysis.report_utils import generated_block, latex_display, show, update_notes

# ---------------------------------------------------------------------------
# merges_report: categories, buckets, tracing
# ---------------------------------------------------------------------------

COUNTS = {" کے": 6_000_000, "کے": 1_000, " ک": 2_250, "یں": 1_063, " نادر": 99, "abc": 500}


@pytest.mark.parametrize("token, expected", [
    ("ا".encode(), "LETTER"),
    (b" \xd8", "SPACE+LEAD"),
    (b" \xe2\x80", "SPACE+LEAD"),  # 3-byte character, 2 of its bytes
    (b"\xa7 ", "INVALID_START"),
    (b"\xa7\xd8", "INVALID_START"),  # cut at both ends: starts mid-character first
    ("ا".encode() + b"\xd8", "INVALID_END"),
    (" ا".encode() + b"\xd8", "INVALID_END"),
    (" ک".encode(), "SPACE+LETTER"),
    (" کے".encode(), "WORD"),
    ("کے".encode(), "WORD"),  # without its space, counted through " کے"
    (" نادر".encode(), "OTHER"),  # a word shape, but under 100 occurrences
    ("یں".encode(), "WORD"),
    (b" 123", "OTHER"),
    ("،".encode(), "LETTER"),  # any single 2-byte character
    ("۔۔".encode(), "OTHER"),
])
def test_category(token: bytes, expected: str) -> None:
    assert merges_report.category(token, COUNTS) == expected


def test_literal_word_has_no_two_letter_minimum() -> None:
    assert merges_report.literal_word(" ک".encode(), COUNTS)
    assert not merges_report.literal_word("ا".encode(), COUNTS)  # LETTER first
    assert not merges_report.literal_word(b" \xd8", COUNTS)


def test_buckets() -> None:
    assert merges_report.buckets(49_743) == [(1, 10), (11, 100), (101, 1000), (1001, 10_000), (10_001, 49_743)]
    assert merges_report.buckets(150) == [(1, 10), (11, 100), (101, 150)]
    assert merges_report.buckets(7) == [(1, 7)]


def test_trace_counts_documents_per_source(tmp_path: Path) -> None:
    eot = "<|endoftext|>"
    docs = [("wiki_ur:1", "ایک اسٹیشن ہے"), ("fineweb2_urd:a", "اسٹیشن اسٹیشن"), ("wiki_ur:2", "کچھ نہیں"),
            ("own_scrape:x", "ریلوے اسٹیشن")]
    train = tmp_path / "train.txt"
    train.write_text("".join(d + eot for _, d in docs), encoding="utf-8")
    ids = tmp_path / "train.ids"
    ids.write_text("\n".join(i for i, _ in docs) + "\n", encoding="utf-8")
    found = merges_report.trace(["اسٹیشن", "نہیں", "غائب"], train, ids)
    assert found == {"اسٹیشن": {"wiki_ur": 1, "fineweb2_urd": 1, "own_scrape": 1}, "نہیں": {"wiki_ur": 1}, "غائب": {}}


# ---------------------------------------------------------------------------
# report_utils: display, LaTeX, notes
# ---------------------------------------------------------------------------

def test_display_and_latex() -> None:
    assert show(" کے".encode()) == "␣کے"
    assert show(b" \xd8") == "␣\\xd8"
    assert latex_display("␣کے") == r"\textvisiblespace{}\ur{کے}"
    assert latex_display("␣\\xd8") == r"\textvisiblespace{}\texttt{\textbackslash{}xd8}"
    assert latex_display("a | b_1 50%") == r"a \textbar{} b\_1 50\%"
    assert latex_display("دُعا") == r"\ur{دُعا}"  # the mark stays inside the Urdu run


def test_update_notes_keeps_text_outside_the_markers(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    update_notes("q1", "Q1. Title", "first draft", notes)
    update_notes("q2", "Q2. Title", "second", notes)
    text = notes.read_text(encoding="utf-8")
    assert text.startswith("# Part C analysis notes") and text.index("## Q1. Title") < text.index("## Q2. Title")
    notes.write_text(text.replace("## Q2. Title", "My own Q1 version.\n\n## Q2. Title"), encoding="utf-8")
    update_notes("q1", "Q1. Title", "new draft, $1.5 \\1", notes)
    text = notes.read_text(encoding="utf-8")
    assert generated_block(text, "q1") == "new draft, $1.5 \\1" and "My own Q1 version." in text
    assert generated_block(text, "q2") == "second" and text.count("## Q1. Title") == 1


def test_excerpt() -> None:
    assert report_utils.excerpt("\n  one two three\nnext line", max_words=2) == "one two"


# ---------------------------------------------------------------------------
# compare_tokenizers and failure_cases on made-up counts
# ---------------------------------------------------------------------------

def test_measure() -> None:
    encoders = {"bytes": lambda t: list(t.encode("utf-8")), "words": lambda t: t.split()}
    metrics, per_word = compare_tokenizers.measure(["ab ab", "abcd"], encoders)
    assert metrics["words"] == 3 and metrics["bytes"] == 9 and metrics["chars"] == 9
    b = metrics["tokenizers"]["bytes"]
    assert b["tokens"] == 9 and b["tokens_per_word"] == 3.0 and b["bytes_per_token"] == 1.0
    assert per_word == {"ab": [2, 3, 1], "abcd": [1, 5, 1]}  # " ab" is 3 bytes, " abcd" 5
    assert b["share_words_3plus"] == 1.0 and metrics["tokenizers"]["words"]["share_words_3plus"] == 0.0


def q2_fixture() -> dict:
    def t(ours: float, o200k: float, gpt2: float, xlmr: float) -> dict:
        return {n: {"tokens_per_word": v, "share_words_3plus": 0.1} for n, v in
                (("ours", ours), ("o200k_base", o200k), ("gpt2", gpt2), ("xlm-roberta-base", xlmr))}
    return {"inputs": {"test": {"tokenizers": t(1.2, 1.6, 6.0, 1.4)},
                       "roman_urdu": {"tokenizers": t(2.5, 1.5, 2.0, 1.5)},
                       "diacritics": {"tokenizers": t(2.7, 3.5, 7.5, 3.8)},
                       "news": {"tokenizers": t(1.1, 1.7, 6.4, 1.3)}}}


def test_rank_inputs() -> None:
    ranked = failure_cases.rank_inputs(q2_fixture())
    # roman_urdu and diacritics tie on rank sum (3); the tie goes to more tokens per word
    assert [r["input"] for r in ranked] == ["diacritics", "roman_urdu", "test", "news"]
    roman = ranked[1]
    assert (roman["rank_tpw"], roman["rank_ratio"], roman["ratio"]) == (2, 1, round(2.5 / 1.5, 4))
    assert roman["best_other"] in ("o200k_base", "xlm-roberta-base") and roman["best_other_tpw"] == 1.5


def test_training_shares() -> None:
    shares = failure_cases.training_shares({" abc": 2, " کِتاب": 1, " کتاب": 5, " 12": 2, "".join(map(chr, (0x643,))): 0})
    assert shares["pretokens"] == 10
    assert shares["latin"]["occurrences"] == 2 and shares["diacritic"]["occurrences"] == 1
    assert shares["digit"]["share"] == 0.2 and shares["arabic_kaf_yeh_heh"]["occurrences"] == 0


# ---------------------------------------------------------------------------
# The real outputs of `make analysis`
# ---------------------------------------------------------------------------

analysis = report_utils.read_analysis()
real = pytest.mark.skipif(not ({"q1_merges", "q2_compare", "q3_failures"} <= set(analysis)
                               and report_utils.NOTES.exists()), reason="run `make analysis` first")


@real
@pytest.mark.parametrize("module, section", [(merges_report, "q1"), (compare_tokenizers, "q2"),
                                             (failure_cases, "q3")])
def test_notes_match_analysis_json(module, section: str) -> None:
    notes = report_utils.NOTES.read_text(encoding="utf-8")
    assert generated_block(notes, section) == module.draft(analysis).strip()


@real
def test_every_output_exists() -> None:
    notes = report_utils.NOTES.read_text(encoding="utf-8")
    assert generated_block(notes, "q4_q5") == failure_cases.Q45_NOTE
    for name in ("merges", "merges_longest", "compare", "compare_metrics", "failure_inputs", "failure_words",
                 "failure_cases", "normalization"):
        text = (report_utils.TABLES_DIR / f"{name}.tex").read_text(encoding="utf-8")
        assert text.startswith("% Generated by analysis/") and r"\bottomrule" in text, name
    for name in ("merge_categories", "tokens_per_word"):
        assert (report_utils.FIGURES_DIR / f"{name}.pdf").read_bytes().startswith(b"%PDF"), name
    q1 = analysis["q1_merges"]
    assert q1["merges"] == 49_743 and sum(q1["category_totals"].values()) == 49_743
    assert {"ours", "gpt2", "o200k_base"} <= set(analysis["q2_compare"]["tokenizer_order"])
    assert len(analysis["q3_failures"]["cases"]) == 3
