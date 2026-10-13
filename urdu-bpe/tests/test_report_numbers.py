"""Tests for report/make_numbers.py (Step 12): numbers.tex and the report's tables from the JSON stats.

The tokenizer is the committed tokenizer/urdu_bpe_50k.json, so the Q1 counts are the real ones; every
other input is a small synthetic JSON with the keys the real files have.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from common import REPO_ROOT
from report import make_numbers

TOKENIZER = REPO_ROOT / "tokenizer" / "urdu_bpe_50k.json"
SOURCES = ("own_scrape", "makhzan", "wiki_ur", "urdu_news_1m", "ufal_umc", "fineweb2_urd")


def stats() -> dict[str, Any]:
    cats = {"zwnj": {"count": 47, "matches": {"U+200C ZERO WIDTH NON-JOINER": 47}},
            "arabic_yeh": {"count": 25, "matches": {"U+064A ARABIC LETTER YEH": 25}}}
    cats_norm = {"zwnj": {"count": 43, "matches": {"U+200C ZERO WIDTH NON-JOINER": 43}},
                 "arabic_yeh": {"count": 0, "matches": {}}}
    neighbours = {"ت + ZWNJ + ح": 19, "ت + ZWNJ + گ": 12, "ل + ZWNJ + د": 2, "<U+200C> + ZWNJ + گ": 1}
    shares = {"LETTER": 0.7, "WORD": 0.0}
    tpw = {"ours": 1.19, "gpt2": 5.99, "o200k_base": 1.63, "xlm-roberta-base": 1.36}
    return {
        "train_log": {"phases": {"count_words": 13.17, "init": 3.36, "merges": 59.87, "save": 0.1},
                      "total_seconds": 76.51, "merges_per_s_overall": 830.8, "peak_rss_mb": {"trainer": 1439.4},
                      "distinct_words": 1425662, "word_occurrences": 170041947, "workers": 7, "merges": 49743,
                      "vocab_size": 50000, "train_txt": {"bytes": 1273551819},
                      "machine": {"cpu_model": "Apple M2", "physical_cores": 8, "ram_bytes": 8 * 2**30,
                                  "os_version": "macOS 14.4.1", "python_version": "3.12.15"},
                      "checks": {"passed": True, "merge_ids": [257, 49999], "round_trip": {"docs": 1000},
                                 "tokens_repeating_earlier_bytes": 0,
                                 "metrics": {"tokens_per_word": 1.1783, "bytes_per_token": 7.1111,
                                             "single_token_word_share": 0.8826}}},
        "reference_log": {"phases": {"train_reference": 151.14}, "trained_on": {"bytes": 986230}},
        "equivalence": {"merges_compared": 3000, "identical_merges": True, "identical_counts": True,
                        "fast_seconds": 0.325, "speedup": 465.0, "distinct_words": 14996, "word_occurrences": 131789},
        "corpus_totals": {"total": {"mb": 1299.86, "mib": 1239.64, "docs": 961176, "words": 153465131,
                                    "unique_words": 1232737},
                          "splits": {"train": {"mb": 1273.55, "docs": 941950}, "val": {"mb": 13.43, "docs": 9613},
                                     "test": {"mb": 12.87, "docs": 9613}}},
        "dedup": {s: {"bytes_out": b} for s, b in zip(SOURCES, (76e6, 50.75e6, 513.78e6, 1958e6, 702.2e6, 5893.9e6))},
        "normalize": {s: {"bytes_in": 100, "bytes_out": 99} for s in SOURCES},
        "char_audit": {"sources": {"makhzan": {"docs": 10, "chars": 1_000_000, "categories": cats,
                                               "zwnj_neighbours": neighbours}}},
        "char_audit_normalized": {"sources": {"makhzan": {"docs": 10, "chars": 990_000, "categories": cats_norm}}},
        "manifest": {
            "seed": 1337,
            "sources": {s: {"docs": 0 if s == "urdu_news_1m" else 10, "bytes": b, "text_bytes": b}
                        for s, b in zip(SOURCES, (76.2e6, 50.8e6, 517.7e6, 0, 70.2e6, 585e6))},
            "totals": {"bytes": 1299.9e6},
            "splits": {"train": {"genres": {"web": 5, "encyclopedia": 4, "news": 1}},
                       "val": {"genres": {"mixed": 1, "opinion/blogs": 0}}},
            "budget": {"policy": {"caps": {"fineweb2_urd": 0.45}},
                       "sources": {s: {"rate": r} for s, r in zip(SOURCES, (1, 1, 1, 0, 0.091, 0.0986))}}},
        "meta": {"train_tokens": 179309360, "val_tokens": 1888474,
                 "splits": {"train": {"bytes_per_token": 7.1, "tokens_per_word": 1.193}}},
        "analysis": {
            "q1_merges": {"first_word": {"rank": 27, "pretoken_count": 6081735},
                          "buckets": [{"ranks": [1, 10], "shares": shares},
                                      {"ranks": [10001, 49743], "shares": {"LETTER": 0.0, "WORD": 0.71}}],
                          "templates": {"asteroid": {"docs": {"wiki_ur": 26065}},
                                        "station": {"docs": {"wiki_ur": 26182, "fineweb2_urd": 3}}},
                          "wiki_train_docs": 296444,
                          "longest": [{"top_source": "wiki_ur", "top_share": 0.99},
                                      {"top_source": "fineweb2_urd", "top_share": 0.83}]},
            "q2_compare": {"tokenizer_order": list(tpw),
                           "labels": {"ours": "Ours (50k)", "gpt2": "GPT-2", "o200k_base": "o200k_base",
                                      "xlm-roberta-base": "xlm-roberta-base"},
                           "inputs": {"test": {"words": 1520007, "tokenizers": {
                               k: {"tokens_per_word": v, "bytes_per_token": 7.04 / v, "share_words_3plus": 0.03 * v}
                               for k, v in tpw.items()}},
                               "news": {"words": 173, "tokenizers": {k: {"tokens_per_word": v} for k, v in tpw.items()}},
                               "roman_urdu": {"words": 158, "tokenizers": {
                                   k: {"tokens_per_word": v} for k, v in {**tpw, "ours": 2.47}.items()}}},
                           "normalization": {"words": 26, "tokens": {"ours": {"raw": 73, "normalized": 27}}}},
            "q3_failures": {"cases": [{"label": lab, "ranking": {"ours": 2.47, "o200k": 1.56, "ratio": 1.58,
                                                                 "share_3plus": 0.424}}
                                      for lab in ("Roman Urdu", "Diacritics", "Numbers, dates")],
                            "training": {"latin": {"share": 0.0106}, "diacritic": {"share": 0.0064}}}},
    }


def run(tmp_path: Path, data: dict[str, Any], drop: str | None = None) -> tuple[int, Path]:
    argv = ["--out", str(tmp_path / "numbers.tex"), "--tables-dir", str(tmp_path / "tables"),
            "--tokenizer", str(TOKENIZER)]
    for name, obj in data.items():
        path = tmp_path / f"{name}.json"
        if name != drop:
            path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        argv += [f"--{name.replace('_', '-')}", str(path)]
    return make_numbers.main(argv), tmp_path / "numbers.tex"


def macros(path: Path) -> dict[str, str]:
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$", path.read_text(encoding="utf-8"), re.M))


def test_numbers_and_tables(tmp_path: Path) -> None:
    code, path = run(tmp_path, stats())
    assert code == 0
    m = macros(path)
    assert m["CorpusMB"] == "1,299.86" and m["CorpusDocs"] == "961,176" and m["Seed"] == "1337"
    assert m["GenresCount"] == "4" and m["FineWebCap"] == r"45\%" and m["SourcesAll"] == "6"
    assert m["RunTotal"] == "76.5" and m["RunMergesDone"] == "49,743" and m["RunPeakGB"] == "1.44"
    assert m["RefSeconds"] == "151.1" and m["FastSeconds"] == "0.33" and m["Speedup"] == "465"
    assert m["EqVerdict"] == "identical" and m["MachineCPU"] == "Apple M2"
    # Q1 straight from the committed tokenizer: 21 letters, 2 punctuation marks, 4 invalid tokens
    assert (m["QOneLetters"], m["QOneChars"], m["QOneInvalid"]) == ("21", "2", "4")
    assert "at 30" in m["QOneCharList"] and "at 47" in m["QOneCharList"]
    assert m["QOneFirstWordRank"] == "27" and m["QOneStationRank"] == "363"
    assert m["QOneLongestMax"] == "29" and m["QOneLongestMin"] == "25" and m["QOneLongestWiki"] == "1"
    assert m["QOneStubDocs"] == "52,247" and m["QOneStubShare"] == r"17.6\%"
    assert m["QTwoOurs"] == "1.19" and m["QTwoGpt"] == "5.99" and m["QTwoHFName"] == "xlm-roberta-base"
    assert m["QTwoWins"] == "2" and m["QTwoInputs"] == "3"
    assert m["QThreeNormRaw"] == "73" and m["QThreeLatinShare"] == r"1.06\%"
    assert (m["ZwnjRaw"], m["ZwnjKept"], m["ZwnjRemoved"]) == ("47", "43", "4")
    assert m["ZwnjMakhzanTeh"] == "31" and m["ZwnjMakhzanBetween"] == "33"
    assert m["GptCountHandout"] == "5" and m["NumberPieces"] == "2,220"
    for name in ("r_corpus", "r_normalization", "r_training", "r_merges", "r_longest"):
        text = (tmp_path / "tables" / f"{name}.tex").read_text(encoding="utf-8")
        assert text.startswith("% Generated by report/make_numbers.py") and r"\bottomrule" in text
    merges = (tmp_path / "tables" / "r_merges.tex").read_text(encoding="utf-8")
    assert "49,743" in merges and "45,000" in merges and r"\ur{ا}" in merges
    corpus = (tmp_path / "tables" / "r_corpus.tex").read_text(encoding="utf-8")
    assert "Urdu News 1M" in corpus and "whole" in corpus and r"9.1\%" in corpus


def test_every_report_macro_is_defined(tmp_path: Path) -> None:
    run(tmp_path, stats())
    defined = set(macros(tmp_path / "numbers.tex"))
    tex = (REPO_ROOT / "report" / "report.tex").read_text(encoding="utf-8")
    builtin = {"ur", "textbf", "texttt", "emph", "ref", "label", "caption", "input", "centering", "small",
               "footnotesize", "section", "verb", "maketitle", "textvisiblespace", "textbar", "textbackslash",
               "times", "to", "dots", "quad", "vspace", "title", "author", "date", "documentclass", "usepackage",
               "babelprovide", "babelfont", "hypersetup", "newcommand", "foreignlanguage", "tracinglostchars",
               "pdfvariable", "setlength", "parskip", "parindent", "textfloatsep", "floatsep", "abovecaptionskip",
               "renewcommand", "arraystretch", "begin", "end", "_", "%"}
    used = set(re.findall(r"\\([A-Za-z]+)", re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", tex, flags=re.S)))
    undefined = sorted(u for u in used - builtin - defined if u[0].isupper())
    assert undefined == [], f"report.tex uses macros make_numbers.py does not define: {undefined}"


def test_missing_input_is_named(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="analysis"):
        run(tmp_path, stats(), drop="analysis")


def test_missing_key_is_named(tmp_path: Path) -> None:
    data = stats()
    del data["meta"]["train_tokens"]
    with pytest.raises(SystemExit, match="train_tokens"):
        run(tmp_path, data)
