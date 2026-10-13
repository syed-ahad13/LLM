"""Part C, question 3: three inputs our tokenizer handles badly, and why.

    python analysis/failure_cases.py                       # after analysis/compare_tokenizers.py
    python analysis/failure_cases.py --cases roman_urdu diacritics names

Reads the counts that analysis/compare_tokenizers.py wrote (data/stats/analysis.json, section q2_compare, and
data/stats/analysis_words.json): this script may not import tiktoken. Our own tokenizer is run here to show
token boundaries, and the training word counts (data/cache/, Step 9) give how often a word or a script occurs
in train.txt.

1. Rank the inputs by our tokens per word and by our-to-o200k_base ratio, and the words of the seven samples
   by our token count and by the same ratio.
2. Choose the three worst samples: the lowest sum of the two input ranks (ties: more tokens per word), unless
   --cases names them. test.txt is ranked but not chosen: it is the corpus, not a hand-picked input.
3. For each: the excerpt's token boundaries (" | ", fragments as \\xNN), its worst words, the numbers, the
   training evidence for the mechanism, and a 2-3 sentence draft.
4. The normalization mini-table: one sentence with Arabic kaf, yeh and heh, raw and normalized, token count
   under each tokenizer, and how often those letters occur in the (normalized) training text.

Writes report/tables/failure_inputs.tex, failure_words.tex, failure_cases.tex, normalization.tex, the section
"q3_failures" of data/stats/analysis.json, the Q3 draft in docs/analysis_notes.md, and a note there that the
handout's questions 4 and 5 do not exist.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.merges_report import load_counts  # noqa: E402
from analysis.report_utils import (ANALYSIS_JSON, CACHE_DIR, INPUT_LABELS, NOTES, SAMPLES, TABLES_DIR,  # noqa: E402
                                   TOKENIZER, WORDS_JSON, boundaries, latex_display, latex_escape, pct,
                                   read_analysis, update_analysis, update_notes, write_table)
from bpe import BPETokenizer  # noqa: E402
from common import setup_logging  # noqa: E402

SCRIPT = "analysis/failure_cases.py"
TOP_WORDS = 12
CASE_WORDS = 3
LATIN = regex.compile(r"[A-Za-z]")
DIACRITIC = regex.compile(r"[\u064B-\u065F\u0670]")  # harakat, tanween, shadda, sukun, superscript alef
ARABIC_LETTERS = regex.compile(r"[\u0643\u064A\u0647]")  # Arabic kaf, yeh, heh: normalize() maps them away
DIGITS = regex.compile(r"\d")
WORD_ONLY = regex.compile(r"[\p{L}\p{M}\u200c]+")  # example words for the letter-based cases


# ---------------------------------------------------------------------------
# Evidence from the training text
# ---------------------------------------------------------------------------

def training_shares(counts: Mapping[str, int]) -> dict[str, Any]:
    """Share of pre-token occurrences in train.txt that contain Latin letters, diacritics, Arabic letters, digits."""
    total = sum(counts.values())
    out: dict[str, Any] = {"pretokens": total, "distinct": len(counts)}
    for name, pattern in (("latin", LATIN), ("diacritic", DIACRITIC), ("arabic_kaf_yeh_heh", ARABIC_LETTERS),
                          ("digit", DIGITS)):
        n = sum(c for w, c in counts.items() if pattern.search(w))
        out[name] = {"occurrences": n, "share": round(n / total, 6) if total else None}
    return out


def train_count(word: str, counts: Mapping[str, int]) -> int:
    """How often the word is a whole pre-token in train.txt, with or without its leading space."""
    return counts.get(" " + word, 0) + counts.get(word, 0)


# ---------------------------------------------------------------------------
# Rankings and cases
# ---------------------------------------------------------------------------

def rank_inputs(q2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key, m in q2["inputs"].items():
        t = m["tokenizers"]
        rows.append({"input": key, "ours": t["ours"]["tokens_per_word"], "o200k": t["o200k_base"]["tokens_per_word"],
                     "gpt2": t["gpt2"]["tokens_per_word"], "ratio": round(t["ours"]["tokens_per_word"]
                                                                          / t["o200k_base"]["tokens_per_word"], 4),
                     "share_3plus": t["ours"]["share_words_3plus"]})
        best = min((n for n in t if n != "ours"), key=lambda n: t[n]["tokens_per_word"])
        rows[-1].update(best_other=best, best_other_tpw=t[best]["tokens_per_word"])
    by_tpw = sorted(rows, key=lambda r: -r["ours"])
    by_ratio = sorted(rows, key=lambda r: -r["ratio"])
    for r in rows:
        r["rank_tpw"] = by_tpw.index(r) + 1
        r["rank_ratio"] = by_ratio.index(r) + 1
        r["score"] = r["rank_tpw"] + r["rank_ratio"]
    return sorted(rows, key=lambda r: (r["score"], -r["ours"]))


def rank_words(words: dict[str, Any], counts: Mapping[str, int]) -> dict[str, list[dict[str, Any]]]:
    cols = words["columns"]
    i_ours, i_o200k, i_gpt2 = cols.index("ours"), cols.index("o200k_base"), cols.index("gpt2")
    rows = []
    for key in SAMPLES:
        for w, entry in words["inputs"][key].items():
            rows.append({"word": w, "input": key, "occurrences": entry[0], "ours": entry[i_ours],
                         "o200k": entry[i_o200k], "gpt2": entry[i_gpt2], "ratio": round(entry[i_ours] / entry[i_o200k], 4),
                         "train_count": train_count(w, counts)})
    by_ours = sorted(rows, key=lambda r: (-r["ours"], -r["ratio"], r["word"]))[:TOP_WORDS]
    by_ratio = sorted(rows, key=lambda r: (-r["ratio"], -r["ours"], r["word"]))[:TOP_WORDS]
    return {"by_ours": by_ours, "by_ratio": by_ratio}


def word_detail(tok: BPETokenizer, w: str, counts: Mapping[str, int]) -> dict[str, Any]:
    ids = tok.encode_ordinary(" " + w)
    return {"word": w, "tokens": len(ids), "boundaries": boundaries(tok.vocab[i] for i in ids),
            "train_count": train_count(w, counts)}


def case(tok: BPETokenizer, key: str, q2: dict[str, Any], words: dict[str, Any], ranked: list[dict[str, Any]],
         counts: Mapping[str, int]) -> dict[str, Any]:
    ex = q2["excerpts"][key]
    ids = tok.encode_ordinary(ex["text"])
    cols = words["columns"]
    i_ours = cols.index("ours")
    candidates = words["inputs"][key].items()
    if key != "numbers_dates" and any(WORD_ONLY.fullmatch(w) for w, _ in candidates):
        candidates = [(w, e) for w, e in candidates if WORD_ONLY.fullmatch(w)]
    worst = sorted(candidates, key=lambda kv: (-kv[1][i_ours], kv[0]))[:CASE_WORDS]
    out = {"input": key, "label": INPUT_LABELS[key], "ranking": next(r for r in ranked if r["input"] == key),
           "excerpt": ex["text"], "excerpt_words": ex["words"], "excerpt_tokens": ex["tokens"],
           "excerpt_boundaries": boundaries(tok.vocab[i] for i in ids),
           "worst_words": [word_detail(tok, w, counts) for w, _ in worst]}
    if key == "diacritics":
        for d in out["worst_words"]:
            bare = DIACRITIC.sub("", d["word"])
            d["bare"] = word_detail(tok, bare, counts)
    return out


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------

def mechanism(c: dict[str, Any], res: dict[str, Any]) -> str:
    """Two or three sentences on why this input costs so many tokens (a draft to edit)."""
    r, train = c["ranking"], res["training"]
    test = next(x for x in res["inputs_ranked"] if x["input"] == "test")
    head = (f"{c['label']} costs {r['ours']:.2f} tokens per word against {r['o200k']:.2f} for o200k_base "
            f"(ratio {r['ratio']:.2f}; {pct(r['share_3plus'])} of its words take 3 or more of our tokens)."
            + (f" We still beat o200k_base here, but it is among our costliest inputs."
               if r["ratio"] < 1 else ""))
    examples = "; ".join(f"{w['word']} → {w['boundaries']} ({w['tokens']} tokens, {w['train_count']:,} times in "
                         "train.txt)" for w in c["worst_words"])
    if c["input"] in ("roman_urdu", "urdu_english"):
        body = (f"Only {pct(train['latin']['share'], 2)} of the pre-token occurrences in train.txt contain a Latin "
                f"letter, so our merges cover few Latin-script words: {examples}. o200k_base learned Latin script "
                f"from English and many other languages; it needs {r['o200k']:.2f} tokens per word here and "
                f"{test['o200k']:.2f} on test.txt.")
    elif c["input"] == "diacritics":
        bare = "; ".join(f"the bare {w['bare']['word']} → {w['bare']['boundaries']} ({w['bare']['tokens']})"
                         for w in c["worst_words"] if "bare" in w)
        body = (f"Written Urdu usually omits diacritics: only {pct(train['diacritic']['share'], 2)} of the pre-token "
                "occurrences in train.txt carry one. Each mark is a separate 2-byte character between two letters, "
                f"so the merges learned for the bare word no longer apply: {examples}; against {bare}.")
    elif c["input"] == "numbers_dates":
        body = ("The pre-tokenizer cuts digit runs into groups of at most three and makes every separator its own "
                f"pre-token, so long numbers, phone numbers and ID numbers fall apart: {examples}.")
        if r["o200k"] > r["ours"]:
            body += " o200k_base, which has the same three-digit rule, does worse"
        if r["best_other_tpw"] < r["ours"]:
            body += (f"; {res['labels'][r['best_other']]} does better ({r['best_other_tpw']:.2f} tokens per word)"
                     if r["o200k"] > r["ours"] else
                     f" {res['labels'][r['best_other']]} does better ({r['best_other_tpw']:.2f} tokens per word)")
        body += "."
    elif c["input"] == "names":
        body = f"Foreign names, transliterated letter by letter, are rare in train.txt, so few merges cover them: {examples}."
    else:
        body = f"Its costliest words: {examples}."
    return f"{head} {body}"


def draft(analysis: dict[str, Any]) -> str:
    """The Q3 draft for docs/analysis_notes.md, from analysis.json alone."""
    res = analysis["q3_failures"]
    ranked = res["inputs_ranked"]
    order = ", ".join(f"{INPUT_LABELS[r['input']]} ({r['ours']:.2f}; {r['ratio']:.2f}x o200k)" for r in ranked[:4])
    parts = [f"Ranked by our tokens per word and by our-to-o200k_base ratio (sum of the two ranks), the worst inputs "
             f"are {order}. The three chosen: " + ", ".join(c["label"] for c in res["cases"]) + "."]
    for i, c in enumerate(res["cases"], start=1):
        ex_tokens = c["excerpt_tokens"]
        parts.append(f"**Case {i}: {c['label']}.** {mechanism(c, res)}\n\n"
                     f"Excerpt ({c['excerpt_words']} words; ours {ex_tokens['ours']} tokens, o200k_base "
                     f"{ex_tokens['o200k_base']}, GPT-2 {ex_tokens['gpt2']}): {c['excerpt_boundaries']}")
    n = res["normalization"]
    t = n["tokens"]
    parts.append(
        f"**Normalization.** The same {n['words']}-word sentence with Arabic kaf, yeh and heh in place of the Urdu "
        f"letters ({n['changed_chars']} characters differ) costs us {t['ours']['raw']} tokens raw and "
        f"{t['ours']['normalized']} normalized; o200k_base {t['o200k_base']['raw']} and "
        f"{t['o200k_base']['normalized']}, GPT-2 {t['gpt2']['raw']} and {t['gpt2']['normalized']}. The Arabic forms "
        f"occur in {res['training']['arabic_kaf_yeh_heh']['occurrences']:,} of the "
        f"{res['training']['pretokens']:,} pre-token occurrences in train.txt, because normalize() mapped them away "
        "before training, so every word containing one falls back to short pieces. That is why the cleaning step "
        "normalizes, and why text must be normalized the same way before encoding."
        + (" GPT-2 is the exception, cheaper on the raw sentence, presumably because the Arabic letters are more "
           "common than their Urdu forms in its training text." if t["gpt2"]["raw"] < t["gpt2"]["normalized"] else "")
        + "\n\n"
        f"Raw: {n['ours_raw_boundaries']}\n\nNormalized: {n['ours_normalized_boundaries']}")
    parts.append("Tables: `report/tables/failure_inputs.tex`, `failure_words.tex`, `failure_cases.tex`, "
                 "`normalization.tex`.")
    return "\n\n".join(parts)


Q45_NOTE = (
    '"Answer five questions": the handout numbers three, and its question 1 asks three separate things, so '
    'the report gives five answers, each with its own table. Q1, the merges 1-50 and every 5,000th '
    '(report/tables/r_merges.tex); Q2, which early merges rebuild a two-byte letter and which are not valid '
    'UTF-8 on their own (r_early.tex); Q3, the first whole word and the 20 longest tokens (r_longest.tex); '
    'Q4, tokens per word against GPT-2, o200k_base and xlm-roberta-base (compare.tex); Q5, three inputs our '
    'tokenizer handles badly (failure_cases.tex).')


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_tables(res: dict[str, Any], tables_dir: Path = TABLES_DIR) -> list[Path]:
    chosen = {c["input"] for c in res["cases"]}
    rows = [[latex_escape(INPUT_LABELS[r["input"]]) + (r" $\ast$" if r["input"] in chosen else ""),
             f"{r['ours']:.2f}", f"{r['o200k']:.2f}", f"{r['gpt2']:.2f}", f"{r['ratio']:.2f}",
             latex_escape(pct(r["share_3plus"])), str(r["rank_tpw"]), str(r["rank_ratio"])]
            for r in res["inputs_ranked"]]
    paths = [write_table("failure_inputs", "lrrrrrrr", ["Input", "Ours", "o200k", "GPT-2", "Ours/o200k",
                                                        r"Ours $\geq$3", "Rank", "Rank ratio"], rows, SCRIPT, tables_dir)]
    rows = [[latex_display(w["word"]), latex_escape(INPUT_LABELS[w["input"]]), str(w["ours"]), str(w["o200k"]),
             str(w["gpt2"]), f"{w['ratio']:.2f}", f"{w['train_count']:,}"] for w in res["words_ranked"]["by_ours"]]
    paths.append(write_table("failure_words", "llrrrrr", ["Word", "Input", "Ours", "o200k", "GPT-2", "Ratio",
                                                          "In train.txt"], rows, SCRIPT, tables_dir))
    rows = [[latex_escape(c["label"]), latex_display(c["excerpt_boundaries"]), str(c["excerpt_tokens"]["ours"]),
             str(c["excerpt_tokens"]["o200k_base"])] for c in res["cases"]]
    paths.append(write_table("failure_cases", r"lp{0.62\linewidth}rr", ["Input", "Our tokens", "Ours", "o200k"],
                             rows, SCRIPT, tables_dir))
    n = res["normalization"]
    names = [k for k in n["tokens"]]
    rows = [[label] + [str(n["tokens"][k][form]) for k in names]
            for label, form in (("Raw (Arabic kaf, yeh, heh)", "raw"), ("Normalized", "normalized"))]
    paths.append(write_table("normalization", "l" + "r" * len(names),
                             ["Sentence"] + [latex_escape(res["labels"][k]) for k in names], rows, SCRIPT, tables_dir))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="where Step 9 cached the word counts")
    p.add_argument("--cases", nargs=3, choices=list(SAMPLES), default=None, help="override the three chosen inputs")
    p.add_argument("--analysis-json", type=Path, default=ANALYSIS_JSON)
    p.add_argument("--words-json", type=Path, default=WORDS_JSON)
    p.add_argument("--notes", type=Path, default=NOTES)
    p.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("analysis")
    analysis = read_analysis(args.analysis_json)
    if "q2_compare" not in analysis or not args.words_json.exists():
        raise SystemExit("run analysis/compare_tokenizers.py first (it writes the token counts this script reads)")
    q2 = analysis["q2_compare"]
    words = json.loads(args.words_json.read_text(encoding="utf-8"))
    tok = BPETokenizer.load(args.tokenizer)
    counts = load_counts(tok, args.cache_dir)
    train = training_shares(counts)
    ranked = rank_inputs(q2)
    chosen = args.cases or [r["input"] for r in ranked if r["input"] in SAMPLES][:3]
    n = q2["normalization"]
    normalization = {**n, "ours_raw_boundaries": boundaries(tok.vocab[i] for i in tok.encode_ordinary(n["raw"])),
                     "ours_normalized_boundaries": boundaries(tok.vocab[i] for i in tok.encode_ordinary(n["normalized"]))}
    res = {"selection": "lowest rank_tpw + rank_ratio among the seven samples, ties to more tokens per word"
                        if not args.cases else f"--cases {' '.join(args.cases)}",
           "inputs_ranked": ranked, "words_ranked": rank_words(words, counts),
           "cases": [case(tok, key, q2, words, ranked, counts) for key in chosen],
           "training": train, "normalization": normalization, "labels": q2["labels"]}
    analysis = update_analysis("q3_failures", res, args.analysis_json)
    for path in write_tables(res, args.tables_dir):
        log.info("wrote %s", path)
    update_notes("q3", "Q3. Three inputs our tokenizer handles badly", draft(analysis), args.notes)
    update_notes("q4_q5", "The five answers", Q45_NOTE, args.notes)
    log.info("chose %s; updated %s and %s", chosen, args.analysis_json, args.notes)
    print(draft(analysis))
    return 0


if __name__ == "__main__":
    sys.exit(main())
