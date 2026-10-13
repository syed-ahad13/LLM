"""Generate the numbers and the compact tables of the report from the JSON stats; nothing is typed by hand.

    python report/make_numbers.py        # -> report/numbers.tex and report/tables/r_*.tex

Every number in report/report.tex is a macro defined in report/numbers.tex,
one \\newcommand per number, written here from these files:

    tokenizer/train_log.json                  the full training run (Step 9)
    tokenizer/reference_1mb_3000.train_log.json   the reference trainer on the 1 MB sample (Step 7)
    tokenizer/urdu_bpe_50k.json               the merges themselves (Q1 counts, merge tables)
    data/stats/equivalence_1mb.json           fast vs reference on the 1 MB sample (Step 8)
    data/stats/corpus_totals.json             MB, documents, words per split (Step 6)
    data/stats/dedup.json, normalize.json     cleaned size per source, normalization (Steps 5-5b)
    data/stats/char_audit.json, char_audit_normalized.json   the Step 4 audit, before and after normalize()
    data/final/manifest.json                  bytes per source and split, budget rates (Step 6)
    data/final/meta.json                      token counts of train.bin and val.bin (Step 10)
    data/stats/analysis.json                  Part C results (make analysis, Step 11)

The report's tables r_corpus, r_normalization, r_training, r_merges and
r_longest are written here too; the Part C tables (compare, failure_cases,
...) come from make analysis. A missing input or key stops the script with
its name, so the report is never built from stale or partial numbers.
"""

import argparse
import json
import sys
import time
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import regex

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report_utils import latex_escape, latex_token  # noqa: E402
from bpe import GPT2_PATTERN, URDU_PATTERN, BPETokenizer  # noqa: E402
from clean.data_card import normalization_rows  # noqa: E402

REPORT = ROOT / "report"
STATS = ROOT / "data" / "stats"
FINAL = ROOT / "data" / "final"
INPUTS = {
    "train_log": ROOT / "tokenizer" / "train_log.json",
    "reference_log": ROOT / "tokenizer" / "reference_1mb_3000.train_log.json",
    "tokenizer": ROOT / "tokenizer" / "urdu_bpe_50k.json",
    "equivalence": STATS / "equivalence_1mb.json",
    "corpus_totals": STATS / "corpus_totals.json",
    "dedup": STATS / "dedup.json",
    "normalize": STATS / "normalize.json",
    "char_audit": STATS / "char_audit.json",
    "char_audit_normalized": STATS / "char_audit_normalized.json",
    "manifest": FINAL / "manifest.json",
    "meta": FINAL / "meta.json",
    "analysis": STATS / "analysis.json",
}
SOURCE_ORDER = ("fineweb2_urd", "wiki_ur", "own_scrape", "ufal_umc", "makhzan", "urdu_news_1m")
SOURCE_LABELS = {"fineweb2_urd": "FineWeb-2 (urd\\_Arab)", "wiki_ur": "Urdu Wikipedia",
                 "own_scrape": "Own crawl: VOA Urdu, Hum Sub", "ufal_umc": "UFAL Urdu Monolingual",
                 "makhzan": "Makhzan", "urdu_news_1m": "Urdu News 1M"}
SOURCE_GENRES = {"fineweb2_urd": "web", "wiki_ur": "encyclopedia", "own_scrape": "news, blogs",
                 "ufal_umc": "mixed", "makhzan": "journalism, essays", "urdu_news_1m": "news"}
# The normalization rules of normalize.py (clean/data_card.py RULES, same order), as LaTeX.
RULE_LABELS = [
    r"Presentation forms $\to$ plain letters",
    r"Arabic yeh \ur{ي ى} $\to$ Urdu yeh \ur{ی}",
    r"Arabic kaf \ur{ك} $\to$ \ur{ک} (Sindhi \ur{ڪ} kept)",
    r"Arabic heh \ur{ه} $\to$ \ur{ہ}; \ur{ة} $\to$ \ur{ۃ}; \ur{ۀ} $\to$ \ur{ۂ}",
    r"Letter + combining madda/hamza $\to$ one code point (NFC)",
    r"Arabic-Indic and Urdu digits $\to$ ASCII",
    r"Tatweel removed",
    r"ZWJ, bidi marks, BOM, controls removed; ZWNJ only inside words",
    r"Whitespace: NBSP, tabs $\to$ space; runs collapsed; lines trimmed",
    r"No space before \ur{۔ ، ؟ ؛}",
]
EVERY = 5000
EARLY = 50
LONGEST = 20
SCRIPT = "report/make_numbers.py"


class MissingInput(SystemExit):
    pass


def load_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise MissingInput("missing inputs for the report numbers:\n  " + "\n  ".join(missing))
    out: dict[str, Any] = {}
    for name, path in paths.items():
        if name == "tokenizer":
            out[name] = BPETokenizer.load(path)
        else:
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def n(x: float) -> str:
    return f"{int(round(x)):,}"


def f1(x: float) -> str:
    return f"{x:,.1f}"


def f2(x: float) -> str:
    return f"{x:,.2f}"


def mb(b: float) -> str:
    return f"{b / 1e6:,.2f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}\\%"


def tok_tex(tok: BPETokenizer, rank: int) -> str:
    return latex_token(tok.vocab[tok.first_merge_id + rank - 1])


def pieces_tex(pieces: Sequence[str]) -> str:
    """Pre-tokenizer pieces as LaTeX, separated by thin bars."""
    return r" \textbar{} ".join(latex_token(p.encode("utf-8")) for p in pieces)


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def is_letter(b: bytes) -> bool:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(s) == 1 and unicodedata.category(s).startswith("L")


def is_valid(b: bytes) -> bool:
    try:
        b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


EARLY_CATEGORIES = ("Two-byte letter", "Other two-byte character", "Space + lead byte (not valid UTF-8)",
                    "Ends mid-character (not valid UTF-8)", "Space + one letter", "Longer piece")


def early_category(token: bytes) -> str:
    """The category of an early merged token, from its bytes alone (Q2 of the report)."""
    if not is_valid(token):
        rest = token[1:]
        if token[:1] == b" " and rest and 0xC2 <= rest[0] <= 0xF4 and len(rest) == 1:
            return EARLY_CATEGORIES[2]
        return EARLY_CATEGORIES[3]
    text = token.decode("utf-8")
    if len(token) == 2 and len(text) == 1:
        return EARLY_CATEGORIES[0] if is_letter(token) else EARLY_CATEGORIES[1]
    core = text[1:] if text.startswith(" ") else None
    if core and len(regex.findall(r"\p{L}", core)) == 1 and regex.fullmatch(r"[\p{L}\p{M}]+", core):
        return EARLY_CATEGORIES[4]
    return EARLY_CATEGORIES[5]


def rank_of(tok: BPETokenizer, token: str) -> int | None:
    target = token.encode("utf-8")
    for i in range(tok.first_merge_id, tok.vocab_size):
        if tok.vocab[i] == target:
            return i - tok.first_merge_id + 1
    return None


def numbers(inp: dict[str, Any]) -> dict[str, str]:
    """{macro name: LaTeX text}; every value is computed from the inputs."""
    tok: BPETokenizer = inp["tokenizer"]
    run, ref, eq = inp["train_log"], inp["reference_log"], inp["equivalence"]
    ct, man, meta, an = inp["corpus_totals"], inp["manifest"], inp["meta"], inp["analysis"]
    m: dict[str, str] = {}

    # Corpus (Step 6)
    t = ct["total"]
    m.update(CorpusMB=f2(t["mb"]), CorpusMiB=f2(t["mib"]), CorpusDocs=n(t["docs"]), CorpusWords=n(t["words"]),
             CorpusUniqueWords=n(t["unique_words"]))
    for split, name in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        s = ct["splits"][split]
        m[f"{name}MB"], m[f"{name}Docs"] = f2(s["mb"]), n(s["docs"])
    src = man["sources"]
    total_bytes = man["totals"]["bytes"]
    m["OwnMB"] = mb(src["own_scrape"]["text_bytes"])
    m["SourcesUsed"] = n(sum(1 for s in src.values() if s["docs"]))
    m["SourcesAll"] = n(len(src))
    m["CleanedMB"] = mb(sum(inp["dedup"][s]["bytes_out"] for s in src if s in inp["dedup"]))
    for key, name in (("fineweb2_urd", "FineWeb"), ("wiki_ur", "Wiki"), ("ufal_umc", "Ufal"),
                      ("urdu_news_1m", "News")):
        m[f"{name}Share"] = pct(src[key]["bytes"] / total_bytes)
        m[f"{name}Rate"] = pct(man["budget"]["sources"][key]["rate"])
    genres: set[str] = set()
    for split in man["splits"].values():
        genres |= {g for g, b in split["genres"].items() if b}
    m["GenresCount"] = n(len(genres))
    m["FineWebCap"] = pct(man["budget"]["policy"]["caps"]["fineweb2_urd"], 0)
    m["Seed"] = str(man["seed"])
    m["TrainTokens"], m["ValTokens"] = n(meta["train_tokens"]), n(meta["val_tokens"])
    m["TrainBytesPerToken"] = f2(meta["splits"]["train"]["bytes_per_token"])
    m["TrainTokensPerWord"] = f2(meta["splits"]["train"]["tokens_per_word"])

    # Normalization and ZWNJ (Steps 4-5)
    rows, chars = normalization_rows({"audit": inp["char_audit"], "audit_norm": inp["char_audit_normalized"]})
    m["AuditChars"] = n(chars)
    norm = inp["normalize"]
    used = list(src)
    b_in = sum(norm[s]["bytes_in"] for s in used if s in norm)
    b_out = sum(norm[s]["bytes_out"] for s in used if s in norm)
    m["NormInMB"], m["NormOutMB"], m["NormShrink"] = mb(b_in), mb(b_out), pct((b_in - b_out) / b_in, 2)
    raw, normed = inp["char_audit"]["sources"], inp["char_audit_normalized"]["sources"]
    zw_raw = sum(r["categories"]["zwnj"]["count"] for r in raw.values())
    zw_kept = sum(r["categories"]["zwnj"]["count"] for r in normed.values())
    m.update(ZwnjRaw=n(zw_raw), ZwnjKept=n(zw_kept), ZwnjRemoved=n(zw_raw - zw_kept))
    mk = raw.get("makhzan", {})
    neigh = mk.get("zwnj_neighbours", {})
    m["ZwnjMakhzan"] = n(mk.get("categories", {}).get("zwnj", {}).get("count", 0))
    m["ZwnjMakhzanTeh"] = n(sum(c for k, c in neigh.items() if k.startswith("ت + ZWNJ +")))
    letter = regex.compile(r"^\p{L} \+ ZWNJ \+ \p{L}$")
    m["ZwnjMakhzanBetween"] = n(sum(c for k, c in neigh.items() if letter.match(k)))

    # Pre-tokenizer (bpe.py)
    gpt2, ours = regex.compile(GPT2_PATTERN), regex.compile(URDU_PATTERN)
    for word, name in (("اُردُو", "Handout"), ("اُردو", "Plain"), ("صورت‌حال", "Zwnj")):
        g = gpt2.findall(word)
        m[f"GptPieces{name}"] = pieces_tex(g)
        m[f"GptCount{name}"] = n(len(g))
        m[f"OurPieces{name}"] = pieces_tex(ours.findall(word))
    m["OurPiecesYear"] = pieces_tex(ours.findall("2024ء"))
    groups = int(regex.search(r"\\p\{N\}\{1,(\d+)\}", URDU_PATTERN).group(1))
    m["NumberPieces"] = n(2 * sum(10**k for k in range(1, groups + 1)))  # each with or without a leading space

    # Training (Steps 7-9)
    m["RefSeconds"] = f1(ref["phases"]["train_reference"])
    m["RefSampleBytes"] = n(ref["trained_on"]["bytes"])
    m["EqMerges"] = n(eq["merges_compared"])
    m["EqVerdict"] = "identical" if eq["identical_merges"] and eq["identical_counts"] else "NOT identical"
    m["FastSeconds"] = f2(eq["fast_seconds"])
    m["Speedup"] = n(eq["speedup"])
    m["EqDistinct"], m["EqOccurrences"] = n(eq["distinct_words"]), n(eq["word_occurrences"])
    ph = run["phases"]
    m.update(RunCount=f1(ph["count_words"]), RunInit=f1(ph["init"]), RunMerges=f1(ph["merges"]),
             RunSave=f1(ph["save"]), RunTotal=f1(run["total_seconds"]), RunMergesPerSec=n(run["merges_per_s_overall"]),
             RunPeakGB=f2(run["peak_rss_mb"]["trainer"] / 1000), RunDistinct=n(run["distinct_words"]),
             RunPretokens=n(run["word_occurrences"]), RunWorkers=n(run["workers"]),
             RunTrainMB=mb(run["train_txt"]["bytes"]), RunMergesDone=n(run["merges"]),
             RunVocab=n(run["vocab_size"]))
    mach = run["machine"]
    m.update(MachineCPU=latex_escape(mach["cpu_model"]), MachineCores=n(mach["physical_cores"]),
             MachineRAM=n(mach["ram_bytes"] / 2**30), MachineOS=latex_escape(mach["os_version"]),
             MachinePython=latex_escape(mach["python_version"]))
    chk = run["checks"]
    m.update(ValTokensPerWord=f2(chk["metrics"]["tokens_per_word"]), ValBytesPerToken=f2(chk["metrics"]["bytes_per_token"]),
             ValSingleShare=pct(chk["metrics"]["single_token_word_share"]),
             ChecksVerdict="all passed" if chk["passed"] else "FAILED",
             MergeIdFirst=n(chk["merge_ids"][0]), MergeIdLast=n(chk["merge_ids"][1]),
             RoundTripDocs=n(chk["round_trip"]["docs"]), RepeatTokens=n(chk["tokens_repeating_earlier_bytes"]))

    # Q1 (counted on the tokenizer itself)
    first = tok.first_merge_id
    early = [tok.vocab[first + r - 1] for r in range(1, EARLY + 1)]
    letters = [r for r, b in enumerate(early, 1) if is_letter(b)]
    other = [r for r, b in enumerate(early, 1) if len(b) == 2 and is_valid(b) and not is_letter(b)
             and len(b.decode("utf-8")) == 1]
    invalid = [r for r, b in enumerate(early, 1) if not is_valid(b)]
    m.update(QOneEarly=n(EARLY), QOneLetters=n(len(letters)), QOneChars=n(len(other)), QOneInvalid=n(len(invalid)),
             QOneCharList=", ".join(f"{tok_tex(tok, r)} at {r}" for r in other),
             QOneInvalidList=", ".join(f"{tok_tex(tok, r)} at {r}" for r in invalid),
             QOneLetterList=", ".join(f"{tok_tex(tok, r)}~{r}" for r in letters[:5]))
    q1 = an["q1_merges"]
    fw = q1["first_word"]
    m.update(QOneFirstWord=latex_token(tok.vocab[first + fw["rank"] - 1]), QOneFirstWordRank=n(fw["rank"]),
             QOneFirstWordCount=n(fw["pretoken_count"]))
    b0, bl = q1["buckets"][0], q1["buckets"][-1]
    m.update(QOneLetterShareFirst=pct(b0["shares"]["LETTER"], 0), QOneWordShareFirst=pct(b0["shares"]["WORD"], 0),
             QOneLetterShareLast=pct(bl["shares"]["LETTER"], 0), QOneWordShareLast=pct(bl["shares"]["WORD"], 0),
             QOneLastBucket=f"{n(bl['ranks'][0])}--{n(bl['ranks'][1])}", QOneFirstBucket=f"1--{n(b0['ranks'][1])}")
    steps = list(range(EVERY, len(tok.merges) + 1, EVERY))
    if steps[-1] != len(tok.merges):
        steps.append(len(tok.merges))
    arabic_word = regex.compile(r" ?[\p{Script=Arabic}\p{M}\u200c]*\p{Script=Arabic}[\p{Script=Arabic}\p{M}\u200c]*")
    words_at = [r for r in steps if is_valid(tok.vocab[first + r - 1]) and arabic_word.fullmatch(
        tok.vocab[first + r - 1].decode("utf-8")) and len(regex.findall(r"\p{L}", tok.vocab[first + r - 1].decode(
            "utf-8"))) >= 2]
    m["QOneStepTotal"], m["QOneStepWords"] = n(len(steps)), n(len(words_at))
    m["QOneStepOtherList"] = ", ".join(tok_tex(tok, r) for r in steps if r not in words_at) or "none"
    by_len = sorted(range(1, len(tok.merges) + 1), key=lambda r: (-len(tok.vocab[first + r - 1]), r))[:LONGEST]
    m["QOneLongestMax"] = n(len(tok.vocab[first + by_len[0] - 1]))
    m["QOneLongestMin"] = n(len(tok.vocab[first + by_len[-1] - 1]))
    station = rank_of(tok, " اسٹیشن")
    if station:
        m["QOneStationRank"] = n(station)
        m["QOneStationCount"] = n(tok.merge_counts[station - 1])
    stubs = sum(t["docs"].get("wiki_ur", 0) for t in q1["templates"].values())
    m["QOneStubDocs"], m["QOneWikiDocs"] = n(stubs), n(q1["wiki_train_docs"])
    m["QOneStubShare"] = pct(stubs / q1["wiki_train_docs"])
    wiki_heavy = [e for e in q1["longest"] if e.get("top_source") == "wiki_ur" and (e.get("top_share") or 0) >= 0.9]
    m["QOneLongestWiki"] = n(len(wiki_heavy))

    # Q2
    q2 = an["q2_compare"]
    test = q2["inputs"]["test"]["tokenizers"]
    names = q2["tokenizer_order"]
    hf = [x for x in names if x not in ("ours", "gpt2", "o200k_base")]
    m["QTwoTestWords"] = n(q2["inputs"]["test"]["words"])
    m.update(QTwoOurs=f2(test["ours"]["tokens_per_word"]), QTwoGpt=f2(test["gpt2"]["tokens_per_word"]),
             QTwoOk=f2(test["o200k_base"]["tokens_per_word"]), QTwoOursBpt=f2(test["ours"]["bytes_per_token"]),
             QTwoGptBpt=f2(test["gpt2"]["bytes_per_token"]), QTwoOkBpt=f2(test["o200k_base"]["bytes_per_token"]),
             QTwoOursThree=pct(test["ours"]["share_words_3plus"]), QTwoGptThree=pct(test["gpt2"]["share_words_3plus"]),
             QTwoOkThree=pct(test["o200k_base"]["share_words_3plus"]))
    if hf:
        m["QTwoHF"] = f2(test[hf[0]]["tokens_per_word"])
        m["QTwoHFName"] = latex_escape(q2["labels"][hf[0]])
    others = [x for x in names if x != "ours"]
    wins = [k for k, v in q2["inputs"].items()
            if v["tokenizers"]["ours"]["tokens_per_word"] <= min(v["tokenizers"][x]["tokens_per_word"] for x in others)]
    m["QTwoWins"], m["QTwoInputs"] = n(len(wins)), n(len(q2["inputs"]))

    # Q3
    q3 = an["q3_failures"]
    for case, name in zip(q3["cases"], ("A", "B", "C")):
        r = case["ranking"]
        m[f"QThree{name}Label"] = latex_escape(case["label"])
        m[f"QThree{name}Ours"], m[f"QThree{name}Ok"] = f2(r["ours"]), f2(r["o200k"])
        m[f"QThree{name}Ratio"] = f2(r["ratio"])
        m[f"QThree{name}ThreePlus"] = pct(r["share_3plus"])
    tr = q3["training"]
    m["QThreeLatinShare"] = pct(tr["latin"]["share"], 2)
    m["QThreeDiacriticShare"] = pct(tr["diacritic"]["share"], 2)
    norm_tokens = q2["normalization"]["tokens"]["ours"]
    m["QThreeNormRaw"], m["QThreeNormNormalized"] = n(norm_tokens["raw"]), n(norm_tokens["normalized"])
    m["QThreeNormWords"] = n(q2["normalization"]["words"])
    return m


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table(path: Path, colspec: str, header: Sequence[str] | None, rows: Sequence[Sequence[str]]) -> Path:
    lines = [f"% Generated by {SCRIPT} (make report); do not edit by hand.", f"\\begin{{tabular}}{{{colspec}}}",
             r"\toprule"]
    if header:
        lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def blocks(entries: Sequence[Sequence[str]], n_blocks: int) -> list[list[str]]:
    """entries laid out column-major in n_blocks side-by-side blocks."""
    height = -(-len(entries) // n_blocks)
    width = len(entries[0])
    rows = []
    for i in range(height):
        row: list[str] = []
        for b in range(n_blocks):
            k = b * height + i
            row += list(entries[k]) if k < len(entries) else [""] * width
        rows.append(row)
    return rows


def tables(inp: dict[str, Any], out_dir: Path) -> list[Path]:
    tok: BPETokenizer = inp["tokenizer"]
    man, dedup = inp["manifest"], inp["dedup"]
    first = tok.first_merge_id
    written = []

    total = man["totals"]["bytes"]
    rows = []
    for s in [x for x in SOURCE_ORDER if x in man["sources"]]:
        e = man["sources"][s]
        rate = man["budget"]["sources"][s]["rate"]
        rows.append([SOURCE_LABELS[s], SOURCE_GENRES[s], mb(dedup[s]["bytes_out"]), mb(e["bytes"]),
                     pct(e["bytes"] / total), "whole" if rate >= 1 else pct(rate)])
    rows.append([r"\textbf{Total}", "", mb(sum(dedup[s]["bytes_out"] for s in man["sources"])), mb(total),
                 r"100\%", ""])
    written.append(table(out_dir / "r_corpus.tex", "llrrrr",
                         ["Source", "Genre", "Cleaned MB", "Corpus MB", "Share", "Kept"], rows))

    nrows, _ = normalization_rows({"audit": inp["char_audit"], "audit_norm": inp["char_audit_normalized"]})
    written.append(table(out_dir / "r_normalization.tex", "lrr", ["Rule (in order)", "Changed", "per M chars"],
                         [[label, n(c), n(r)] for label, (_, c, r) in zip(RULE_LABELS, nrows)]))

    run, ref, eq = inp["train_log"], inp["reference_log"], inp["equivalence"]
    ph = run["phases"]
    written.append(table(out_dir / "r_training.tex", "lr", ["Full run, phase", "Seconds"],
                         [["Word counting (" + n(run["workers"]) + " processes)", f1(ph["count_words"])],
                          ["Initialization (pairs, index, heap)", f1(ph["init"])],
                          ["Merges (" + n(run["merges"]) + ")", f1(ph["merges"])],
                          ["Save", f1(ph["save"])],
                          [r"\textbf{Total}", f1(run["total_seconds"])]]))

    ranks = list(range(1, EARLY + 1)) + list(range(EVERY, len(tok.merges) + 1, EVERY))
    if ranks[-1] != len(tok.merges):
        ranks.append(len(tok.merges))
    entries = [[n(r), tok_tex(tok, r)] for r in ranks]
    written.append(table(out_dir / "r_merges.tex", "rl" * 5, ["Rank", "Token"] * 5, blocks(entries, 5)))

    groups: dict[str, list[int]] = {c: [] for c in EARLY_CATEGORIES}
    for r in range(1, EARLY + 1):
        groups[early_category(tok.vocab[first + r - 1])].append(r)
    rows = [[c, n(len(rs)), ", ".join(f"{tok_tex(tok, r)}\\,{r}" for r in rs) or "--"]
            for c, rs in groups.items() if rs]
    written.append(table(out_dir / "r_early.tex", r"lrp{0.62\linewidth}",
                         [f"Merges 1--{EARLY}", "Count", "Tokens and their ranks"], rows))

    by_len = sorted(range(1, len(tok.merges) + 1), key=lambda r: (-len(tok.vocab[first + r - 1]), r))[:LONGEST]
    entries = [[n(r), tok_tex(tok, r), n(len(tok.vocab[first + r - 1]))] for r in by_len]
    written.append(table(out_dir / "r_longest.tex", "rlr" * 2, ["Rank", "Token", "Bytes"] * 2, blocks(entries, 2)))
    return written


def write_numbers(values: dict[str, str], path: Path, sources: dict[str, Path]) -> Path:
    bad = [k for k in values if not regex.fullmatch(r"[A-Za-z]+", k)]
    if bad:
        raise ValueError(f"macro names must be letters only: {bad}")
    lines = [f"% Generated by {SCRIPT} on {time.strftime('%Y-%m-%d %H:%M')} from:"]
    lines += [f"%   {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}" for p in sources.values()]
    lines += ["% Do not edit by hand; rerun the script."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPORT / "numbers.tex")
    p.add_argument("--tables-dir", type=Path, default=REPORT / "tables")
    for name, path in INPUTS.items():
        p.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path, dest=name)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = {name: getattr(args, name) for name in INPUTS}
    inp = load_inputs(sources)
    try:
        values = numbers(inp)
    except KeyError as e:
        raise SystemExit(f"a key the report needs is missing from the inputs: {e}") from None
    write_numbers(values, args.out, sources)
    for path in tables(inp, args.tables_dir):
        print(f"wrote {path}")
    print(f"wrote {args.out} ({len(values)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
