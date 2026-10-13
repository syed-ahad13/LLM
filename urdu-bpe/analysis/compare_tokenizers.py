"""Part C, question 2: how does our tokenizer compare with existing tokenizers?

    python analysis/compare_tokenizers.py            # -> report/tables/compare*.tex, report/figures/tokens_per_word.pdf
    python analysis/compare_tokenizers.py --no-hf    # without the optional Hugging Face tokenizer

The only file allowed to import tiktoken and tokenizers (CLAUDE.md, hard rule 3), so it also counts tokens
for question 3 (analysis/failure_cases.py reads them from data/stats/).

Tokenizers: ours (tokenizer/urdu_bpe_50k.json); tiktoken "gpt2" and "o200k_base"; and, if it downloads
without a login, one Hugging Face multilingual tokenizer.json (default FacebookAI/xlm-roberta-base, fetched
with requests from its resolve URL into data/cache/hf/ and loaded with tokenizers.Tokenizer.from_file; no
special tokens added).

Inputs: test.txt (split on <|endoftext|>, separators removed, each document encoded on its own) and the
seven samples in analysis/samples/ (one document each). Every tokenizer gets the same text, normalize(text);
tiktoken encodes with disallowed_special=() so a literal special string is ordinary text.

Metrics per input and tokenizer:
- tokens per word: tokens / words, words = str.split() of the same text for every tokenizer;
- bytes per token (UTF-8 bytes of the text) and characters per token;
- share of words split into 3 or more tokens: each distinct word is encoded as " " + word, the form it
  takes after a space in running text, and the share is weighted by how often the word occurs.

Also counted for question 3: every distinct word of every input under every tokenizer
(data/stats/analysis_words.json), the excerpt of each sample shown with token boundaries
(report_utils.excerpt), and the sentence in analysis/samples/unnormalized.txt before and after normalize().

Writes report/tables/compare.tex (tokens per word: rows = inputs, columns = tokenizers),
report/tables/compare_metrics.tex (all four metrics on test.txt), report/figures/tokens_per_word.pdf
(grouped bars), the section "q2_compare" of data/stats/analysis.json and the Q2 draft in
docs/analysis_notes.md.
"""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import requests
import tiktoken
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.report_utils import (ANALYSIS_JSON, CACHE_DIR, INPUT_LABELS, NOTES, PALETTE, SAMPLES,  # noqa: E402
                                   SAMPLES_DIR, TABLES_DIR, TEST_TXT, TOKENIZER, UNNORMALIZED, WORDS_JSON,
                                   excerpt, latex_escape, load_inputs, pct, pyplot, save_figure, update_analysis,
                                   update_notes, write_table)
from bpe import BPETokenizer  # noqa: E402
from common import setup_logging, sha256_file, write_json  # noqa: E402
from normalize import normalize  # noqa: E402

SCRIPT = "analysis/compare_tokenizers.py"
HF_MODEL = "FacebookAI/xlm-roberta-base"
LABELS = {"ours": "Ours (50k)", "gpt2": "GPT-2", "o200k_base": "o200k_base"}

Encoder = Callable[[str], list[int]]


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

def hf_tokenizer(model: str, cache_dir: Path, log: Any) -> tuple[Encoder, dict[str, Any]] | None:
    """A Hugging Face tokenizer.json, downloaded once without a login; None if it cannot be had."""
    path = cache_dir / "hf" / (model.replace("/", "__") + ".tokenizer.json")
    url = f"https://huggingface.co/{model}/resolve/main/tokenizer.json"
    info: dict[str, Any] = {"model": model, "url": url, "path": str(path.relative_to(ROOT))
                            if path.is_relative_to(ROOT) else str(path)}
    if not path.exists():
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            log.warning("skipping %s: %s", model, e)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        info["revision"] = r.headers.get("x-repo-commit")
    info["sha256"] = sha256_file(path)
    tk = Tokenizer.from_file(str(path))
    info["vocab_size"] = tk.get_vocab_size()
    return (lambda text: tk.encode(text, add_special_tokens=False).ids), info


def load_tokenizers(args: argparse.Namespace, log: Any) -> tuple[dict[str, Encoder], dict[str, Any]]:
    ours = BPETokenizer.load(args.tokenizer)
    encoders: dict[str, Encoder] = {"ours": ours.encode_ordinary}
    info: dict[str, Any] = {"ours": {"path": str(args.tokenizer), "vocab_size": ours.vocab_size}}
    for name in ("gpt2", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        encoders[name] = lambda text, enc=enc: enc.encode(text, disallowed_special=())
        info[name] = {"library": f"tiktoken {tiktoken.__version__}", "vocab_size": enc.n_vocab}
    if not args.no_hf:
        hf = hf_tokenizer(args.hf_model, args.cache_dir, log)
        if hf:
            key = args.hf_model.split("/")[-1]
            encoders[key], info[key] = hf
            LABELS[key] = key
    return encoders, info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def measure(docs: Sequence[str], encoders: dict[str, Encoder]) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Metrics of one input under every tokenizer, and {word: [occurrences, tokens under each tokenizer]}."""
    words = Counter(w for d in docs for w in d.split())
    n_words = sum(words.values())
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    n_chars = sum(len(d) for d in docs)
    per_word: dict[str, list[int]] = {w: [c] for w, c in words.items()}
    metrics: dict[str, Any] = {"docs": len(docs), "words": n_words, "distinct_words": len(words),
                               "bytes": n_bytes, "chars": n_chars, "tokenizers": {}}
    for name, encode in encoders.items():
        tokens = sum(len(encode(d)) for d in docs)
        split3 = 0
        for w, entry in per_word.items():
            k = len(encode(" " + w))
            entry.append(k)
            if k >= 3:
                split3 += entry[0]
        metrics["tokenizers"][name] = {
            "tokens": tokens,
            "tokens_per_word": round(tokens / n_words, 4) if n_words else None,
            "bytes_per_token": round(n_bytes / tokens, 4) if tokens else None,
            "chars_per_token": round(n_chars / tokens, 4) if tokens else None,
            "share_words_3plus": round(split3 / n_words, 4) if n_words else None,
        }
    return metrics, per_word


def counts_for(text: str, encoders: dict[str, Encoder]) -> dict[str, int]:
    return {name: len(encode(text)) for name, encode in encoders.items()}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_tables(res: dict[str, Any], tables_dir: Path = TABLES_DIR) -> list[Path]:
    names = res["tokenizer_order"]
    header = ["Input", "Words"] + [latex_escape(res["labels"][n]) for n in names]
    rows = []
    for key, m in res["inputs"].items():
        best = min(m["tokenizers"][n]["tokens_per_word"] for n in names)
        cells = []
        for n in names:
            v = m["tokenizers"][n]["tokens_per_word"]
            cells.append(rf"\textbf{{{v:.2f}}}" if v == best else f"{v:.2f}")
        rows.append([latex_escape(INPUT_LABELS[key]), f"{m['words']:,}", *cells])
    paths = [write_table("compare", "lr" + "r" * len(names), header, rows, SCRIPT, tables_dir)]
    t = res["inputs"]["test"]["tokenizers"]
    metric_rows = [[latex_escape(res["labels"][n]), f"{t[n]['tokens_per_word']:.2f}", f"{t[n]['bytes_per_token']:.2f}",
                    f"{t[n]['chars_per_token']:.2f}", latex_escape(pct(t[n]['share_words_3plus']))] for n in names]
    paths.append(write_table("compare_metrics", "lrrrr", ["Tokenizer", "Tokens/word", "Bytes/token", "Chars/token",
                                                          r"Words $\geq$3 tokens"], metric_rows, SCRIPT, tables_dir))
    return paths


def plot(res: dict[str, Any]) -> Path:
    plt = pyplot()
    names = res["tokenizer_order"]
    keys = list(res["inputs"])
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    width = 0.8 / len(names)
    top = max(res["inputs"][k]["tokenizers"][n]["tokens_per_word"] for k in keys for n in names)
    for j, (color, name) in enumerate(zip(PALETTE, names)):
        xs = [i + (j - (len(names) - 1) / 2) * width for i in range(len(keys))]
        vals = [res["inputs"][k]["tokenizers"][name]["tokens_per_word"] for k in keys]
        ax.bar(xs, vals, width=width * 0.92, color=color, edgecolor="white", linewidth=0.8,
               label=res["labels"][name])
        if name == "ours":  # selective direct labels: our tokenizer's values only
            for x, v in zip(xs, vals):
                ax.text(x, v + top * 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(keys)), [INPUT_LABELS[k] for k in keys], rotation=25, ha="right")
    ax.set_ylabel("Tokens per word")
    ax.set_ylim(0, top * 1.12)
    ax.legend(ncols=len(names), loc="upper left", fontsize=7.5)
    return save_figure(fig, "tokens_per_word")


def draft(analysis: dict[str, Any]) -> str:
    """The Q2 draft for docs/analysis_notes.md, from analysis.json alone."""
    res = analysis["q2_compare"]
    labels, names = res["labels"], res["tokenizer_order"]
    others = [n for n in names if n != "ours"]
    t = res["inputs"]["test"]["tokenizers"]
    s1 = (f"On test.txt ({res['inputs']['test']['words']:,} words) our tokenizer needs "
          f"{t['ours']['tokens_per_word']:.2f} tokens per word, against "
          + ", ".join(f"{t[n]['tokens_per_word']:.2f} for {labels[n]}" for n in others)
          + f"; it splits {pct(t['ours']['share_words_3plus'])} of words into 3 or more tokens, GPT-2 "
          f"{pct(t['gpt2']['share_words_3plus'])}. GPT-2's byte-level vocabulary was learned on English web text, so "
          f"an Urdu letter (2 bytes) often costs a token of its own ({t['gpt2']['bytes_per_token']:.2f} bytes per "
          f"token against our {t['ours']['bytes_per_token']:.2f}), while o200k_base, with four times our "
          "vocabulary spread over many languages, sits in between.")
    wins = [k for k, m in res["inputs"].items()
            if m["tokenizers"]["ours"]["tokens_per_word"] <= min(m["tokenizers"][n]["tokens_per_word"] for n in others)]
    losses = {k: min(others, key=lambda n: m["tokenizers"][n]["tokens_per_word"])
              for k, m in res["inputs"].items() if k not in wins}
    s2 = (f"Ours is the most efficient on {len(wins)} of {len(res['inputs'])} inputs ("
          + ", ".join(INPUT_LABELS[k] for k in wins) + ")")
    if losses:
        s2 += ("; it loses on " + ", ".join(
            f"{INPUT_LABELS[k]} ({res['inputs'][k]['tokenizers']['ours']['tokens_per_word']:.2f} against "
            f"{res['inputs'][k]['tokenizers'][n]['tokens_per_word']:.2f} for {labels[n]})" for k, n in losses.items())
            + "; question 3 looks at why")
    s2 += "."
    return (f"{s1}\n\n{s2}\n\nTables: `report/tables/compare.tex` (tokens per word), "
            "`report/tables/compare_metrics.tex` (all metrics, test.txt). Figure: `report/figures/tokens_per_word.pdf`.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--test", type=Path, default=TEST_TXT)
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument("--hf-model", default=HF_MODEL, help="optional Hugging Face tokenizer (public, no login)")
    p.add_argument("--no-hf", action="store_true", help="compare with tiktoken's encodings only")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--analysis-json", type=Path, default=ANALYSIS_JSON)
    p.add_argument("--words-json", type=Path, default=WORDS_JSON)
    p.add_argument("--notes", type=Path, default=NOTES)
    p.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("analysis")
    encoders, info = load_tokenizers(args, log)
    inputs = load_inputs(args.test, args.samples_dir)
    metrics, words = {}, {}
    for key, docs in inputs.items():
        start = time.perf_counter()
        metrics[key], words[key] = measure(docs, encoders)
        log.info("%s: %d words, tokens per word %s (%.1f s)", key, metrics[key]["words"],
                 {n: m["tokens_per_word"] for n, m in metrics[key]["tokenizers"].items()}, time.perf_counter() - start)
    excerpts = {}
    for key in SAMPLES:
        text = excerpt(inputs[key][0])
        excerpts[key] = {"text": text, "words": len(text.split()), "tokens": counts_for(text, encoders)}
    raw = (args.samples_dir / UNNORMALIZED).read_text(encoding="utf-8").strip()
    normed = normalize(raw)
    normalization = {"raw": raw, "normalized": normed, "words": len(normed.split()),
                     "changed_chars": sum(a != b for a, b in zip(raw, normed)),
                     "tokens": {n: {"raw": len(e(raw)), "normalized": len(e(normed))} for n, e in encoders.items()}}
    res = {"tokenizers": info, "tokenizer_order": list(encoders), "labels": {n: LABELS[n] for n in encoders},
           "inputs": metrics, "excerpts": excerpts, "normalization": normalization,
           "definitions": {"words": "str.split() of normalize(text), the same for every tokenizer",
                           "share_words_3plus": "occurrence-weighted share of words for which ' ' + word encodes "
                                                "to 3 or more tokens"}}
    write_json(args.words_json, {"tokenizers": list(encoders), "columns": ["occurrences", *encoders],
                                 "inputs": words})
    analysis = update_analysis("q2_compare", res, args.analysis_json)
    for path in write_tables(res, args.tables_dir):
        log.info("wrote %s", path)
    log.info("wrote %s and %s", plot(res), args.words_json)
    update_notes("q2", "Q2. How does it compare with existing tokenizers?", draft(analysis), args.notes)
    print(draft(analysis))
    return 0


if __name__ == "__main__":
    sys.exit(main())
