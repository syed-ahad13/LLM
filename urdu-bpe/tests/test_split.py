"""Tests for clean/split.py (Step 6b), and the integrity checks on the real split files.

The first part runs split.py on small synthetic corpora. The second part
checks data/final/{train,val,test}.txt as written by the full run; it is
skipped when data/final/manifest.json does not exist. Point it at another
run with URDU_FINAL_DIR (and URDU_CLEAN_DIR for the rerun check). The rerun
check takes a few minutes; skip it with -m "not slow". Run with -s to see the
own_scrape byte count printed.
"""

import gzip
import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from clean import budget, split
from clean.extract_common import make_doc
from common import REPO_ROOT, sha256_file, write_jsonl
from normalize import normalize

EOT = b"<|endoftext|>"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Unit tests on synthetic data
# ---------------------------------------------------------------------------

def test_cut_sizes() -> None:
    assert split.cut_sizes(1000) == (980, 10, 10)
    assert split.cut_sizes(3) == (1, 1, 1)
    assert split.cut_sizes(149) == (147, 1, 1)
    assert split.cut_sizes(150) == (146, 2, 2)  # round(1.5) half up
    assert split.cut_sizes(16950) == (16610, 170, 170)
    for n in (0, 1, 2):
        with pytest.raises(ValueError):
            split.cut_sizes(n)


def test_split_source_is_a_seeded_partition() -> None:
    ids = [f"wiki_ur:{i}" for i in range(500)]
    a = split.split_source("wiki_ur", list(reversed(ids)))
    assert a == split.split_source("wiki_ur", ids)  # sorted first, so input order does not matter
    assert [len(a[s]) for s in SPLITS] == [490, 5, 5]
    assert sorted(a["train"] + a["val"] + a["test"]) == sorted(ids)
    expected = sorted(ids)
    random.Random("1337:wiki_ur").shuffle(expected)
    assert a["train"] == expected[:490] and a["val"] == expected[490:495] and a["test"] == expected[495:]
    assert split.split_source("ufal_umc", ids) != a  # the source name is part of the seed


def test_order_split_mixes_sources() -> None:
    per_source = {s: [f"{s}:{i}" for i in range(50)] for s in ("a", "b", "c")}
    order = split.order_split("train", per_source)
    assert sorted(order) == sorted(d for ids in per_source.values() for d in ids)
    runs = 1 + sum(1 for x, y in zip(order, order[1:]) if x.split(":")[0] != y.split(":")[0])
    assert runs > 30
    assert order == split.order_split("train", dict(reversed(per_source.items())))
    assert order != split.order_split("val", per_source)


def test_line_doc_id() -> None:
    plain = json.dumps(make_doc("wiki_ur", "12", "encyclopedia", "متن"), ensure_ascii=False).encode()
    assert split.line_doc_id(plain) == "wiki_ur:12"
    odd = json.dumps(make_doc("fineweb2_urd", 'x"y\\z', "web", "متن"), ensure_ascii=False).encode()
    assert split.line_doc_id(odd) == 'fineweb2_urd:x"y\\z'
    reordered = json.dumps({"text": "متن", "doc_id": "own_scrape:1"}).encode()
    assert split.line_doc_id(reordered) == "own_scrape:1"


def test_read_docs_handles_block_edges(tmp_path: Path) -> None:
    docs = ["اردو", "", "a" * 50, "ایک\nدو", "<|endoftext"]
    path = tmp_path / "x.txt"
    path.write_bytes(b"".join(d.encode() + EOT for d in docs))
    for block in (1, 5, 13, 14, 1000):
        assert [d.decode() for d in split.read_docs(path, block=block)] == docs
    path.write_bytes(path.read_bytes() + b"tail")
    with pytest.raises(ValueError):
        list(split.read_docs(path))


def text(i: int, source: str) -> str:
    return f"{source} دستاویز نمبر {i}\n" + " ".join(["لفظ"] * (5 + i % 40))


def make_corpus(tmp_path: Path, sizes: dict[str, int]) -> Path:
    d = tmp_path / "clean"
    for source, n in sizes.items():
        write_jsonl(d / f"{source}.jsonl.gz",
                    [make_doc(source, str(i), "news" if i % 3 else "opinion/blogs", text(i, source),
                              url=f"https://example.{('pk', 'com')[i % 2]}/{i}") for i in range(n)])
    return d


def make_budget(tmp_path: Path, *extra: str) -> Path:
    out = tmp_path / "final" / "budget.json"
    argv = ["--clean-dir", str(tmp_path / "clean"), "--scan", str(tmp_path / "scan.json"), "--out", str(out),
            "--jobs", "2", *extra]
    assert budget.main(argv) == 0
    return out


def run_split(tmp_path: Path, out: str, *extra: str) -> dict[str, Any]:
    out_dir = tmp_path / out
    argv = ["--budget", str(tmp_path / "final" / "budget.json"), "--clean-dir", str(tmp_path / "clean"),
            "--out-dir", str(out_dir), "--jobs", "2", *extra]
    assert split.main(argv) == 0
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


def reference(tmp_path: Path) -> dict[str, list[tuple[str, str]]]:
    """The split computed in memory, document by document, from the rules in the brief."""
    b = json.loads((tmp_path / "final" / "budget.json").read_text(encoding="utf-8"))
    texts: dict[str, str] = {}
    per_split: dict[str, dict[str, list[str]]] = {s: {} for s in SPLITS}
    for source, row in b["sources"].items():
        kept = []
        with gzip.open(tmp_path / "clean" / f"{source}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                if int(hashlib.blake2b(doc["doc_id"].encode()).hexdigest(), 16) % 10**6 < row["keep_below"]:
                    kept.append(doc["doc_id"])
                    texts[doc["doc_id"]] = doc["text"]
        if not kept:
            continue
        ids = sorted(kept)
        random.Random(f"1337:{source}").shuffle(ids)
        k = max(1, (len(ids) + 50) // 100)
        cut = {"train": ids[:len(ids) - 2 * k], "val": ids[len(ids) - 2 * k:len(ids) - k], "test": ids[len(ids) - k:]}
        for s in SPLITS:
            per_split[s][source] = cut[s]
    out = {}
    for s in SPLITS:
        pool = [d for source in sorted(per_split[s]) for d in per_split[s][source]]
        random.Random(f"1337:order:{s}").shuffle(pool)
        out[s] = [(d, texts[d]) for d in pool]
    return out


SIZES = {"own_scrape": 120, "wiki_ur": 300, "fineweb2_urd": 2000, "ufal_umc": 700}


def test_split_matches_reference_and_is_reproducible(tmp_path: Path) -> None:
    make_corpus(tmp_path, SIZES)
    make_budget(tmp_path, "--total-mb", "0.2", "--min-total-mb", "0.05", "--fill", "fineweb2_urd", "ufal_umc")
    m1 = run_split(tmp_path, "a")
    m2 = run_split(tmp_path, "b", "--buckets", "7")
    m3 = run_split(tmp_path, "c", "--buckets", "1")
    ref = reference(tmp_path)
    for s in SPLITS:
        expected = b"".join(t.encode() + EOT for _, t in ref[s])
        assert (tmp_path / "a" / f"{s}.txt").read_bytes() == expected
        assert (tmp_path / "a" / f"{s}.ids").read_text(encoding="utf-8").split("\n")[:-1] == [d for d, _ in ref[s]]
        assert m1["splits"][s]["sha256"] == m2["splits"][s]["sha256"] == m3["splits"][s]["sha256"]
        assert m1["splits"][s]["sha256"] == hashlib.sha256(expected).hexdigest()
        assert m1["splits"][s]["bytes"] == len(expected) == (tmp_path / "a" / f"{s}.txt").stat().st_size
    ids = {s: set((tmp_path / "a" / f"{s}.ids").read_text(encoding="utf-8").split()) for s in SPLITS}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    for source, row in m1["sources"].items():
        n = row["docs"]
        assert (row["train"], row["val"], row["test"]) == split.cut_sizes(n)
        assert row["docs"] == json.loads((tmp_path / "final" / "budget.json").read_text())["sources"][source]["docs"]
    assert m1["totals"]["docs"] == sum(m1["splits"][s]["docs"] for s in SPLITS)
    assert m1["sources"]["own_scrape"]["url_domains"] == {"com": 60, "pk": 60}
    assert sum(m1["splits"]["train"]["genres"].values()) == m1["splits"]["train"]["text_bytes"]
    assert not (tmp_path / "a" / ".split_tmp").exists()


def test_source_with_no_documents_is_left_out(tmp_path: Path) -> None:
    make_corpus(tmp_path, {**SIZES, "urdu_news_1m": 500})
    make_budget(tmp_path, "--total-mb", "0.2", "--min-total-mb", "0.05",
                "--fill", "fineweb2_urd", "ufal_umc", "urdu_news_1m")
    b = json.loads((tmp_path / "final" / "budget.json").read_text(encoding="utf-8"))
    assert b["sources"]["urdu_news_1m"]["docs"] == 0 and b["sources"]["urdu_news_1m"]["keep_below"] == 0
    m = run_split(tmp_path, "a")
    assert m["sources"]["urdu_news_1m"]["docs"] == 0
    assert "urdu_news_1m" not in (tmp_path / "a" / "train.ids").read_text(encoding="utf-8")


def test_endoftext_in_text_fails(tmp_path: Path) -> None:
    make_corpus(tmp_path, {"own_scrape": 10, "wiki_ur": 10})
    write_jsonl(tmp_path / "clean" / "wiki_ur.jsonl.gz",
                [make_doc("wiki_ur", str(i), "encyclopedia", text(i, "wiki") + ("<|endoftext|>" if i == 3 else ""))
                 for i in range(10)])
    make_budget(tmp_path, "--total-mb", "1", "--min-total-mb", "0", "--fill")
    with pytest.raises(ValueError, match="endoftext"):
        run_split(tmp_path, "a")


def test_changed_input_needs_a_new_budget(tmp_path: Path) -> None:
    make_corpus(tmp_path, {"own_scrape": 10, "wiki_ur": 10})
    make_budget(tmp_path, "--total-mb", "1", "--min-total-mb", "0", "--fill")
    make_corpus(tmp_path, {"own_scrape": 11})
    with pytest.raises(SystemExit, match="rerun clean/budget.py"):
        run_split(tmp_path, "a")


# ---------------------------------------------------------------------------
# Integrity checks on the real split files
# ---------------------------------------------------------------------------

FINAL_DIR = Path(os.environ.get("URDU_FINAL_DIR", REPO_ROOT / "data" / "final"))
CLEAN_DIR = Path(os.environ.get("URDU_CLEAN_DIR", REPO_ROOT / "data" / "clean"))
MANIFEST = FINAL_DIR / "manifest.json"
NORMALIZE_SAMPLE = 2000
real = pytest.mark.skipif(not MANIFEST.exists(), reason=f"no {MANIFEST}; run clean/split.py first")


def count_occurrences(path: Path, needle: bytes, block: int = 8 * 1024 * 1024) -> int:
    """Occurrences of needle in a file, counted independently of split.read_docs."""
    n, carry = 0, b""
    with open(path, "rb") as f:
        while chunk := f.read(block):
            data = carry + chunk
            n += data.count(needle)
            carry = data[-(len(needle) - 1):]  # too short to hold a match, so none is counted twice
    return n


def iter_ids(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


@pytest.fixture(scope="module")
def corpus() -> dict[str, Any]:
    """One streaming pass over the three files, collecting what the checks need."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    total = manifest["totals"]["docs"]
    picks = set(random.Random(1337).sample(range(total), min(NORMALIZE_SAMPLE, total)))
    out: dict[str, Any] = {"manifest": manifest, "ids": {}, "hashes": {}, "docs": {}, "ids_lines": {},
                           "utf8_errors": [], "empty": 0, "sample": [], "source_text_bytes": Counter()}
    i = 0
    for s in SPLITS:
        ids_iter = iter_ids(FINAL_DIR / f"{s}.ids")
        ids: set[str] = set()
        hashes: set[bytes] = set()
        n = 0
        for raw in split.read_docs(FINAL_DIR / f"{s}.txt"):
            doc_id = next(ids_iter)
            ids.add(doc_id)
            hashes.add(hashlib.blake2b(raw, digest_size=8).digest())
            out["source_text_bytes"][doc_id.partition(":")[0]] += len(raw)
            out["empty"] += not raw.strip()
            try:
                t = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as e:
                out["utf8_errors"].append((s, doc_id, str(e)))
                t = None
            if i in picks and t is not None:
                out["sample"].append((doc_id, t))
            n += 1
            i += 1
        out["ids_lines"][s] = n + sum(1 for _ in ids_iter)
        out["ids"][s], out["hashes"][s], out["docs"][s] = ids, hashes, n
    return out


@real
def test_no_document_in_two_splits(corpus: dict[str, Any]) -> None:
    ids, hashes = corpus["ids"], corpus["hashes"]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not ids[a] & ids[b], f"doc_ids in both {a} and {b}"
        assert not hashes[a] & hashes[b], f"identical texts in both {a} and {b}"
    for s in SPLITS:
        assert len(ids[s]) == corpus["docs"][s], f"repeated doc_id in {s}"


@real
def test_endoftext_count_equals_documents(corpus: dict[str, Any]) -> None:
    m = corpus["manifest"]
    for s in SPLITS:
        n_eot = count_occurrences(FINAL_DIR / f"{s}.txt", EOT)
        assert n_eot == corpus["docs"][s] == corpus["ids_lines"][s] == m["splits"][s]["docs"]
    assert corpus["empty"] == 0


@real
def test_strict_utf8(corpus: dict[str, Any]) -> None:
    assert corpus["utf8_errors"] == []


@real
def test_documents_are_normalized(corpus: dict[str, Any]) -> None:
    assert len(corpus["sample"]) == min(NORMALIZE_SAMPLE, corpus["manifest"]["totals"]["docs"])
    bad = [doc_id for doc_id, t in corpus["sample"] if normalize(t) != t]
    assert bad == [], f"{len(bad)} of {len(corpus['sample'])} sampled documents change under normalize(): {bad[:5]}"


@real
def test_sizes(corpus: dict[str, Any]) -> None:
    total = sum((FINAL_DIR / f"{s}.txt").stat().st_size for s in SPLITS)
    own = corpus["source_text_bytes"]["own_scrape"]
    print(f"\nfinal corpus: {total:,} bytes ({total / 1e6:,.2f} MB, {total / 2**20:,.2f} MiB)")
    print(f"own_scrape text in the final corpus: {own:,} bytes ({own / 1e6:,.2f} MB)")
    assert total >= 500e6
    assert own >= 50e6
    assert total == corpus["manifest"]["totals"]["bytes"]


@real
def test_split_counts_follow_the_rule(corpus: dict[str, Any]) -> None:
    m = corpus["manifest"]
    for source, row in m["sources"].items():
        if row["docs"]:
            assert (row["train"], row["val"], row["test"]) == split.cut_sizes(row["docs"]), source
        for s in SPLITS:
            assert m["splits"][s]["sources"][source]["docs"] == row[s]
    per_source = Counter()
    for s in SPLITS:
        for doc_id in corpus["ids"][s]:
            per_source[(s, doc_id.partition(":")[0])] += 1
    for (s, source), n in per_source.items():
        assert m["splits"][s]["sources"][source]["docs"] == n


@real
def test_manifest_hashes_match_files() -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for s in SPLITS:
        assert sha256_file(FINAL_DIR / f"{s}.txt") == m["splits"][s]["sha256"]
        assert sha256_file(FINAL_DIR / f"{s}.ids") == m["splits"][s]["ids_sha256"]


@real
@pytest.mark.slow
def test_rerun_reproduces_sha256(tmp_path: Path) -> None:
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    budget_path = Path(m["budget"]["path"])
    assert sha256_file(budget_path) == m["budget"]["sha256"], "budget.json changed after the split"
    assert split.main(["--budget", str(budget_path), "--clean-dir", str(CLEAN_DIR), "--out-dir", str(tmp_path),
                       "--seed", str(m["seed"]), "--buckets", str(m["buckets"])]) == 0
    rerun = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for s in SPLITS:
        assert rerun["splits"][s]["sha256"] == m["splits"][s]["sha256"], s
        assert rerun["splits"][s]["ids_sha256"] == m["splits"][s]["ids_sha256"], s
