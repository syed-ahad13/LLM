"""Tests for encode_corpus.py (Step 10) on a small synthetic corpus and a small tokenizer.

Parallel and serial encoding, cache on and off, and every block size give the
same .bin, equal to encoding the whole text at once; the verification passes
and catches a changed ID, an ID outside the vocabulary, a truncated file and a
document count that disagrees with the manifest; the files load with
np.memmap(dtype=np.uint16); meta.json records both splits.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import encode_corpus
from bpe import ENDOFTEXT, BPETokenizer, count_words
from common import sha256_file
from tests.urdu_text import urdu_text

EXTRA_DOCS = [
    "Hello world, it's 2026! 😀🇵🇰\n\ttabs  and   spaces ",
    "اُردُو صورت‌حال\r\nCRLF line\n\n\n",
    "a partial <|endoftext separator stays text",
    "١٢٣ 4567890 ۔۔۔",
]


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("encode")
    tok = BPETokenizer()
    tok.train(count_words(urdu_text(80_000)), 400)
    tok_path = tok.save(root / "tok" / "t.json")
    final = root / "final"
    final.mkdir()
    train_text = urdu_text(150_000, seed=5) + "".join(d + ENDOFTEXT for d in EXTRA_DOCS)
    (final / "train.txt").write_text(train_text, encoding="utf-8")
    (final / "val.txt").write_text(urdu_text(30_000, seed=6), encoding="utf-8")
    return SimpleNamespace(root=root, tok=BPETokenizer.load(tok_path), tok_path=tok_path, final=final,
                           train=final / "train.txt", val=final / "val.txt", train_text=train_text)


def expected_ids(env: SimpleNamespace, text: str) -> np.ndarray:
    return np.asarray(BPETokenizer.load(env.tok_path).encode(text, allowed_special="all"), dtype="<u2")


def cli(env: SimpleNamespace, *extra: str) -> list[str]:
    return ["--tokenizer", str(env.tok_path), "--manifest", str(env.root / "no_manifest.json"),
            "--samples-dir", str(env.root / "samples"), "--block-mb", "0.01", "--verify-chunk", "7", *extra]


def test_parallel_serial_and_block_size_give_the_same_bin(env: SimpleNamespace, tmp_path: Path) -> None:
    serial = encode_corpus.encode_file(env.tok_path, env.train, tmp_path / "serial.bin", 1, block_bytes=8192)
    parallel = encode_corpus.encode_file(env.tok_path, env.train, tmp_path / "parallel.bin", 2, block_bytes=8192)
    one_block = encode_corpus.encode_file(env.tok_path, env.train, tmp_path / "one.bin", 1)
    assert serial["blocks"] > 10 and one_block["blocks"] == 1
    data = (tmp_path / "serial.bin").read_bytes()
    assert data == (tmp_path / "parallel.bin").read_bytes() == (tmp_path / "one.bin").read_bytes()
    assert data == expected_ids(env, env.train_text).tobytes()
    assert serial["tokens"] == parallel["tokens"] == len(data) // 2
    assert len(parallel["cache"]["per_worker"]) == 2
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_cache_on_and_off_give_the_same_bin(env: SimpleNamespace, tmp_path: Path) -> None:
    on = encode_corpus.encode_file(env.tok_path, env.train, tmp_path / "on.bin", 1, block_bytes=8192)
    off = encode_corpus.encode_file(env.tok_path, env.train, tmp_path / "off.bin", 1, block_bytes=8192,
                                    use_cache=False)
    assert (tmp_path / "on.bin").read_bytes() == (tmp_path / "off.bin").read_bytes()
    assert on["cache"]["hit_rate"] > 0.9 and off["cache"]["hit_rate"] == 0.0
    assert on["cache"]["per_worker"][0]["cached_words"] > 0


def test_report_counts(env: SimpleNamespace, tmp_path: Path) -> None:
    rep = encode_corpus.encode_file(env.tok_path, env.train, tmp_path / "t.bin", 2, block_bytes=8192)
    docs = env.train_text.split(ENDOFTEXT)[:-1]
    assert rep["docs"] == len(docs) == env.train_text.count(ENDOFTEXT)
    assert rep["words"] == sum(len(d.split()) for d in docs)
    assert rep["input_bytes"] == env.train.stat().st_size
    assert rep["bytes_per_token"] == round(rep["input_bytes"] / rep["tokens"], 4)
    assert rep["tokens_per_word"] == round(rep["tokens"] / rep["words"], 4)
    assert rep["seconds"] > 0 and rep["mb_per_s"] > 0
    assert "tokens per word" in encode_corpus.report_text(rep)


def test_bin_loads_with_memmap(env: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "t.bin"
    encode_corpus.encode_file(env.tok_path, env.train, out, 2, block_bytes=8192)
    ids = np.memmap(out, dtype=np.uint16, mode="r")
    assert np.array_equal(ids, expected_ids(env, env.train_text))
    assert int(ids.max()) < 400 and int(np.count_nonzero(ids == 256)) == env.train_text.count(ENDOFTEXT)
    assert out.stat().st_size == 2 * len(ids)  # raw IDs, no .npy header
    assert env.tok.decode(ids.tolist()) == env.train_text


def test_verify_passes_and_detects_corruption(env: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "t.bin"
    encode_corpus.encode_file(env.tok_path, env.train, out, 1)
    res = encode_corpus.verify(env.tok_path, env.train, out, None, chunk_ids=7)  # chunk edges inside characters
    assert res["passed"], res["failures"]
    assert res["decoded_sha256"] == res["txt_sha256"] == sha256_file(env.train)
    assert res["eot_ids"] == res["eot_in_txt"] == env.train_text.count(ENDOFTEXT)
    assert "VERIFIED" in encode_corpus.verify_text(res)
    good = out.read_bytes()

    ids = np.frombuffer(good, dtype="<u2").copy()
    i = int(np.flatnonzero(ids > 300)[0])
    ids[i] -= 1  # another valid token
    out.write_bytes(ids.tobytes())
    res = encode_corpus.verify(env.tok_path, env.train, out, None, chunk_ids=7)
    assert not res["passed"] and any("sha256" in f for f in res["failures"])

    corrupt = bytearray(good)
    corrupt[11] ^= 0x80  # high byte of an ID: now far outside the vocabulary
    out.write_bytes(bytes(corrupt))
    res = encode_corpus.verify(env.tok_path, env.train, out, None)
    assert not res["passed"] and any("not in the vocabulary" in f for f in res["failures"])
    assert "FAILED" in encode_corpus.verify_text(res)

    out.write_bytes(good[:-1])
    res = encode_corpus.verify(env.tok_path, env.train, out, None)
    assert not res["passed"] and any("whole number" in f for f in res["failures"])

    eot_dropped = np.frombuffer(good, dtype="<u2")
    out.write_bytes(eot_dropped[eot_dropped != 256].tobytes())
    res = encode_corpus.verify(env.tok_path, env.train, out, None)
    assert not res["passed"] and res["eot_ids"] == 0 and len(res["failures"]) == 2

    out.write_bytes(bytes(corrupt))
    assert encode_corpus.main(cli(env, "--verify", "--input", str(env.train), "--out", str(out))) == 1


def test_manifest_document_count(env: SimpleNamespace, tmp_path: Path) -> None:
    out = tmp_path / "t.bin"
    encode_corpus.encode_file(env.tok_path, env.train, out, 1)
    docs = env.train_text.count(ENDOFTEXT)
    manifest = tmp_path / "manifest.json"
    splits = {"val": {"sha256": sha256_file(env.val), "docs": 1},
              "train": {"sha256": sha256_file(env.train), "docs": docs}}
    manifest.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    res = encode_corpus.verify(env.tok_path, env.train, out, manifest)
    assert res["passed"] and res["manifest"]["split"] == "train" and res["manifest"]["docs"] == docs
    splits["train"]["docs"] = docs + 1
    manifest.write_text(json.dumps({"splits": splits}), encoding="utf-8")
    res = encode_corpus.verify(env.tok_path, env.train, out, manifest)
    assert not res["passed"] and any("documents in train" in f for f in res["failures"])


def test_cli_encodes_train_and_val_and_writes_meta(env: SimpleNamespace, capsys: pytest.CaptureFixture[str]) -> None:
    for split in ("train", "val"):
        args = cli(env, "--input", str(env.final / f"{split}.txt"), "--out", str(env.final / f"{split}.bin"),
                   "--workers", "2")
        assert encode_corpus.main(args) == 0
    err = capsys.readouterr().err  # the reports go to the log (stderr and logs/encode_corpus.log)
    assert err.count("VERIFIED") == 2 and "cache hit rate" in err
    meta = json.loads((env.final / "meta.json").read_text(encoding="utf-8"))
    assert {k: meta[k] for k in ("vocab_size", "eot_id", "dtype", "byteorder")} == {
        "vocab_size": 400, "eot_id": 256, "dtype": "uint16", "byteorder": "little"}
    assert meta["tokenizer_sha256"] == sha256_file(env.tok_path) and meta["pattern"] == env.tok.pattern
    assert list(meta).index("train_tokens") < list(meta).index("val_tokens") < list(meta).index("pattern")
    for split in ("train", "val"):
        assert meta[f"{split}_tokens"] == (env.final / f"{split}.bin").stat().st_size // 2
        entry = meta["splits"][split]
        assert entry["verified"] and entry["txt_sha256"] == sha256_file(env.final / f"{split}.txt")
        assert entry["bin_sha256"] == sha256_file(env.final / f"{split}.bin")
    assert encode_corpus.main(cli(env, "--verify", "--input", str(env.val), "--out", str(env.final / "val.bin"))) == 0


def test_meta_from_another_tokenizer_is_dropped(env: SimpleNamespace, tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"tokenizer_sha256": "old", "val_tokens": 5, "splits": {"val": {"tokens": 5}}}))
    encode_corpus.update_meta(meta, env.tok_path, "train", {"tokens": 7})
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["train_tokens"] == 7 and "val_tokens" not in data and set(data["splits"]) == {"train"}


def test_sample_run(env: SimpleNamespace) -> None:
    assert encode_corpus.main(cli(env, "--input", str(env.train), "--sample-mb", "0.05", "--workers", "1")) == 0
    sample = env.root / "samples" / "train_0.05mb.txt"
    assert sample.exists() and env.train.read_bytes().startswith(sample.read_bytes())
    ids = np.memmap(sample.with_suffix(".bin"), dtype=np.uint16, mode="r")
    assert env.tok.decode_bytes(ids.tolist()) == sample.read_bytes()
    assert "train_0.05mb_tokens" in json.loads((env.root / "samples" / "meta.json").read_text(encoding="utf-8"))


def test_refusals(env: SimpleNamespace, tmp_path: Path) -> None:
    test_txt = tmp_path / "test.txt"
    test_txt.write_text(urdu_text(2000), encoding="utf-8")
    with pytest.raises(SystemExit, match="test.txt is not encoded"):
        encode_corpus.main(cli(env, "--input", str(test_txt)))
    with pytest.raises(SystemExit, match="must be a .bin"):
        encode_corpus.main(cli(env, "--input", str(env.train), "--out", str(tmp_path / "train.npy")))
    with pytest.raises(SystemExit, match="no .* to verify"):
        encode_corpus.main(cli(env, "--verify", "--input", str(env.train), "--out", str(tmp_path / "none.bin")))
