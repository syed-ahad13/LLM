import gzip
import hashlib
import json
import logging
import platform
import time
import uuid
from importlib import metadata
from pathlib import Path

import numpy
import pytest

import common
from common import (
    HASH_CHUNK_BYTES,
    Stats,
    Timer,
    iter_jsonl,
    machine_info,
    parse_cpuinfo,
    parse_meminfo,
    setup_logging,
    sha256_file,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

ROWS = [
    {"id": 1, "text": "کتاب اُردُو", "source": "wiki"},
    {"id": 2, "text": "line one\nline two\r\nthree", "tags": ["a", "b"]},
    {"id": 3, "text": "separators:     \x85 \x1c", "n": None},
    {"id": 4, "text": "emoji 👍🏽 and ZWNJ ‌", "x": 1.5},
    {"id": 5, "text": ""},
]


def raw_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


@pytest.mark.parametrize("name", ["rows.jsonl", "rows.jsonl.gz"])
def test_jsonl_roundtrip(tmp_path: Path, name: str) -> None:
    path = tmp_path / "nested" / name
    assert write_jsonl(path, iter(ROWS)) == len(ROWS)
    assert list(iter_jsonl(path)) == ROWS
    raw = raw_bytes(path)
    assert raw.count(b"\n") == len(ROWS)  # one object per line, even with embedded separators
    assert "کتاب".encode() in raw  # ensure_ascii=False: Urdu stored as UTF-8, not \u escapes


def test_jsonl_gz_is_really_gzip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl.gz"
    write_jsonl(path, ROWS)
    assert path.read_bytes()[:2] == b"\x1f\x8b"


def test_iter_jsonl_is_lazy(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, ROWS)
    it = iter_jsonl(path)
    assert next(it) == ROWS[0]
    it.close()


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_reports_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"rows\.jsonl:2: invalid JSON"):
        list(iter_jsonl(path))


@pytest.mark.parametrize("name", ["rows.jsonl", "rows.jsonl.gz"])
def test_write_jsonl_is_atomic(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    write_jsonl(path, [{"old": True}])

    def failing_rows():
        yield {"new": 1}
        raise RuntimeError("source died")

    with pytest.raises(RuntimeError):
        write_jsonl(path, failing_rows())
    assert list(iter_jsonl(path)) == [{"old": True}]
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]  # no temp file left behind


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------

def test_sha256_known_vectors(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    abc = tmp_path / "abc"
    abc.write_bytes(b"abc")
    assert sha256_file(empty) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_file(abc) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_across_chunk_boundaries(tmp_path: Path) -> None:
    assert HASH_CHUNK_BYTES == 8 * 1024 * 1024
    data = bytes(range(256)) * (HASH_CHUNK_BYTES // 256) + b"tail"  # 8 MiB + 4 bytes: two chunks
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    assert sha256_file(path) == expected
    assert sha256_file(path, chunk_size=7) == expected


def test_file_digests_one_pass_matches_hashlib(tmp_path: Path) -> None:
    data = bytes(range(256)) * 1000
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    got = common.file_digests(path, ["md5", "sha1", "sha256", "sha256"], chunk_size=1000)
    assert got == {a: hashlib.new(a, data).hexdigest() for a in ("md5", "sha1", "sha256")}


def test_write_json_is_utf8_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "x.json"
    common.write_json(path, {"name": "مخزن", "n": 1})
    assert "مخزن" in path.read_text(encoding="utf-8") and path.read_text().endswith("}\n")
    with pytest.raises(TypeError):
        common.write_json(path, {"bad": object()})
    assert json.loads(path.read_text(encoding="utf-8")) == {"name": "مخزن", "n": 1}  # old content survives
    assert sorted(p.name for p in path.parent.iterdir()) == ["x.json"]


# ---------------------------------------------------------------------------
# machine_info
# ---------------------------------------------------------------------------

def test_machine_info_fields() -> None:
    info = machine_info()
    json.dumps(info)  # serialisable, so it can go straight into a run report
    assert info["os"] == platform.system()
    assert info["python_version"] == platform.python_version()
    assert info["regex_version"] == metadata.version("regex")
    assert info["numpy_version"] == numpy.__version__
    assert info["logical_cores"] >= 1
    if info["physical_cores"] is not None:
        assert 1 <= info["physical_cores"] <= info["logical_cores"]
    if platform.system() in ("Darwin", "Linux"):
        assert info["cpu_model"]
        assert info["ram_bytes"] > 2**30


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS sysctl path")
def test_machine_info_macos() -> None:
    info = machine_info()
    assert info["os_version"].startswith("macOS ")
    assert info["physical_cores"] >= 1


CPUINFO_X86 = """\
processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Xeon(R) CPU @ 2.20GHz
physical id\t: 0
core id\t\t: 0

processor\t: 1
model name\t: Intel(R) Xeon(R) CPU @ 2.20GHz
physical id\t: 0
core id\t\t: 0

processor\t: 2
model name\t: Intel(R) Xeon(R) CPU @ 2.20GHz
physical id\t: 0
core id\t\t: 1

processor\t: 3
model name\t: Intel(R) Xeon(R) CPU @ 2.20GHz
physical id\t: 0
core id\t\t: 1
"""

CPUINFO_ARM = """\
processor\t: 0
BogoMIPS\t: 50.00
Features\t: fp asimd

processor\t: 1
BogoMIPS\t: 50.00
"""


def test_parse_cpuinfo() -> None:
    assert parse_cpuinfo(CPUINFO_X86) == ("Intel(R) Xeon(R) CPU @ 2.20GHz", 2)
    assert parse_cpuinfo(CPUINFO_ARM) == (None, None)


def test_parse_meminfo() -> None:
    assert parse_meminfo("MemTotal:       16303428 kB\nMemFree:  1 kB\n") == 16303428 * 1024
    assert parse_meminfo("MemFree: 1 kB\n") is None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_stats_counts_and_dump(tmp_path: Path) -> None:
    stats = Stats(tmp_path)
    stats.add("normalize", "wiki", "docs_in")
    stats.add("normalize", "wiki", "docs_in")
    stats.add("normalize", "wiki", "bytes_in", 1000)
    stats.update("normalize", "news", {"docs_in": 3, "seconds": 0.5})
    stats.add("langfilter", "wiki", "lines_out", 7)
    assert stats.get("normalize", "wiki", "docs_in") == 2
    assert stats.get("normalize", "missing", "docs_in") == 0

    written = stats.dump()
    assert sorted(p.name for p in written) == ["langfilter.json", "normalize.json"]
    assert read_json(tmp_path / "normalize.json") == {
        "news": {"docs_in": 3, "seconds": 0.5},
        "wiki": {"bytes_in": 1000, "docs_in": 2},
    }
    assert read_json(tmp_path / "langfilter.json") == {"wiki": {"lines_out": 7}}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["langfilter.json", "normalize.json"]


def test_stats_merge_keeps_other_sources_and_replaces_touched_ones(tmp_path: Path) -> None:
    (tmp_path / "normalize.json").write_text(
        json.dumps({"wiki": {"docs_in": 10}, "news": {"docs_in": 5, "stale": 1}}), encoding="utf-8"
    )
    stats = Stats(tmp_path)
    stats.add("normalize", "news", "docs_in", 6)
    stats.dump()
    assert read_json(tmp_path / "normalize.json") == {"wiki": {"docs_in": 10}, "news": {"docs_in": 6}}


def test_stats_rerun_does_not_double_count(tmp_path: Path) -> None:
    for _ in range(2):
        stats = Stats(tmp_path)
        stats.add("dedup", "wiki", "docs_out", 4)
        stats.dump()
    assert read_json(tmp_path / "dedup.json") == {"wiki": {"docs_out": 4}}


def test_stats_default_dir_is_data_stats() -> None:
    assert Stats().path("split") == common.REPO_ROOT / "data" / "stats" / "split.json"


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

def test_timer_records_and_accumulates() -> None:
    timings: dict[str, float] = {}
    with Timer("step", timings) as t1:
        time.sleep(0.02)
    assert 0.02 <= t1.seconds < 5
    assert timings == {"step": t1.seconds}
    with Timer("step", timings) as t2:
        pass
    assert timings["step"] == pytest.approx(t1.seconds + t2.seconds)


def test_timer_records_on_exception() -> None:
    timings: dict[str, float] = {}
    with pytest.raises(KeyError):
        with Timer("boom", timings):
            raise KeyError("x")
    assert timings["boom"] >= 0


def test_timer_without_dict() -> None:
    with Timer("solo") as t:
        pass
    assert t.into == {"solo": t.seconds}


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

@pytest.fixture
def logger_name():
    name = f"test_{uuid.uuid4().hex[:8]}"
    yield name
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()


def test_setup_logging_writes_console_and_file(tmp_path: Path, logger_name: str, capsys) -> None:
    logger = setup_logging(logger_name, log_dir=tmp_path / "logs")
    logger.info("سلام from %s", "test")
    for h in logger.handlers:
        h.flush()
    log_file = tmp_path / "logs" / f"{logger_name}.log"
    assert "سلام from test" in log_file.read_text(encoding="utf-8")
    assert "سلام from test" in capsys.readouterr().err


def test_setup_logging_twice_does_not_duplicate(tmp_path: Path, logger_name: str) -> None:
    setup_logging(logger_name, log_dir=tmp_path)
    logger = setup_logging(logger_name, log_dir=tmp_path)
    assert len(logger.handlers) == 2
    logger.info("once")
    for h in logger.handlers:
        h.flush()
    assert (tmp_path / f"{logger_name}.log").read_text(encoding="utf-8").count("once") == 1


def test_setup_logging_default_dir_is_repo_logs() -> None:
    assert common.LOG_DIR == common.REPO_ROOT / "logs"


# ---------------------------------------------------------------------------
# ordered_imap
# ---------------------------------------------------------------------------

def _square(x: int) -> int:
    return x * x


def test_ordered_imap_in_process_and_pool() -> None:
    import multiprocessing

    from common import ordered_imap

    assert list(ordered_imap(None, _square, range(5), window=2)) == [0, 1, 4, 9, 16]
    with multiprocessing.Pool(2) as pool:
        assert list(ordered_imap(pool, _square, range(50), window=3)) == [x * x for x in range(50)]


def test_ordered_imap_reads_input_lazily() -> None:
    import multiprocessing

    from common import ordered_imap

    consumed: list[int] = []

    def items():
        for i in range(100):
            consumed.append(i)
            yield i

    with multiprocessing.Pool(2) as pool:
        it = ordered_imap(pool, _square, items(), window=4)
        assert next(it) == 0
        assert len(consumed) <= 5  # at most window items in flight, plus the one being read
