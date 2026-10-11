"""Shared helpers for the collection, cleaning, training and analysis scripts.

- iter_jsonl / write_jsonl: streaming JSON Lines I/O, UTF-8, gzip when the path ends in .gz.
- sha256_file / file_digests: hash a file in 8 MiB chunks (several algorithms in one pass).
- write_json: indented UTF-8 JSON, replaced atomically.
- machine_info: OS, CPU, cores, RAM and library versions, for reporting run times.
- Stats: per-(stage, source) counters, written to data/stats/<stage>.json.
- Timer: context manager that records wall-clock seconds into a dict.
- setup_logging: logger that writes to the console and logs/<name>.log.

Standard library only, so every script can import it.
"""

import gzip
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from types import TracebackType
from typing import IO, Any

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "logs"
STATS_DIR = REPO_ROOT / "data" / "stats"

HASH_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB

PathLike = str | os.PathLike[str]
Number = int | float


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _open_text(path: Path, mode: str) -> IO[str]:
    """Open a UTF-8 text file, through gzip when the name ends in .gz.

    newline="\\n" disables newline translation, so a JSON line written on any
    platform is read back unchanged.
    """
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="\n")
    return open(path, mode, encoding="utf-8", newline="\n")


@contextmanager
def _atomic_target(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling path; move it onto `path` only if the block succeeds.

    A crash halfway through a long write leaves the previous file (or no file),
    never a truncated one that looks complete. The temporary name keeps the
    real suffix, so .gz detection still applies to it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def iter_jsonl(path: PathLike) -> Iterator[Any]:
    """Yield one parsed object per line of a JSON Lines file. Blank lines are skipped."""
    path = Path(path)
    with _open_text(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e


def write_jsonl(path: PathLike, rows: Iterable[Any]) -> int:
    """Write rows as JSON Lines (UTF-8, ensure_ascii=False) and return the row count.

    Rows are consumed lazily, so a generator over a large corpus is fine. The
    file is replaced atomically when all rows have been written.
    """
    path = Path(path)
    n = 0
    with _atomic_target(path) as tmp, _open_text(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def file_digests(path: PathLike, algorithms: Iterable[str], chunk_size: int = HASH_CHUNK_BYTES) -> dict[str, str]:
    """Return {algorithm: hex digest} for a file, computing every digest in one read pass."""
    hashers = {name: hashlib.new(name) for name in dict.fromkeys(algorithms)}
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            for h in hashers.values():
                h.update(chunk)
    return {name: h.hexdigest() for name, h in hashers.items()}


def sha256_file(path: PathLike, chunk_size: int = HASH_CHUNK_BYTES) -> str:
    """Return the hex SHA-256 of a file, read in chunks of `chunk_size` bytes."""
    return file_digests(path, ["sha256"], chunk_size)["sha256"]


def write_json(path: PathLike, obj: Any) -> None:
    """Write obj as indented UTF-8 JSON (ensure_ascii=False), replacing the file atomically."""
    path = Path(path)
    with _atomic_target(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")


# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------

def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def parse_cpuinfo(text: str) -> tuple[str | None, int | None]:
    """Return (CPU model, physical core count) from the text of Linux /proc/cpuinfo.

    Physical cores are the distinct (physical id, core id) pairs. Returns None
    for anything the file does not state (common on ARM and in some VMs).
    """
    model: str | None = None
    cores: set[tuple[str, str]] = set()
    for block in text.strip().split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
        if model is None and fields.get("model name"):
            model = fields["model name"]
        if "physical id" in fields and "core id" in fields:
            cores.add((fields["physical id"], fields["core id"]))
    return model, (len(cores) or None)


def parse_meminfo(text: str) -> int | None:
    """Return total RAM in bytes from the text of Linux /proc/meminfo."""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            value, unit = line.split()[1:3]
            if unit.lower() != "kb":
                raise ValueError(f"unexpected MemTotal unit: {unit!r}")
            return int(value) * 1024
    return None


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def machine_info() -> dict[str, Any]:
    """Describe this machine for run reports: OS, CPU, cores, RAM, Python and library versions.

    Fields that cannot be determined are None.
    """
    system = platform.system()
    info: dict[str, Any] = {
        "os": system,
        "os_version": platform.version(),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_model": None,
        "physical_cores": None,
        "logical_cores": os.cpu_count(),
        "ram_bytes": None,
        "python_version": platform.python_version(),
        "regex_version": _package_version("regex"),
        "numpy_version": _package_version("numpy"),
    }
    if system == "Darwin":
        mac_version = platform.mac_ver()[0]
        if mac_version:
            info["os_version"] = f"macOS {mac_version}"
        info["cpu_model"] = _sysctl("machdep.cpu.brand_string")
        physical = _sysctl("hw.physicalcpu")
        info["physical_cores"] = int(physical) if physical else None
        memsize = _sysctl("hw.memsize")
        info["ram_bytes"] = int(memsize) if memsize else None
    elif system == "Linux":
        try:
            info["os_version"] = platform.freedesktop_os_release().get("PRETTY_NAME", info["os_version"])
        except OSError:
            pass
        cpuinfo = _read_text("/proc/cpuinfo")
        if cpuinfo:
            info["cpu_model"], info["physical_cores"] = parse_cpuinfo(cpuinfo)
        meminfo = _read_text("/proc/meminfo")
        if meminfo:
            info["ram_bytes"] = parse_meminfo(meminfo)
    if not info["cpu_model"]:
        info["cpu_model"] = platform.processor() or None
    return info


# ---------------------------------------------------------------------------
# Stats and timing
# ---------------------------------------------------------------------------

class Stats:
    """Named counters per (stage, source), written to <stats_dir>/<stage>.json.

    Each stage file maps source -> {counter: value}. dump() merges with the
    existing file: entries for sources this object did not touch are kept,
    and entries for sources it did touch are replaced as a whole. Re-running a
    stage for one source is therefore idempotent and never double-counts, but
    the counters for one (stage, source) must come from one complete run.

        stats = Stats()
        stats.add("normalize", "wikipedia", "docs_in")
        stats.add("normalize", "wikipedia", "bytes_in", len(raw))
        stats.dump()
    """

    def __init__(self, stats_dir: PathLike = STATS_DIR) -> None:
        self.stats_dir = Path(stats_dir)
        self._counters: dict[str, dict[str, dict[str, Number]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

    def add(self, stage: str, source: str, name: str, value: Number = 1) -> None:
        self._counters[stage][source][name] += value

    def update(self, stage: str, source: str, counts: Mapping[str, Number]) -> None:
        for name, value in counts.items():
            self.add(stage, source, name, value)

    def get(self, stage: str, source: str, name: str) -> Number:
        return self._counters.get(stage, {}).get(source, {}).get(name, 0)

    def as_dict(self) -> dict[str, dict[str, dict[str, Number]]]:
        return {
            stage: {source: dict(counts) for source, counts in sources.items()}
            for stage, sources in self._counters.items()
        }

    def path(self, stage: str) -> Path:
        return self.stats_dir / f"{stage}.json"

    def dump(self) -> list[Path]:
        """Merge into each stage's JSON file and return the paths written."""
        written = []
        for stage, sources in self.as_dict().items():
            path = self.path(stage)
            merged: dict[str, dict[str, Number]] = {}
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    merged = json.load(f)
            merged.update(sources)
            write_json(path, {src: dict(sorted(c.items())) for src, c in sorted(merged.items())})
            written.append(path)
        return written


class Timer:
    """Context manager that adds the wall-clock seconds of its block to into[name].

    Repeated blocks with the same name accumulate. The time is recorded even if
    the block raises; the exception still propagates.

        timings: dict[str, float] = {}
        with Timer("train", timings):
            ...
    """

    def __init__(self, name: str, into: dict[str, float] | None = None) -> None:
        self.name = name
        self.into = into if into is not None else {}
        self.seconds = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.seconds = time.perf_counter() - self._start
        self.into[self.name] = self.into.get(self.name, 0.0) + self.seconds


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_TAG = "_urdu_bpe_handler"


def setup_logging(name: str, log_dir: PathLike = LOG_DIR, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stderr and appends to <log_dir>/<name>.log.

    Calling it again for the same name replaces the handlers it added before,
    so messages are never duplicated.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)
    return logger
