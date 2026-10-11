"""Resumable downloader for the published corpora listed in collect/sources.json.

    python collect/download.py --source all --dry-run
    python collect/download.py --source wiki_ur
    python collect/download.py --source fineweb2_urd --files 001_00000.parquet
    python collect/download.py --source wiki_ur --limit-bytes 20000000      # smoke test
    python collect/download.py --source urdu_news_1m --register-manual      # after a manual download

Each file streams to data/raw/<source>/<name>.part. An interrupted download
resumes with an HTTP Range request. If-Range carries the saved ETag (or
Last-Modified), so a server whose file has changed sends the whole file again
instead of a mismatched tail. A complete file is checked against the source's
published checksum, renamed, and recorded in data/raw/<source>/MANIFEST.json.
The source's entry in collect/sources.json then gets collection_date,
raw_bytes and sha256.

On HTTP 401/403, or when robots.txt disallows a URL, the script prints manual
download steps and moves on to the next source. It never logs in.
HTTP etiquette (User-Agent, robots.txt, pacing, retries) lives in collect/polite.py.
"""

import argparse
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from tqdm import tqdm  # noqa: E402

from collect.polite import RETRYABLE_EXCEPTIONS, AccessDenied, MissingContact, PoliteClient, RobotsDisallowed  # noqa: E402
from common import LOG_DIR, file_digests, setup_logging, write_json  # noqa: E402

SOURCES_PATH = ROOT / "collect" / "sources.json"
RAW_DIR = ROOT / "data" / "raw"
CHUNK_BYTES = 1 << 20
DISK_MARGIN_BYTES = 1 << 30
MAX_STREAM_RETRIES = 8


@dataclass
class FileSpec:
    url: str
    filename: str
    checksum: tuple[str, str] | None = None  # (hashlib name, hex digest) as published by the source
    size: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # provenance copied into the manifest


class ChecksumMismatch(Exception):
    pass


class DownloadFailed(Exception):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000
    raise AssertionError("unreachable")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Resolvers: turn a sources.json entry into the exact files to fetch
# ---------------------------------------------------------------------------

WIKI_BASE = "https://dumps.wikimedia.org/urwiki/"
WIKI_SHA1SUMS = WIKI_BASE + "latest/urwiki-latest-sha1sums.txt"

LINDAT_API = "https://lindat.mff.cuni.cz/repository/server/api"
UFAL_HANDLE = "11858/00-097C-0000-0023-65A9-5"
UFAL_FILE = "urdu-plain-text-corpus.gz"

MENDELEY_DATASET = "834vsxnb99"
MENDELEY_VERSION = 4

HF_REPO = "HuggingFaceFW/fineweb-2"
HF_FOLDER = "data/urd_Arab/train"

MAKHZAN_REPO = "zeerakahmed/makhzan"


def resolve_wiki_ur(client: PoliteClient, entry: dict[str, Any]) -> list[FileSpec]:
    """Pin 'latest' to the dated dump its published SHA-1 list names, so the file cannot change mid-download."""
    sums = client.get(WIKI_SHA1SUMS).text
    m = re.search(r"^([0-9a-f]{40})\s+(urwiki-(\d{8})-pages-articles\.xml\.bz2)\s*$", sums, re.M)
    if not m:
        raise DownloadFailed(f"no pages-articles line in {WIKI_SHA1SUMS}")
    sha1, name, date = m.groups()
    status = client.get(f"{WIKI_BASE}{date}/dumpstatus.json").json()
    size = status["jobs"]["articlesdump"]["files"].get(name, {}).get("size")
    return [FileSpec(f"{WIKI_BASE}{date}/{name}", name, ("sha1", sha1), size=size,
                     extra={"dump_date": date, "checksum_file": WIKI_SHA1SUMS})]


def resolve_ufal_umc(client: PoliteClient, entry: dict[str, Any]) -> list[FileSpec]:
    """Resolve the handle through LINDAT's API; robots.txt disallows /repository/handle, where hdl.handle.net points."""
    item = client.get(f"{LINDAT_API}/pid/find", params={"id": f"hdl:{UFAL_HANDLE}"}).json()
    bundles = client.get(f"{LINDAT_API}/core/items/{item['uuid']}/bundles", params={"embed": "bitstreams"}).json()
    for bundle in bundles["_embedded"]["bundles"]:
        if bundle["name"] != "ORIGINAL":
            continue
        for bs in bundle["_embedded"]["bitstreams"]["_embedded"]["bitstreams"]:
            if bs["name"] == UFAL_FILE:
                checksum = bs.get("checkSum") or {}
                algo = checksum.get("checkSumAlgorithm", "").lower().replace("-", "")
                return [FileSpec(
                    bs["_links"]["content"]["href"], UFAL_FILE,
                    (algo, checksum["value"].lower()) if algo in hashlib.algorithms_available else None,
                    size=bs.get("sizeBytes"),
                    extra={"handle": UFAL_HANDLE, "item_uuid": item["uuid"], "bitstream_uuid": bs["uuid"]},
                )]
    raise DownloadFailed(f"{UFAL_FILE} not found in LINDAT item {item['uuid']}")


def resolve_urdu_news_1m(client: PoliteClient, entry: dict[str, Any]) -> list[FileSpec]:
    """List the dataset's files through Mendeley's public API (robots.txt disallows /api/, not /public-api/)."""
    files = client.get(
        f"https://data.mendeley.com/public-api/datasets/{MENDELEY_DATASET}/files",
        params={"folder_id": "root", "version": str(MENDELEY_VERSION)},
    ).json()
    specs = []
    for f in files:
        details = f["content_details"]
        sha = details.get("sha256_hash")
        specs.append(FileSpec(
            details["download_url"], f["filename"], ("sha256", sha.lower()) if sha else None,
            size=details.get("size"),
            extra={"dataset": MENDELEY_DATASET, "version": MENDELEY_VERSION, "file_id": f["id"]},
        ))
    if not specs:
        raise DownloadFailed(f"Mendeley dataset {MENDELEY_DATASET} v{MENDELEY_VERSION} lists no files")
    return specs


def resolve_fineweb2_urd(client: PoliteClient, entry: dict[str, Any]) -> list[FileSpec]:
    """List the train parquet files at main's current commit, so every file comes from one revision."""
    revision = client.get(f"https://huggingface.co/api/datasets/{HF_REPO}/revision/main").json()["sha"]
    url: str | None = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{revision}/{HF_FOLDER}"
    listing: list[dict[str, Any]] = []
    while url:
        resp = client.get(url)
        listing.extend(resp.json())
        url = resp.links.get("next", {}).get("url")
    specs = []
    for item in sorted(listing, key=lambda e: e["path"]):
        path = item["path"]
        if item.get("type") != "file" or not path.endswith(".parquet"):
            continue
        if "_removed" in path or not path.startswith(HF_FOLDER + "/"):
            raise DownloadFailed(f"refusing {path}: only {HF_FOLDER}/ is allowed, never a *_removed subset")
        oid = (item.get("lfs") or {}).get("oid")
        specs.append(FileSpec(
            f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/{path}", path.rsplit("/", 1)[1],
            ("sha256", oid) if oid else None, size=item.get("size"),
            extra={"revision": revision, "path": path},
        ))
    return specs


def resolve_makhzan(client: PoliteClient, entry: dict[str, Any]) -> list[FileSpec]:
    """Tarball of master's current commit via api.github.com (github.com/robots.txt disallows /*/archive/)."""
    commit = client.get(f"https://api.github.com/repos/{MAKHZAN_REPO}/commits/master",
                        headers={"Accept": "application/vnd.github+json"}).json()["sha"]
    return [FileSpec(f"https://api.github.com/repos/{MAKHZAN_REPO}/tarball/{commit}",
                     f"makhzan-{commit[:12]}.tar.gz", extra={"commit": commit})]


def resolve_static(client: PoliteClient, entry: dict[str, Any]) -> list[FileSpec]:
    """Default: fetch download_urls as listed; the file name is the last URL segment."""
    return [FileSpec(u, u.rstrip("/").rsplit("/", 1)[-1]) for u in entry.get("download_urls", [])]


Resolver = Callable[[PoliteClient, dict[str, Any]], list[FileSpec]]
RESOLVERS: dict[str, Resolver] = {
    "wiki_ur": resolve_wiki_ur,
    "ufal_umc": resolve_ufal_umc,
    "urdu_news_1m": resolve_urdu_news_1m,
    "fineweb2_urd": resolve_fineweb2_urd,
    "makhzan": resolve_makhzan,
}


def select(specs: list[FileSpec], names: Sequence[str] | None, max_files: int | None) -> list[FileSpec]:
    if names:
        available = {s.filename for s in specs}
        missing = sorted(set(names) - available)
        if missing:
            raise DownloadFailed(f"--files {missing} not found; available: {sorted(available)}")
        specs = [s for s in specs if s.filename in set(names)]
    if max_files is not None:
        specs = specs[:max_files]
    return specs


# ---------------------------------------------------------------------------
# Resumable download of one file
# ---------------------------------------------------------------------------

class _Part:
    """<name>.part holds the bytes so far; <name>.part.json the URL and validators needed to resume safely."""

    def __init__(self, final: Path) -> None:
        self.path = final.with_name(final.name + ".part")
        self.meta_path = final.with_name(final.name + ".part.json")
        has_meta = self.path.exists() and self.meta_path.exists()
        self.meta: dict[str, Any] = _read_json(self.meta_path) if has_meta else {}

    @property
    def size(self) -> int:
        return self.path.stat().st_size if self.path.exists() else 0

    def save_meta(self) -> None:
        write_json(self.meta_path, self.meta)

    def reset(self) -> None:
        self.path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
        self.meta = {}

    def finish(self, final: Path) -> None:
        os.replace(self.path, final)
        self.meta_path.unlink(missing_ok=True)


def _content_range(value: str | None) -> tuple[int | None, int | None]:
    """Parse 'bytes a-b/total' or 'bytes */total' into (start, total)."""
    m = re.fullmatch(r"bytes (?:(\d+)-\d+|\*)/(\d+|\*)", (value or "").strip())
    if not m:
        return None, None
    return (int(m.group(1)) if m.group(1) else None), (int(m.group(2)) if m.group(2) != "*" else None)


def _validator(meta: dict[str, Any]) -> str | None:
    etag = meta.get("etag")
    if etag and not etag.startswith("W/"):  # If-Range needs a strong validator
        return etag
    return meta.get("last_modified")


def _check_disk(directory: Path, needed: int) -> None:
    free = shutil.disk_usage(directory).free
    if needed + DISK_MARGIN_BYTES > free:
        raise DownloadFailed(f"not enough disk space in {directory}: need {human(needed)} + 1 GB margin, have {human(free)}")


def _stream(resp: requests.Response, part: _Part, mode: str, offset: int, total: int | None,
            limit_bytes: int | None, name: str, log: logging.Logger) -> str:
    encoding = resp.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding not in ("", "identity"):
        raise DownloadFailed(f"{name}: server applied Content-Encoding {encoding!r}; the stored bytes would not match")
    step = max(total // 10, 1) if total else 256 * 1024 * 1024
    next_log = (offset // step + 1) * step
    with open(part.path, mode) as f, tqdm(total=total, initial=offset, unit="B", unit_scale=True,
                                          desc=name, disable=None, leave=False) as bar:
        for chunk in resp.iter_content(CHUNK_BYTES):
            if limit_bytes is not None:
                chunk = chunk[: limit_bytes - offset]
            f.write(chunk)
            offset += len(chunk)
            bar.update(len(chunk))
            if offset >= next_log:
                pct = f" ({100 * offset / total:.0f}%)" if total else ""
                log.info("%s: %s of %s%s", name, human(offset), human(total), pct)
                next_log = (offset // step + 1) * step
            if limit_bytes is not None and offset >= limit_bytes:
                return "limit"
    return "eof"


def _fetch(client: PoliteClient, spec: FileSpec, part: _Part, offset: int,
           limit_bytes: int | None, log: logging.Logger) -> str:
    """One request: append to the .part ('eof' or 'limit'), or discard it and ask for a fresh start ('restart')."""
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if validator := _validator(part.meta):
            headers["If-Range"] = validator
    with client.get(spec.url, headers=headers, stream=True, accept=(416,)) as resp:
        if resp.status_code == 416:
            _, remote_total = _content_range(resp.headers.get("Content-Range"))
            if offset and remote_total == offset:  # killed after the last byte, before the rename
                part.meta["total"] = offset
                part.save_meta()
                return "eof"
            log.warning("%s: HTTP 416 at offset %d (remote size %s); starting over", spec.filename, offset, remote_total)
            part.reset()
            return "restart"
        if resp.status_code == 206:
            start, remote_total = _content_range(resp.headers.get("Content-Range"))
            if start != offset:
                log.warning("%s: asked for byte %d, got Content-Range %r; starting over",
                            spec.filename, offset, resp.headers.get("Content-Range"))
                part.reset()
                return "restart"
            log.info("%s: resuming at %s (HTTP 206)", spec.filename, human(offset))
            if remote_total is not None and part.meta.get("total") != remote_total:
                part.meta["total"] = remote_total
                part.save_meta()
            mode = "ab"
        else:
            if offset:
                log.warning("%s: server sent the whole file (no Range support, or the file changed); starting over",
                            spec.filename)
            offset = 0
            length = resp.headers.get("Content-Length", "")
            part.meta = {
                "url": spec.url,
                "etag": resp.headers.get("ETag"),
                "last_modified": resp.headers.get("Last-Modified"),
                "total": int(length) if length.isdigit() else None,
                "started_utc": utc_now(),
            }
            part.save_meta()
            log.info("%s: downloading %s", spec.filename, human(part.meta["total"]))
            mode = "wb"
        total = part.meta.get("total") or spec.size
        return _stream(resp, part, mode, offset, total, limit_bytes, spec.filename, log)


def download_file(client: PoliteClient, spec: FileSpec, dest_dir: Path, *, log: logging.Logger,
                  limit_bytes: int | None = None, max_stream_retries: int = MAX_STREAM_RETRIES) -> dict[str, Any] | None:
    """Download spec into dest_dir via a .part file, resuming where it stopped.

    Returns the manifest entry, or None if --limit-bytes stopped it (the .part
    stays for the next run). Raises ChecksumMismatch (the .part is deleted),
    DownloadFailed, AccessDenied or RobotsDisallowed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / spec.filename
    part = _Part(final)
    if part.path.exists() and part.meta.get("url") != spec.url:
        log.warning("%s: existing .part has no resume metadata for this URL; starting over", spec.filename)
        part.reset()
    failures = 0
    while True:
        offset = part.size
        total = part.meta.get("total") or spec.size
        if limit_bytes is not None and offset >= limit_bytes:
            log.info("%s: stopped at --limit-bytes after %s; %s kept for resume", spec.filename, human(offset), part.path.name)
            return None
        if total is not None and offset >= total:
            break
        if bounds := [x for x in (total, limit_bytes) if x is not None]:
            _check_disk(dest_dir, min(bounds) - offset)
        try:
            outcome = _fetch(client, spec, part, offset, limit_bytes, log)
        except RETRYABLE_EXCEPTIONS as e:
            outcome, error = "error", f"{type(e).__name__}: {e}"
        if outcome in ("restart", "limit"):
            continue
        if outcome == "eof":
            total = part.meta.get("total") or spec.size
            if total is None or part.size >= total:
                break
            error = f"connection closed at {human(part.size)} of {human(total)}"
        failures = 0 if part.size > offset else failures + 1
        if failures > max_stream_retries:
            raise DownloadFailed(f"{spec.filename}: giving up after {max_stream_retries} retries without progress: {error}")
        client.backoff(failures, f"{spec.filename}: {error}")

    size = part.size
    if spec.size is not None and size != spec.size:
        part.reset()
        raise ChecksumMismatch(f"{spec.filename}: {size} bytes, source lists {spec.size}; .part deleted")
    algorithms = ["sha256"] + ([spec.checksum[0]] if spec.checksum else [])
    digests = file_digests(part.path, algorithms)
    published = None
    if spec.checksum:
        algo, expected = spec.checksum
        if digests[algo] != expected.lower():
            part.reset()
            raise ChecksumMismatch(f"{spec.filename}: {algo} {digests[algo]} != published {expected}; .part deleted")
        published = {"algorithm": algo, "value": expected.lower(), "verified": True}
        log.info("%s: %s matches the published checksum", spec.filename, algo)
    else:
        log.info("%s: no published checksum to verify against", spec.filename)
    entry = {
        "url": spec.url,
        "bytes": size,
        "sha256": digests["sha256"],
        "published_checksum": published,
        "started_utc": part.meta.get("started_utc"),
        "downloaded_utc": utc_now(),
        "etag": part.meta.get("etag"),
        "last_modified": part.meta.get("last_modified"),
        "obtained": "download",
        **spec.extra,
    }
    part.finish(final)
    log.info("%s: done, %s, sha256 %s", spec.filename, human(size), digests["sha256"])
    return entry


def describe_local_file(path: Path, spec: FileSpec | None, log: logging.Logger) -> dict[str, Any]:
    """Hash a file that is already in place (placed by hand), verifying a published checksum if we know one."""
    algorithms = ["sha256"] + ([spec.checksum[0]] if spec and spec.checksum else [])
    digests = file_digests(path, algorithms)
    published = None
    if spec and spec.checksum:
        algo, expected = spec.checksum
        if digests[algo] != expected.lower():
            raise ChecksumMismatch(f"{path.name}: {algo} {digests[algo]} != published {expected} (file left in place)")
        published = {"algorithm": algo, "value": expected.lower(), "verified": True}
    mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    log.info("%s: recorded %s, sha256 %s%s", path.name, human(path.stat().st_size), digests["sha256"],
             f", {published['algorithm']} verified" if published else "")
    return {
        "url": spec.url if spec else None,
        "bytes": path.stat().st_size,
        "sha256": digests["sha256"],
        "published_checksum": published,
        "downloaded_utc": mtime,  # file modification time: when it was saved
        "obtained": "manual",
        **(spec.extra if spec else {}),
    }


# ---------------------------------------------------------------------------
# Manifest and sources.json
# ---------------------------------------------------------------------------

def load_manifest(dest: Path, source_id: str) -> dict[str, Any]:
    path = dest / "MANIFEST.json"
    return _read_json(path) if path.exists() else {"source": source_id, "files": {}}


def save_manifest(dest: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_utc"] = utc_now()
    manifest["files"] = dict(sorted(manifest["files"].items()))
    write_json(dest / "MANIFEST.json", manifest)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive lock, so downloads running in parallel tmux windows can all update sources.json."""
    with open(path.with_name(path.name + ".lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def combined_sha256(files: dict[str, dict[str, Any]]) -> str:
    """One file: its SHA-256. Several: SHA-256 of the `sha256sum`-style listing sorted by file name."""
    if len(files) == 1:
        return next(iter(files.values()))["sha256"]
    listing = "".join(f"{f['sha256']}  {name}\n" for name, f in sorted(files.items()))
    return hashlib.sha256(listing.encode()).hexdigest()


def update_sources(sources_path: Path, source_id: str, manifest: dict[str, Any]) -> None:
    files = manifest["files"]
    with _locked(sources_path):
        data = _read_json(sources_path)
        entry = next(s for s in data["sources"] if s["id"] == source_id)
        entry["collection_date"] = max(f["downloaded_utc"] for f in files.values())[:10]
        entry["raw_bytes"] = sum(f["bytes"] for f in files.values())
        entry["sha256"] = combined_sha256(files)
        write_json(sources_path, data)


# ---------------------------------------------------------------------------
# Per-source driver
# ---------------------------------------------------------------------------

def manual_steps(entry: dict[str, Any], reason: str, specs: list[FileSpec], dest: Path) -> str:
    wanted = [f"{s.filename}  ({s.url})" for s in specs] or list(entry.get("download_urls", [])) or ["see the landing page"]
    try:
        shown = dest.relative_to(ROOT)
    except ValueError:
        shown = dest
    return "\n".join([
        f"MANUAL DOWNLOAD NEEDED for {entry['id']}: {reason}",
        "  This script never logs in and never fetches what robots.txt disallows.",
        f"  1. Open {entry['landing_url']} in a browser; accept the licence if asked ({entry['licence_url']}).",
        "  2. Download:",
        *[f"       {w}" for w in wanted],
        f"  3. Put the file(s) in {shown}/",
        f"  4. Run: python collect/download.py --source {entry['id']} --register-manual",
        "     It hashes the files, checks published checksums when known, and fills MANIFEST.json and sources.json.",
    ])


def _resolve(client: PoliteClient, entry: dict[str, Any], args: argparse.Namespace) -> list[FileSpec]:
    specs = RESOLVERS.get(entry["id"], resolve_static)(client, entry)
    return select(specs, args.files, args.max_files)


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_plan(client: PoliteClient, entry: dict[str, Any], specs: list[FileSpec], dest: Path, note: str) -> None:
    print(f"[{entry['id']}] {entry['name']}{note}")
    for s in specs:
        final = dest / s.filename
        part = final.with_name(final.name + ".part")
        state = ("done" if final.exists() else
                 f"partial {human(part.stat().st_size)}" if part.exists() else "not started")
        checksum = f"{s.checksum[0]} {s.checksum[1][:12]}..." if s.checksum else "no published checksum"
        robots = "allowed" if client.allowed(s.url) else "DISALLOWED -> manual download"
        print(f"  GET {s.url}")
        print(f"   -> {_shown(final)}  [{human(s.size)}; {checksum}; robots.txt {robots}; {state}]")


def process_source(client: PoliteClient, entry: dict[str, Any], args: argparse.Namespace, log: logging.Logger) -> str:
    """Download (or with --dry-run, list) one source. Returns ok, partial, manual, failed or skipped."""
    sid = entry["id"]
    dest = args.raw_dir / sid
    note = "" if entry.get("enabled", True) else "  (disabled: skipped by --source all)"
    if entry["access_method"] == "scrape":
        if args.dry_run:
            print(f"[{sid}] {entry['name']}\n  nothing to download: collected by the scraper (Step 2)")
        return "skipped"
    try:
        specs = _resolve(client, entry, args)
    except (RobotsDisallowed, AccessDenied) as e:
        log.warning(manual_steps(entry, str(e), [], dest))
        return "manual"
    if args.dry_run:
        print_plan(client, entry, specs, dest, note)
        return "ok"

    manifest = load_manifest(dest, sid)
    status = "ok"
    for spec in specs:
        final = dest / spec.filename
        recorded = manifest["files"].get(spec.filename)
        if final.exists() and recorded and recorded.get("bytes") == final.stat().st_size:
            log.info("%s: already downloaded (%s); skipping", spec.filename, human(recorded["bytes"]))
            continue
        try:
            if final.exists():
                log.info("%s: present but not in MANIFEST.json; verifying and recording it", spec.filename)
                file_entry = describe_local_file(final, spec, log)
            else:
                file_entry = download_file(client, spec, dest, log=log, limit_bytes=args.limit_bytes)
        except (RobotsDisallowed, AccessDenied) as e:
            log.warning(manual_steps(entry, str(e), specs, dest))
            return "manual"
        except (ChecksumMismatch, DownloadFailed, requests.RequestException) as e:
            log.error("%s: %s", spec.filename, e)
            status = "failed"
            continue
        if file_entry is None:
            status = "partial" if status == "ok" else status
            continue
        manifest["files"][spec.filename] = file_entry
        save_manifest(dest, manifest)
    if manifest["files"]:
        update_sources(args.sources, sid, manifest)
    return status


def register_manual(client: PoliteClient, entry: dict[str, Any], args: argparse.Namespace, log: logging.Logger) -> str:
    """Record files placed in data/raw/<source>/ by hand."""
    sid = entry["id"]
    dest = args.raw_dir / sid
    files = sorted(
        p for p in dest.glob("*")
        if p.is_file() and not p.name.startswith(".") and p.name != "MANIFEST.json"
        and not p.name.endswith((".part", ".part.json"))
    ) if dest.is_dir() else []
    if not files:
        log.error("no files to register in %s", _shown(dest))
        return "failed"
    try:
        known = {s.filename: s for s in RESOLVERS.get(sid, resolve_static)(client, entry)}
    except Exception as e:  # metadata only; registering still works without it
        log.warning("could not fetch published checksums (%s); recording without verification", e)
        known = {}
    manifest = load_manifest(dest, sid)
    status = "ok"
    for path in files:
        try:
            manifest["files"][path.name] = describe_local_file(path, known.get(path.name), log)
        except ChecksumMismatch as e:
            log.error("%s", e)
            status = "failed"
    if manifest["files"]:
        save_manifest(dest, manifest)
        update_sources(args.sources, sid, manifest)
    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="source id from collect/sources.json, or 'all' (enabled sources)")
    p.add_argument("--max-files", type=int, metavar="N", help="only the first N files (sorted by name)")
    p.add_argument("--files", nargs="+", metavar="NAME", help="only these file names, e.g. 001_00000.parquet")
    p.add_argument("--limit-bytes", type=int, metavar="N",
                   help="stop each file after N bytes and keep the .part (smoke tests; a later run resumes)")
    p.add_argument("--dry-run", action="store_true", help="list URLs and target paths; download nothing")
    p.add_argument("--register-manual", action="store_true",
                   help="hash and record files placed in data/raw/<source>/ by hand")
    p.add_argument("--min-interval", type=float, default=1.0, metavar="SEC",
                   help="minimum seconds between requests to one host (default 1)")
    p.add_argument("--sources", type=Path, default=SOURCES_PATH, help=argparse.SUPPRESS)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR, help=argparse.SUPPRESS)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def run(args: argparse.Namespace, client: PoliteClient) -> int:
    sources = _read_json(args.sources)["sources"]
    by_id = {s["id"]: s for s in sources}
    if args.source == "all" and args.register_manual:
        print("--register-manual needs one --source id", file=sys.stderr)
        return 2
    if args.source == "all":
        ids = [s["id"] for s in sources if args.dry_run or s.get("enabled", True)]
    elif args.source in by_id:
        ids = [args.source]
    else:
        print(f"unknown source {args.source!r}; choose from: all, {', '.join(by_id)}", file=sys.stderr)
        return 2
    results: dict[str, str] = {}
    for sid in ids:
        entry = by_id[sid]
        log = setup_logging(f"download_{sid}", args.log_dir)
        client.log = log
        if not entry.get("enabled", True) and args.source != "all":
            log.warning("%s is disabled in sources.json; running it because it was named explicitly", sid)
        try:
            action = register_manual if args.register_manual else process_source
            results[sid] = action(client, entry, args, log)
        except Exception:
            log.exception("%s failed", sid)
            results[sid] = "failed"
    if not args.dry_run:
        print("summary: " + ", ".join(f"{sid}={status}" for sid, status in results.items()))
    return 1 if any(r in ("failed", "manual") for r in results.values()) else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        client = PoliteClient(min_interval=args.min_interval)
    except MissingContact as e:
        print(e, file=sys.stderr)
        return 2
    return run(args, client)


if __name__ == "__main__":
    sys.exit(main())
