import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from collect import download
from collect.download import (
    ChecksumMismatch,
    DownloadFailed,
    FileSpec,
    combined_sha256,
    download_file,
    parse_args,
    run,
    select,
)
from collect.polite import AccessDenied, PoliteClient, RobotsDisallowed

LOG = logging.getLogger("test_download")
DATA = bytes(range(256)) * 1200  # 307,200 bytes
OTHER = b"changed upstream " * 20000


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def spec_for(site, path: str, data: bytes | None = None, algo: str = "sha256", **kw: Any) -> FileSpec:
    digest = hashlib.new(algo, data if data is not None else site.files[path]).hexdigest()
    return FileSpec(site.url(path), path.rsplit("/", 1)[1], (algo, digest), **kw)


# ---------------------------------------------------------------------------
# download_file: resume logic and checksums
# ---------------------------------------------------------------------------

def test_fresh_download_verifies_checksum(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    entry = download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)
    final = tmp_path / "f.bin"
    assert final.read_bytes() == DATA
    assert not (tmp_path / "f.bin.part").exists() and not (tmp_path / "f.bin.part.json").exists()
    assert entry["bytes"] == len(DATA) and entry["sha256"] == sha256(DATA)
    assert entry["published_checksum"] == {"algorithm": "sha256", "value": sha256(DATA), "verified": True}
    assert entry["etag"] == '"v1"' and entry["last_modified"]
    assert datetime.fromisoformat(entry["downloaded_utc"]).tzinfo == timezone.utc
    [req] = site.file_requests("/f.bin")
    assert req["range"] is None and req["accept_encoding"] == "identity"


@pytest.mark.parametrize("algo", ["md5", "sha1", "sha256"])
def test_published_checksum_algorithms(site, client, tmp_path: Path, algo: str) -> None:
    site.files["/f.bin"] = DATA
    entry = download_file(client, spec_for(site, "/f.bin", algo=algo), tmp_path, log=LOG)
    assert entry["published_checksum"]["algorithm"] == algo


def test_stop_then_resume_uses_range_and_if_range(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    spec = spec_for(site, "/f.bin")
    assert download_file(client, spec, tmp_path, log=LOG, limit_bytes=100_000) is None
    part = tmp_path / "f.bin.part"
    assert part.stat().st_size == 100_000
    meta = json.loads((tmp_path / "f.bin.part.json").read_text())
    assert meta["url"] == spec.url and meta["etag"] == '"v1"' and meta["total"] == len(DATA)

    entry = download_file(client, spec, tmp_path, log=LOG)  # a fresh process would do the same
    assert (tmp_path / "f.bin").read_bytes() == DATA
    assert entry["sha256"] == sha256(DATA)
    resume = site.file_requests("/f.bin")[-1]
    assert resume["range"] == "bytes=100000-" and resume["if_range"] == '"v1"'


def test_connection_drop_mid_body_is_resumed(site, client, sleeps, tmp_path: Path) -> None:
    big = DATA * 10  # 3 MB: the drop lands after the first full 1 MiB chunk was written
    site.files["/f.bin"] = big
    site.drop_after["/f.bin"] = [3 * download.CHUNK_BYTES // 2]
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == big
    first, second = site.file_requests("/f.bin")
    # The half-received second chunk is discarded by urllib3, never written, so the resume starts at 1 MiB.
    assert first["range"] is None and second["range"] == f"bytes={download.CHUNK_BYTES}-"
    assert len(sleeps) == 1  # one backoff before resuming


def test_retry_budget_resets_whenever_bytes_arrive(site, client, tmp_path: Path) -> None:
    big = DATA * 20  # 6 MB
    site.files["/f.bin"] = big
    site.drop_after["/f.bin"] = [3 * download.CHUNK_BYTES // 2] * 4  # four drops, each after one chunk of progress
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG, max_stream_retries=1)
    assert (tmp_path / "f.bin").read_bytes() == big
    ranges = [r["range"] for r in site.file_requests("/f.bin")]
    assert ranges == [None] + [f"bytes={k * download.CHUNK_BYTES}-" for k in range(1, 5)]


def test_206_from_the_wrong_offset_is_not_appended(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    spec = spec_for(site, "/f.bin")
    download_file(client, spec, tmp_path, log=LOG, limit_bytes=100_000)
    site.range_from_zero.add("/f.bin")
    download_file(client, spec, tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == DATA


def test_server_ignoring_range_restarts_from_zero(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    spec = spec_for(site, "/f.bin")
    download_file(client, spec, tmp_path, log=LOG, limit_bytes=100_000)
    site.ignore_range.add("/f.bin")
    download_file(client, spec, tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == DATA  # rewritten, not appended


def test_changed_upstream_file_is_fetched_whole(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG, limit_bytes=100_000)
    site.files["/f.bin"], site.etags["/f.bin"] = OTHER, '"v2"'
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == OTHER  # If-Range failed, so the server sent all of v2
    assert site.file_requests("/f.bin")[-1]["if_range"] == '"v1"'


def test_complete_part_is_finished_after_416(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    spec = spec_for(site, "/f.bin")
    (tmp_path / "f.bin.part").write_bytes(DATA)  # killed after the last byte, before the rename
    (tmp_path / "f.bin.part.json").write_text(json.dumps({"url": spec.url, "etag": '"v1"', "total": None}))
    download_file(client, spec, tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == DATA
    assert site.file_requests("/f.bin")[-1]["range"] == f"bytes={len(DATA)}-"


def test_part_without_metadata_starts_over(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    (tmp_path / "f.bin.part").write_bytes(b"garbage of unknown origin")
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == DATA


def test_checksum_mismatch_deletes_part(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    spec = spec_for(site, "/f.bin", data=b"something else")
    with pytest.raises(ChecksumMismatch):
        download_file(client, spec, tmp_path, log=LOG)
    assert list(tmp_path.iterdir()) == []


def test_size_mismatch_is_rejected(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    with pytest.raises(ChecksumMismatch):
        download_file(client, FileSpec(site.url("/f.bin"), "f.bin", size=len(DATA) + 1), tmp_path, log=LOG)
    assert not (tmp_path / "f.bin").exists()


def test_gives_up_after_retries_without_progress(site, client, sleeps, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    site.hang_up_always.add("/f.bin")
    with pytest.raises(DownloadFailed):
        download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG, max_stream_retries=2)
    assert len(site.file_requests("/f.bin")) == 3 and len(sleeps) == 2


# ---------------------------------------------------------------------------
# HTTP etiquette (collect/polite.py) as the downloader sees it
# ---------------------------------------------------------------------------

def test_retries_5xx_with_backoff_and_retry_after(site, client, sleeps, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    site.status_queue["/f.bin"] = [503, 503]
    site.retry_after = "7"
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)
    assert (tmp_path / "f.bin").read_bytes() == DATA
    assert len(sleeps) == 2 and all(s >= 7 for s in sleeps)


def test_403_raises_access_denied(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    site.status_queue["/f.bin"] = [403]
    with pytest.raises(AccessDenied):
        download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)


def test_robots_disallow_means_no_request(site, client, tmp_path: Path) -> None:
    site.files["/private/f.bin"] = DATA
    with pytest.raises(RobotsDisallowed):
        download_file(client, spec_for(site, "/private/f.bin"), tmp_path, log=LOG)
    assert site.file_requests("/private/f.bin") == []


def test_unreachable_robots_means_disallow(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    site.robots_status = 500
    with pytest.raises(RobotsDisallowed):
        download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)


def test_missing_robots_allows_and_user_agent_is_sent(site, client, tmp_path: Path) -> None:
    site.files["/f.bin"] = DATA
    site.robots = None
    download_file(client, spec_for(site, "/f.bin"), tmp_path, log=LOG)
    assert {r["user_agent"] for r in site.requests} == {client.user_agent}


def test_requests_to_one_host_are_spaced(site, sleeps) -> None:
    now = [100.0]
    c = PoliteClient("TestBot/1.0", min_interval=5, sleep=lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)),
                     clock=lambda: now[0])
    c.session.trust_env = False
    site.files["/a"] = b"a"
    c.get(site.url("/a")).close()  # robots.txt first, then /a
    c.get(site.url("/a")).close()
    assert sleeps == [5, 5]


# ---------------------------------------------------------------------------
# run(): manifest, sources.json, manual steps, dry run
# ---------------------------------------------------------------------------

def make_sources(path: Path, site) -> None:
    def entry(sid: str, urls: list[str], **kw: Any) -> dict[str, Any]:
        return {"id": sid, "name": sid, "genre": "test", "enabled": True, "landing_url": site.url("/"),
                "download_urls": urls, "licence": "test", "licence_url": site.url("/licence"),
                "terms_notes": "", "access_method": "dump", "collected_by_us": False,
                "collection_date": None, "raw_bytes": None, "sha256": None, "notes": "", **kw}
    data = {"_about": "test", "sources": [
        entry("good", [site.url("/d/a.bin"), site.url("/d/b.bin")]),
        entry("blocked", [site.url("/private/c.bin")]),
        entry("denied", [site.url("/d/denied.bin")]),
        entry("off", [site.url("/d/off.bin")], enabled=False),
        entry("scraped", [], access_method="scrape", collected_by_us=True),
    ]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.fixture
def project(site, tmp_path: Path) -> SimpleNamespace:
    site.files.update({"/d/a.bin": DATA, "/d/b.bin": OTHER, "/private/c.bin": DATA, "/d/off.bin": DATA})
    site.status_queue["/d/denied.bin"] = [403]
    sources = tmp_path / "sources.json"
    make_sources(sources, site)
    return SimpleNamespace(sources=sources, raw=tmp_path / "raw", logs=tmp_path / "logs")


def args_for(project: SimpleNamespace, *extra: str):
    return parse_args(["--sources", str(project.sources), "--raw-dir", str(project.raw),
                       "--log-dir", str(project.logs), *extra])


def source(project: SimpleNamespace, sid: str) -> dict[str, Any]:
    return next(s for s in json.loads(project.sources.read_text())["sources"] if s["id"] == sid)


def test_run_all_writes_manifest_and_sources(project, site, client) -> None:
    assert run(args_for(project, "--source", "all"), client) == 1  # 1 because two sources need manual steps

    manifest = json.loads((project.raw / "good" / "MANIFEST.json").read_text())
    assert manifest["source"] == "good" and sorted(manifest["files"]) == ["a.bin", "b.bin"]
    a = manifest["files"]["a.bin"]
    assert a["url"] == site.url("/d/a.bin") and a["bytes"] == len(DATA) and a["sha256"] == sha256(DATA)
    assert {"downloaded_utc", "etag", "last_modified"} <= a.keys()

    good = source(project, "good")
    assert good["raw_bytes"] == len(DATA) + len(OTHER)
    assert good["collection_date"] == datetime.now(timezone.utc).date().isoformat()
    assert good["sha256"] == combined_sha256(manifest["files"])

    for sid in ("blocked", "denied", "off"):
        assert source(project, sid)["raw_bytes"] is None
    assert site.file_requests("/private/c.bin") == [] and site.file_requests("/d/off.bin") == []
    assert "MANUAL DOWNLOAD NEEDED for blocked" in (project.logs / "download_blocked.log").read_text()
    assert "MANUAL DOWNLOAD NEEDED for denied" in (project.logs / "download_denied.log").read_text()


def test_rerun_skips_finished_files(project, site, client) -> None:
    run(args_for(project, "--source", "good"), client)
    before = len(site.requests)
    assert run(args_for(project, "--source", "good"), client) == 0
    assert site.requests[before:] == []


def test_limit_bytes_leaves_sources_untouched(project, client) -> None:
    run(args_for(project, "--source", "good", "--limit-bytes", "1000"), client)
    assert not (project.raw / "good" / "MANIFEST.json").exists()
    assert (project.raw / "good" / "a.bin.part").stat().st_size == 1000
    assert source(project, "good")["raw_bytes"] is None


def test_disabled_source_runs_when_named(project, site, client) -> None:
    assert run(args_for(project, "--source", "off"), client) == 0
    assert source(project, "off")["raw_bytes"] == len(DATA)


def test_register_manual(project, client) -> None:
    folder = project.raw / "blocked"
    folder.mkdir(parents=True)
    (folder / "c.bin").write_bytes(DATA)
    assert run(args_for(project, "--source", "blocked", "--register-manual"), client) == 0
    entry = json.loads((folder / "MANIFEST.json").read_text())["files"]["c.bin"]
    assert entry["obtained"] == "manual" and entry["sha256"] == sha256(DATA)
    assert source(project, "blocked")["sha256"] == sha256(DATA)


def test_file_already_in_place_is_recorded_not_fetched(project, site, client) -> None:
    folder = project.raw / "good"
    folder.mkdir(parents=True)
    (folder / "a.bin").write_bytes(DATA)
    run(args_for(project, "--source", "good"), client)
    assert site.file_requests("/d/a.bin") == []
    files = json.loads((folder / "MANIFEST.json").read_text())["files"]
    assert files["a.bin"]["obtained"] == "manual" and files["b.bin"]["obtained"] == "download"


def test_dry_run_downloads_nothing(project, site, client, capsys) -> None:
    assert run(args_for(project, "--source", "all", "--dry-run"), client) == 0
    out = capsys.readouterr().out
    assert site.url("/d/a.bin") in out and str(project.raw / "good" / "a.bin") in out
    assert "robots.txt DISALLOWED" in out and "(disabled" in out
    assert {r["path"] for r in site.requests} == {"/robots.txt"}
    assert not project.raw.exists()


def test_unknown_source(project, client) -> None:
    assert run(args_for(project, "--source", "nope"), client) == 2


# ---------------------------------------------------------------------------
# Resolvers, against canned API answers
# ---------------------------------------------------------------------------

class CannedClient:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers

    def get(self, url: str, params: dict[str, str] | None = None, headers: Any = None) -> SimpleNamespace:
        key = url + ("?" + "&".join(f"{k}={v}" for k, v in params.items()) if params else "")
        answer = self.answers[key]
        links = answer.pop("_links_header", {}) if isinstance(answer, dict) else {}
        return SimpleNamespace(json=lambda: answer["json"] if isinstance(answer, dict) else answer,
                               text=answer if isinstance(answer, str) else "", links=links)


def test_resolve_wiki_pins_dated_dump() -> None:
    sums = "aaa  urwiki-20261001-stub-articles.xml.gz\n" + "f" * 40 + "  urwiki-20261001-pages-articles.xml.bz2\n"
    status = {"jobs": {"articlesdump": {"files": {"urwiki-20261001-pages-articles.xml.bz2": {"size": 123}}}}}
    c = CannedClient({download.WIKI_SHA1SUMS: sums,
                      download.WIKI_BASE + "20261001/dumpstatus.json": {"json": status}})
    [spec] = download.resolve_wiki_ur(c, {})
    assert spec.url == "https://dumps.wikimedia.org/urwiki/20261001/urwiki-20261001-pages-articles.xml.bz2"
    assert spec.checksum == ("sha1", "f" * 40) and spec.size == 123


def test_resolve_ufal_takes_plain_file_with_md5() -> None:
    def bs(name: str, md5: str) -> dict[str, Any]:
        return {"name": name, "uuid": name, "sizeBytes": 10, "checkSum": {"checkSumAlgorithm": "MD5", "value": md5.upper()},
                "_links": {"content": {"href": f"https://x/{name}/content"}}}
    bundles = {"_embedded": {"bundles": [{"name": "ORIGINAL", "_embedded": {"bitstreams": {"_embedded": {
        "bitstreams": [bs("urdu-tagged-corpus.gz", "1" * 32), bs("urdu-plain-text-corpus.gz", "a" * 32)]}}}}]}}
    c = CannedClient({
        f"{download.LINDAT_API}/pid/find?id=hdl:{download.UFAL_HANDLE}": {"json": {"uuid": "item"}},
        f"{download.LINDAT_API}/core/items/item/bundles?embed=bitstreams": {"json": bundles},
    })
    [spec] = download.resolve_ufal_umc(c, {})
    assert spec.filename == "urdu-plain-text-corpus.gz" and spec.checksum == ("md5", "a" * 32)


def test_resolve_mendeley_reads_sha256() -> None:
    files = [{"id": "fid", "filename": "news.csv", "content_details": {
        "download_url": "https://data.mendeley.com/public-files/x/file_downloaded", "sha256_hash": "B" * 64, "size": 9}}]
    key = (f"https://data.mendeley.com/public-api/datasets/{download.MENDELEY_DATASET}/files"
           f"?folder_id=root&version={download.MENDELEY_VERSION}")
    [spec] = download.resolve_urdu_news_1m(CannedClient({key: {"json": files}}), {})
    assert spec.filename == "news.csv" and spec.checksum == ("sha256", "b" * 64) and spec.size == 9


def test_resolve_fineweb_paginates_and_refuses_removed() -> None:
    api = f"https://huggingface.co/api/datasets/{download.HF_REPO}"
    def item(path: str) -> dict[str, Any]:
        return {"type": "file", "path": path, "size": 5, "lfs": {"oid": "c" * 64}}
    answers = {
        f"{api}/revision/main": {"json": {"sha": "rev1"}},
        f"{api}/tree/rev1/{download.HF_FOLDER}": {"json": [item(f"{download.HF_FOLDER}/001_00000.parquet")],
                                                   "_links_header": {"next": {"url": "page2"}}},
        "page2": {"json": [item(f"{download.HF_FOLDER}/000_00000.parquet")]},
    }
    specs = download.resolve_fineweb2_urd(CannedClient(answers), {})
    assert [s.filename for s in specs] == ["000_00000.parquet", "001_00000.parquet"]
    assert specs[0].url.endswith(f"/resolve/rev1/{download.HF_FOLDER}/000_00000.parquet")
    assert specs[0].checksum == ("sha256", "c" * 64) and specs[0].extra["revision"] == "rev1"

    answers[f"{api}/tree/rev1/{download.HF_FOLDER}"] = {"json": [item("data/urd_Arab_removed/train/000_00000.parquet")]}
    with pytest.raises(DownloadFailed):
        download.resolve_fineweb2_urd(CannedClient(answers), {})


def test_select_files_and_max_files() -> None:
    specs = [FileSpec(f"u{i}", f"{i:03d}.parquet") for i in range(3)]
    assert [s.filename for s in select(specs, ["001.parquet"], None)] == ["001.parquet"]
    assert [s.filename for s in select(specs, None, 2)] == ["000.parquet", "001.parquet"]
    with pytest.raises(DownloadFailed):
        select(specs, ["999.parquet"], None)


def test_combined_sha256() -> None:
    one = {"a": {"sha256": "x" * 64}}
    assert combined_sha256(one) == "x" * 64
    two = {"b": {"sha256": "2" * 64}, "a": {"sha256": "1" * 64}}
    assert combined_sha256(two) == sha256(f"{'1' * 64}  a\n{'2' * 64}  b\n".encode())
