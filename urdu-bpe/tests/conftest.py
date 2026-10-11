"""Shared fixtures: a local HTTP server that the downloader and scraper tests can script."""

import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import pytest

from collect.polite import PoliteClient

LAST_MODIFIED = "Wed, 01 Oct 2026 00:00:00 GMT"


@dataclass
class FakeSite:
    """In-memory files served over HTTP/1.1 with Range and If-Range, plus scripted misbehaviour."""

    base_url: str = ""
    files: dict[str, bytes] = field(default_factory=dict)
    etags: dict[str, str] = field(default_factory=dict)
    robots: str | None = "User-agent: *\nDisallow: /private/\n"
    robots_status: int = 200
    status_queue: dict[str, list[int]] = field(default_factory=dict)  # path -> statuses to answer first
    retry_after: str | None = None
    drop_after: dict[str, list[int]] = field(default_factory=dict)  # path -> per response: send N body bytes, hang up
    hang_up_always: set[str] = field(default_factory=set)  # paths that never send a body byte
    ignore_range: set[str] = field(default_factory=set)
    range_from_zero: set[str] = field(default_factory=set)  # answer 206 but serve from byte 0 (a buggy server)
    redirects: dict[str, tuple[int, str]] = field(default_factory=dict)  # path -> (status, Location)
    scripted: dict[str, list[tuple[int, dict[str, str], bytes]]] = field(default_factory=dict)  # answered first
    headers: dict[str, dict[str, str]] = field(default_factory=dict)  # extra headers on a file's responses
    set_cookie: str | None = None  # Set-Cookie on every file response
    requests: list[dict[str, Any]] = field(default_factory=list)

    def url(self, path: str) -> str:
        return self.base_url + path

    def file_requests(self, path: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"] == path]


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        site: FakeSite = self.server.site  # type: ignore[attr-defined]
        path = urlsplit(self.path).path
        site.requests.append({
            "path": path,
            "range": self.headers.get("Range"),
            "if_range": self.headers.get("If-Range"),
            "user_agent": self.headers.get("User-Agent"),
            "accept_encoding": self.headers.get("Accept-Encoding"),
            "cookie": self.headers.get("Cookie"),
        })
        if script := site.scripted.get(path):
            status, headers, body = script.pop(0)
            return self._send(status, body, headers)
        if path == "/robots.txt":
            if site.robots_status != 200 or site.robots is None:
                return self._send(site.robots_status if site.robots is not None else 404)
            return self._send(200, site.robots.encode(), {"Content-Type": "text/plain"})
        if queue := site.status_queue.get(path):
            status = queue.pop(0)
            return self._send(status, b"no", {"Retry-After": site.retry_after} if site.retry_after else {})
        if path in site.redirects:
            status, location = site.redirects[path]
            return self._send(status, b"", {"Location": location})
        if path not in site.files:
            return self._send(404)
        data, etag = site.files[path], site.etags.get(path, '"v1"')
        start, status = 0, 200
        wanted = self.headers.get("Range")
        if_range = self.headers.get("If-Range")
        if wanted and path not in site.ignore_range and if_range in (None, etag, LAST_MODIFIED):
            start = int(re.fullmatch(r"bytes=(\d+)-", wanted).group(1))
            if start >= len(data):
                return self._send(416, headers={"Content-Range": f"bytes */{len(data)}"})
            status = 206
            if path in site.range_from_zero:
                start = 0
        body = data[start:]
        self.send_response(status)
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", LAST_MODIFIED)
        self.send_header("Accept-Ranges", "bytes")
        for k, v in site.headers.get(path, {}).items():
            self.send_header(k, v)
        if site.set_cookie:
            self.send_header("Set-Cookie", site.set_cookie)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.end_headers()
        queue = site.drop_after.get(path)
        cut = 0 if path in site.hang_up_always else (queue.pop(0) if queue else None)
        if cut is not None:
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.close_connection = True  # hang up mid-body
            return
        self.wfile.write(body)


class _QuietServer(ThreadingHTTPServer):
    def handle_error(self, request: Any, client_address: Any) -> None:
        pass  # clients hanging up mid-response is part of what these tests do


@pytest.fixture
def site():
    server = _QuietServer(("127.0.0.1", 0), _Handler)
    fake = FakeSite(base_url=f"http://127.0.0.1:{server.server_port}")
    server.site = fake  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def client(sleeps: list[float]) -> PoliteClient:
    c = PoliteClient("TestBot/1.0 (tests; contact: nobody@example.invalid)", min_interval=0,
                     max_retries=3, backoff_base=0.01, sleep=sleeps.append)
    c.session.trust_env = False  # never route 127.0.0.1 through a proxy from the environment
    return c
