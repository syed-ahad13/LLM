"""Polite fetching for the scraper: two robots.txt parsers, evidence log, HTML cache, block detection.

ScrapeClient is a PoliteClient (collect/polite.py) with the scraper's settings:
- A URL is fetched only if BOTH our RFC 9309 matcher (collect/robots.py) and
  urllib.robotparser allow it for our user agent (docs/decisions.md). The
  check runs before every request, redirect hops included.
- Delay between requests to a host: max(Crawl-delay, min_interval) plus 0 to
  `jitter` seconds. One connection per host, no concurrency.
- No cookies, no login. Redirects are followed by hand (at most max_redirects)
  so each hop is robots-checked and can be rejected by the caller's URL filter.
- 429/503 back off exponentially and honour Retry-After. A Retry-After longer
  than retry_after_cap stops the site instead of retrying early.
- A bot challenge (Cloudflare, Sucuri, Imperva, DataDome, Akamai, AWS WAF)
  raises BotChallenge at once: we never try to get past one.
- max_consecutive_403 forbidden answers in a row raise SiteBlocked.
- Every attempt is written to an EvidenceLog (gzipped JSON Lines), one line per
  request: url, utc, status, bytes, robots_allowed, elapsed_ms, kind. URLs
  robots.txt blocks get a line with status null and are never requested.
"""

import gzip
import hashlib
import json
import os
import signal
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from collect.polite import RETRYABLE_EXCEPTIONS, Attempt, PoliteClient, retry_after_seconds, utc_now_ms

REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
ACCEPTED_ERRORS = (401, 403, 404, 410)


class BotChallenge(Exception):
    """The site answered with a bot challenge or WAF block page. Stop the site."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{reason} at {url}")
        self.url, self.reason = url, reason


class SiteBlocked(Exception):
    """The site keeps refusing us (consecutive 403s, or a Retry-After we will not shorten). Stop the site."""


# ---------------------------------------------------------------------------
# Bot challenge detection
# ---------------------------------------------------------------------------

# Markers looked for in the first 200 kB of 403/429/503 bodies. Cloudflare also
# injects /cdn-cgi/challenge-platform/ scripts into ordinary 200 pages, so body
# markers are never checked on a 200 except the interstitial title on tiny pages.
_BLOCK_MARKERS: tuple[tuple[bytes, str], ...] = (
    (b"cf-browser-verification", "Cloudflare challenge"),
    (b"cf_chl_opt", "Cloudflare challenge"),
    (b"challenge-platform", "Cloudflare challenge"),
    (b"<title>Just a moment...</title>", "Cloudflare challenge"),
    (b"Attention Required! | Cloudflare", "Cloudflare block"),
    (b"Sucuri WebSite Firewall", "Sucuri firewall block"),
    (b"_Incapsula_Resource", "Imperva/Incapsula challenge"),
    (b"Incapsula incident ID", "Imperva/Incapsula block"),
    (b"captcha-delivery.com", "DataDome challenge"),
    (b"px-captcha", "PerimeterX/HUMAN challenge"),
    (b"errors.edgesuite.net", "Akamai block"),
    (b"awswaf", "AWS WAF challenge"),
)
_INTERSTITIAL_TITLES = (b"<title>Just a moment...</title>", b"<title>Attention Required! | Cloudflare</title>")


def detect_challenge(status: int, headers: Mapping[str, str], body: bytes) -> str | None:
    """Return a reason if this response is a bot challenge or WAF block, else None."""
    h = {k.lower(): v for k, v in headers.items()}
    if h.get("cf-mitigated", "").lower() == "challenge":
        return "Cloudflare challenge (cf-mitigated header)"
    if h.get("x-amzn-waf-action", "").lower() in ("challenge", "captcha"):
        return "AWS WAF challenge"
    if "x-datadome" in h and status in (401, 403):
        return "DataDome block"
    head = body[:200_000]
    if status in (202, 403, 429, 503):
        for marker, reason in _BLOCK_MARKERS:
            if marker in head:
                return reason
    elif status == 200 and len(body) < 50_000 and any(t in head for t in _INTERSTITIAL_TITLES):
        return "Cloudflare challenge (interstitial page)"
    return None


# ---------------------------------------------------------------------------
# Evidence log
# ---------------------------------------------------------------------------

def iter_log(path: Path) -> Iterator[dict[str, Any]]:
    """Read a gzipped JSON Lines log, multi-member and tolerant of a truncated last member.

    A crawl killed with SIGKILL leaves its gzip member without a trailer. The
    lines it had flushed are still read; a partial last line is skipped.
    """
    if not path.exists():
        return
    data = path.read_bytes()
    out = bytearray()
    while data:
        d = zlib.decompressobj(wbits=31)
        try:
            out += d.decompress(data)
        except zlib.error:
            break
        if not d.eof:
            break  # truncated member: keep what it produced
        data = d.unused_data
    for line in bytes(out).split(b"\n"):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # partial last line of a truncated member


class EvidenceLog:
    """Append-only gzipped JSON Lines log; each run writes one gzip member.

    Every line is sync-flushed, so a killed run loses at most its gzip trailer.
    On open, a log whose last member is truncated is rewritten from its
    readable lines first, so members appended later stay readable by any
    gzip tool.
    """

    def __init__(self, path: Path, extra: Mapping[str, Any] | None = None) -> None:
        self.path = path
        self.extra = dict(extra or {})
        path.parent.mkdir(parents=True, exist_ok=True)
        self._repair()
        self._fh = gzip.open(path, "ab")
        self.lines = 0

    def _repair(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            with gzip.open(self.path, "rb") as f:
                while f.read(1 << 20):
                    pass
            return
        except (EOFError, OSError, zlib.error):
            pass
        rows = list(iter_log(self.path))
        tmp = self.path.with_name(f".{self.path.name}.repair-{os.getpid()}")
        with gzip.open(tmp, "wb") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False).encode() + b"\n")
        os.replace(tmp, self.path)

    def write(self, record: Mapping[str, Any]) -> None:
        row = {**self.extra, **record}
        self._fh.write(json.dumps(row, ensure_ascii=False).encode() + b"\n")
        self._fh.flush(zlib.Z_SYNC_FLUSH)
        self.lines += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "EvidenceLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def raise_on_sigterm() -> None:
    """Turn SIGTERM (tmux kill-session, kill) into SystemExit so logs and files close cleanly."""
    def handler(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, handler)


# ---------------------------------------------------------------------------
# HTML cache
# ---------------------------------------------------------------------------

def url_sha1(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


class HtmlCache:
    """Raw response bodies at <dir>/<sha1(url)><suffix> (gzip), written atomically."""

    def __init__(self, directory: Path, suffix: str = ".html.gz") -> None:
        self.dir = directory
        self.suffix = suffix

    def path(self, url: str) -> Path:
        return self.dir / f"{url_sha1(url)}{self.suffix}"

    def get(self, url: str) -> bytes | None:
        p = self.path(url)
        if not p.exists():
            return None
        try:
            return gzip.decompress(p.read_bytes())
        except (OSError, EOFError, zlib.error):
            return None  # damaged entry: treat as missing and fetch again

    def put(self, url: str, body: bytes) -> Path:
        p = self.path(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
        tmp.write_bytes(gzip.compress(body, compresslevel=6, mtime=0))
        os.replace(tmp, p)
        return p


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str  # as requested
    final_url: str  # after redirects
    outcome: str  # ok | robots | not_found | forbidden | login_required | http_error | error |
    #              redirect_rejected | too_many_redirects | unexpected_status
    status: int | None = None
    body: bytes = b""
    content_type: str = ""
    redirects: list[str] = field(default_factory=list)
    error: str | None = None
    fetched_at_utc: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"


class ScrapeClient(PoliteClient):
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        evidence: EvidenceLog | None = None,
        min_interval: float = 1.0,
        jitter: float = 0.5,
        max_consecutive_403: int = 20,
        retry_after_cap: float = 1800.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("backoff_cap", retry_after_cap)
        super().__init__(user_agent, min_interval=min_interval, jitter=jitter, cookies=False,
                         on_response=self._on_attempt, **kwargs)
        self.evidence = evidence
        self.max_consecutive_403 = max_consecutive_403
        self.retry_after_cap = retry_after_cap
        self.consecutive_403 = 0
        self.attempts = 0
        self.kind = "page"
        self._stdlib: dict[str, RobotFileParser] = {}

    # -- robots.txt: both parsers must allow ----------------------------------

    def stdlib_parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._stdlib:
            rp = RobotFileParser(origin + "/robots.txt")
            rp.parse(self.robots(url).text.splitlines())
            self._stdlib[origin] = rp
        return self._stdlib[origin]

    def robots_verdicts(self, url: str, agent: str | None = None) -> tuple[bool, bool]:
        """(RFC 9309 matcher allows, urllib.robotparser allows) for agent (default: our product token)."""
        robots = self.robots(url)
        token = agent or self.agent_token
        rfc = robots.allowed(url, token)
        if robots.disallow_all:
            return rfc, rfc  # unreachable robots.txt: the stdlib parser has no text to go on
        return rfc, self.stdlib_parser(url).can_fetch(agent or self.user_agent, url)

    def allowed(self, url: str) -> bool:
        rfc, stdlib = self.robots_verdicts(url)
        return rfc and stdlib

    # -- evidence ------------------------------------------------------------

    def _log(self, record: dict[str, Any]) -> None:
        if self.evidence is not None:
            self.evidence.write(record)

    def _on_attempt(self, a: Attempt) -> None:
        self.attempts += 1
        is_robots = urlsplit(a.url).path == "/robots.txt"
        record: dict[str, Any] = {
            "url": a.url, "utc": a.started_utc, "status": a.status, "bytes": a.nbytes,
            "robots_allowed": True, "elapsed_ms": a.elapsed_ms, "kind": "robots" if is_robots else self.kind,
        }
        resp = a.response
        if resp is not None and resp.status_code in REDIRECT_STATUSES and resp.headers.get("Location"):
            record["location"] = urljoin(a.url, resp.headers["Location"])
        if a.error is not None:
            record["error"] = f"{type(a.error).__name__}: {a.error}"[:300]
        if resp is not None:
            reason = detect_challenge(resp.status_code, resp.headers, resp.content if not a.streamed else b"")
            if reason:
                record["challenge"] = reason
                self._log(record)
                raise BotChallenge(a.url, reason)
            if resp.status_code in (429, 503):
                retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
                if retry_after is not None and retry_after > self.retry_after_cap:
                    record["error"] = f"Retry-After {retry_after:.0f}s exceeds cap {self.retry_after_cap:.0f}s"
                    self._log(record)
                    raise SiteBlocked(f"{a.url}: {record['error']}")
        self._log(record)

    def _log_blocked(self, url: str, kind: str) -> None:
        rfc, stdlib = self.robots_verdicts(url)
        self._log({"url": url, "utc": utc_now_ms(), "status": None, "bytes": 0, "robots_allowed": False,
                   "elapsed_ms": 0, "kind": kind, "robots_rfc9309": rfc, "robots_stdlib": stdlib})

    # -- fetching ------------------------------------------------------------

    def fetch(self, url: str, *, kind: str = "page", allow_url: Callable[[str], bool] | None = None,
              max_redirects: int = 5) -> FetchResult:
        """GET url politely, following up to max_redirects redirects by hand.

        Each hop must pass robots.txt and, if given, allow_url (e.g. same site,
        not a login page). Returns a FetchResult for every outcome; raises only
        BotChallenge and SiteBlocked, which mean "stop this site".
        """
        result = FetchResult(url=url, final_url=url, outcome="error", fetched_at_utc=utc_now_ms())
        current = url
        for _ in range(max_redirects + 1):
            result.final_url = current
            if not self.allowed(current):
                self._log_blocked(current, kind)
                result.outcome = "robots"
                return result
            self.kind = kind
            try:
                resp = self.get(current, accept=ACCEPTED_ERRORS, allow_redirects=False)
            except requests.HTTPError as e:
                result.status = e.response.status_code if e.response is not None else None
                result.outcome, result.error = "http_error", str(e)[:300]
                self.consecutive_403 = 0
                return result
            except RETRYABLE_EXCEPTIONS as e:
                result.outcome, result.error = "error", f"{type(e).__name__}: {e}"[:300]
                return result
            with resp:
                status = resp.status_code
                result.status = status
                result.fetched_at_utc = utc_now_ms()
                if status == 403:
                    self.consecutive_403 += 1
                    result.outcome = "forbidden"
                    if self.consecutive_403 >= self.max_consecutive_403:
                        raise SiteBlocked(f"{self.consecutive_403} consecutive HTTP 403 responses (last: {current})")
                    return result
                self.consecutive_403 = 0
                if status in REDIRECT_STATUSES and resp.headers.get("Location"):
                    target = urljoin(current, resp.headers["Location"])
                    result.redirects.append(target)
                    if allow_url is not None and not allow_url(target):
                        result.outcome, result.final_url = "redirect_rejected", target
                        return result
                    current = target
                    continue
                if status == 401:
                    result.outcome = "login_required"
                elif status in (404, 410):
                    result.outcome = "not_found"
                elif status == 200:
                    result.outcome = "ok"
                    result.body = resp.content
                    result.content_type = resp.headers.get("Content-Type", "")
                else:
                    result.outcome = "unexpected_status"
                return result
        result.outcome = "too_many_redirects"
        return result


def same_site(url: str, domain: str, path_prefix: str = "") -> bool:
    """True if url is http(s) on domain (or a subdomain) and its path starts with path_prefix."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return (parts.scheme in ("http", "https")
            and (host == domain or host.endswith("." + domain))
            and (parts.path or "/").startswith(path_prefix))
