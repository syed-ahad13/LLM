"""Polite HTTP client shared by the downloaders and the scraper (CLAUDE.md, hard rule 5).

- Identifies itself with USER_AGENT_TEMPLATE. The contact address comes from
  $URDU_BOT_CONTACT or the first line of collect/contact.local (gitignored).
  There is deliberately no default: the repository is public, and a built-in
  address would make every clone send it to the sites we download from.
- Checks robots.txt (collect/robots.py) for every URL it is asked to fetch.
  robots.txt is fetched once per origin; redirect targets (signed CDN URLs) are
  not re-checked.
- One connection per host: requests are sequential, each host keeps at most one
  pooled connection, and requests to a host are spaced by min_interval seconds
  (or the site's Crawl-delay, if longer).
- Retries connection errors, timeouts, 429 and 5xx with exponential backoff and
  jitter, honouring Retry-After.
- Never logs in: 401 and 403 raise AccessDenied so the caller can print manual steps.
"""

import logging
import os
import random
import time
from collections.abc import Callable, Collection, Mapping
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

from collect.robots import RobotsTxt

CONTACT_FILE = Path(__file__).resolve().parent / "contact.local"
USER_AGENT_TEMPLATE = "UrduCorpusCourseBot/1.0 (NUST SEECS coursework; contact: {contact})"
ROBOTS_MAX_BYTES = 500 * 1024  # RFC 9309: parse at least the first 500 KiB

RETRYABLE_EXCEPTIONS = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


class MissingContact(RuntimeError):
    """No contact address is configured for the User-Agent."""


def contact_address() -> str:
    """$URDU_BOT_CONTACT, else the first non-empty line of collect/contact.local."""
    contact = os.environ.get("URDU_BOT_CONTACT", "").strip()
    if not contact and CONTACT_FILE.exists():
        lines = [line.strip() for line in CONTACT_FILE.read_text(encoding="utf-8").splitlines()]
        contact = next((line for line in lines if line), "")
    if not contact:
        raise MissingContact(
            "No contact address for the User-Agent. Put your email on one line in "
            f"{CONTACT_FILE} (gitignored) or set URDU_BOT_CONTACT."
        )
    return contact


def default_user_agent() -> str:
    return USER_AGENT_TEMPLATE.format(contact=contact_address())


class AccessDenied(Exception):
    """The server answered 401 or 403. We never log in, so this needs a manual download."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.url, self.status = url, status


class RobotsDisallowed(Exception):
    """robots.txt does not allow our user agent to fetch this URL."""

    def __init__(self, url: str) -> None:
        super().__init__(f"robots.txt disallows {url}")
        self.url = url


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


class PoliteClient:
    def __init__(
        self,
        user_agent: str | None = None,
        *,
        min_interval: float = 1.0,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        backoff_cap: float = 300.0,
        timeout: tuple[float, float] = (30.0, 120.0),
        logger: logging.Logger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self.user_agent = user_agent or default_user_agent()
        self.agent_token = self.user_agent.split("/", 1)[0]
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.timeout = timeout
        self.log = logger or logging.getLogger(__name__)
        self.sleep = sleep
        self.clock = clock
        self.rng = rng or random.Random()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = self.user_agent
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=1, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._robots: dict[str, RobotsTxt] = {}
        self._last_request: dict[str, float] = {}

    # -- robots.txt --------------------------------------------------------

    def robots(self, url: str) -> RobotsTxt:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._fetch_robots(origin)
        return self._robots[origin]

    def _fetch_robots(self, origin: str) -> RobotsTxt:
        url = origin + "/robots.txt"
        try:
            resp = self._send(url, check_robots=False, accept=range(400, 500))
        except (requests.RequestException, AccessDenied) as e:
            self.log.warning("robots.txt unreachable at %s (%s): assuming disallow-all", url, e)
            return RobotsTxt.unreachable()
        with resp:
            if resp.status_code >= 400:
                # RFC 9309 2.3.1.3: 4xx (including 401/403) means no restrictions.
                self.log.info("robots.txt %s: HTTP %d, no restrictions", url, resp.status_code)
                return RobotsTxt("")
            text = resp.content[:ROBOTS_MAX_BYTES].decode("utf-8", errors="replace")
        self.log.info("robots.txt %s: %d bytes", url, len(text))
        return RobotsTxt(text)

    def allowed(self, url: str) -> bool:
        return self.robots(url).allowed(url, self.agent_token)

    # -- requests ----------------------------------------------------------

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        stream: bool = False,
        accept: Collection[int] = (),
    ) -> requests.Response:
        """GET with robots check, pacing and retries.

        Returns the response for any status below 400 or listed in `accept`.
        Raises RobotsDisallowed, AccessDenied (401/403), or requests exceptions
        once retries are exhausted. The caller must close streamed responses.
        """
        return self._send(url, headers=headers, params=params, stream=stream, accept=accept)

    def _send(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        stream: bool = False,
        accept: Collection[int] = (),
        check_robots: bool = True,
    ) -> requests.Response:
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(url)
        attempt = 0
        while True:
            self._pace(url)
            try:
                resp = self.session.get(url, headers=headers, params=params, stream=stream, timeout=self.timeout)
            except RETRYABLE_EXCEPTIONS as e:
                if attempt >= self.max_retries:
                    raise
                self.backoff(attempt, f"{type(e).__name__} for {url}")
                attempt += 1
                continue
            status = resp.status_code
            if status in accept:
                return resp
            if status in (401, 403):
                resp.close()
                raise AccessDenied(url, status)
            if status == 429 or status >= 500:
                if attempt >= self.max_retries:
                    resp.close()
                    resp.raise_for_status()
                retry_after = _retry_after_seconds(resp.headers.get("Retry-After"))
                resp.close()
                self.backoff(attempt, f"HTTP {status} for {url}", retry_after)
                attempt += 1
                continue
            if status >= 400:
                resp.close()
                resp.raise_for_status()
            return resp

    def backoff(self, attempt: int, reason: str, retry_after: float | None = None) -> float:
        """Sleep for exponential backoff with jitter (at least retry_after) and return the delay."""
        delay = min(self.backoff_cap, self.backoff_base * 2**attempt)
        delay = delay / 2 + self.rng.uniform(0, delay / 2)
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_cap))
        self.log.warning("%s; retry %d/%d in %.1fs", reason, attempt + 1, self.max_retries, delay)
        self.sleep(delay)
        return delay

    def _pace(self, url: str) -> None:
        host = urlsplit(url).netloc
        interval = self.min_interval
        robots = self._robots.get(f"{urlsplit(url).scheme}://{host}")
        if robots is not None:
            interval = max(interval, robots.crawl_delay(self.agent_token) or 0.0)
        last = self._last_request.get(host)
        if last is not None:
            wait = last + interval - self.clock()
            if wait > 0:
                self.sleep(wait)
        self._last_request[host] = self.clock()
