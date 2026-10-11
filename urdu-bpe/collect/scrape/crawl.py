"""Phase B crawler: polite, resumable collection of one site's articles.

    python collect/scrape/crawl.py --site voa_urdu --target-mb 70
    python collect/scrape/crawl.py --site humsub --target-mb 20 --max-hours 7   # whichever comes first
    python collect/scrape/crawl.py --site voa_urdu --limit 50 --max-sitemaps 1   # smoke test
    python collect/scrape/crawl.py --site voa_urdu --reextract                   # rebuild from cache, no requests

Every request goes through ScrapeClient (collect/scrape/fetch.py): both
robots.txt parsers must allow it, max(Crawl-delay, 1 s) + 0-0.5 s jitter
between requests, one connection, no cookies, back-off on 429/503 honouring
Retry-After. A bot challenge or 20 consecutive 403s stop the site.

1. Discovery. The adapter's sitemap entry points (collect/scrape/sites/) are
   read breadth-first. Children of a sitemap index are followed only if the
   adapter's follow_sitemap() accepts them. Only URLs matching the adapter's
   article regex (and none of its blocked patterns) are kept. Sitemaps
   already read are skipped on a rerun (--rediscover reads them again).
   --max-sitemaps stops discovery early, for smoke tests; the next run carries on.
2. Crawl order. A seeded pseudo-random order: URLs sorted by
   sha1("<seed>:<url>"). The text then spreads over the whole archive, and a
   URL's place does not depend on which other URLs were found.
3. Fetch and extract. The raw HTML is cached gzipped before extraction.
   Pages the adapter rejects (media pages, reprints, too little text) and
   duplicate canonical URLs are recorded in skips.jsonl.
4. Stop at --target-mb of extracted UTF-8 text (MB = 10^6 bytes, default 70),
   when --max-hours of wall-clock time have passed since the start of this run
   (discovery included), at --limit kept articles, or when the URLs run out.

Resumable: URLs already kept or skipped are passed over; URLs fetched with
status 200 are re-extracted from the cache instead of fetched again; 404s,
robots-blocked URLs and rejected redirects are not retried. Kill it any time
and rerun the same command.

Files, in data/raw/own_scrape/<site>/:
  sitemaps.jsonl, sitemaps/   sitemaps read (with an index's children) and their raw bodies
  urls.jsonl                  article URLs discovered
  fetched.jsonl               one row per article fetch: url, final_url, outcome, status, time, html_sha256
  html/<sha1(url)>.html.gz    raw HTML
  articles.jsonl              {url, canonical_url, title, text, published, fetched_at_utc, http_status,
                               html_sha256, extractor_version}; text = title + "\\n" + body
  skips.jsonl                 {url, reason, extractor_version}
Evidence: evidence/crawl_log_<site>.jsonl.gz, one line per request.
Log: logs/crawl_<site>.log.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect.polite import MissingContact, default_user_agent, utc_now_ms  # noqa: E402
from collect.scrape.fetch import (  # noqa: E402
    BotChallenge, EvidenceLog, HtmlCache, ScrapeClient, SiteBlocked, raise_on_sigterm,
)
from collect.scrape.sitemaps import SitemapTooLarge, parse_sitemap  # noqa: E402
from collect.scrape.sites.base import NotArticle, SiteAdapter  # noqa: E402
from collect.scrape.sites.registry import ADAPTERS  # noqa: E402
from common import LOG_DIR, setup_logging, write_jsonl  # noqa: E402

OWN_DIR = ROOT / "data" / "raw" / "own_scrape"
EVIDENCE_DIR = ROOT / "evidence"
PERMANENT_OUTCOMES = frozenset({"robots", "not_found", "login_required", "redirect_rejected", "too_many_redirects"})
MAX_CONSECUTIVE_ERRORS = 50
PROGRESS_EVERY = 100


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SitePaths:
    root: Path

    @property
    def sitemaps(self) -> Path:
        return self.root / "sitemaps.jsonl"

    @property
    def urls(self) -> Path:
        return self.root / "urls.jsonl"

    @property
    def fetched(self) -> Path:
        return self.root / "fetched.jsonl"

    @property
    def articles(self) -> Path:
        return self.root / "articles.jsonl"

    @property
    def skips(self) -> Path:
        return self.root / "skips.jsonl"

    @property
    def html(self) -> HtmlCache:
        return HtmlCache(self.root / "html")

    @property
    def sitemap_cache(self) -> HtmlCache:
        return HtmlCache(self.root / "sitemaps", suffix=".xml.gz")


def _cut_partial_last_line(path: Path) -> None:
    """A run killed mid-write can leave a last line without its newline: cut it off."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        data = path.read_bytes()
        f.truncate(data.rfind(b"\n") + 1)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON Lines file we append to; a partial last line from a killed run is ignored."""
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.endswith("\n"):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


class Appender:
    """Append JSON rows to a file, one flushed line each."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _cut_partial_last_line(path)
        self._fh = open(path, "a", encoding="utf-8", newline="\n")

    def write(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class Discovery:
    urls: list[str]
    complete: bool
    sitemaps_read: int = 0  # fetched in this run
    new_urls: int = 0


def discover(client: ScrapeClient, adapter: SiteAdapter, paths: SitePaths, *, max_sitemaps: int | None,
             rediscover: bool, log: logging.Logger, deadline: float | None = None) -> Discovery:
    """Read sitemaps breadth-first and return every article URL found so far (this run and earlier ones)."""
    done = {} if rediscover else {r["url"]: r for r in read_rows(paths.sitemaps) if r["outcome"] == "ok"}
    found: dict[str, None] = dict.fromkeys(r["url"] for r in read_rows(paths.urls))
    seeds = list(adapter.sitemap_urls) or [u for u in client.robots(adapter.home_url).sitemaps if adapter.allow_url(u)]
    queue: deque[str] = deque(seeds)
    seen: set[str] = set()
    result = Discovery([], complete=True)
    url_lists = 0
    sitemap_log, url_log = Appender(paths.sitemaps), Appender(paths.urls)
    try:
        while queue:
            sm_url = queue.popleft()
            if sm_url in seen:
                continue
            seen.add(sm_url)
            if sm_url in done:
                queue.extend(c for c in done[sm_url].get("children", []) if adapter.follow_sitemap(c))
                continue
            if not adapter.allow_url(sm_url):
                log.warning("sitemap %s is not allowed by the adapter; skipped", sm_url)
                continue
            if max_sitemaps is not None and url_lists >= max_sitemaps:
                result.complete = False
                log.info("--max-sitemaps %d reached; discovery continues on the next run", max_sitemaps)
                break
            if deadline is not None and time.monotonic() >= deadline:
                result.complete = False
                log.info("--max-hours reached during discovery; discovery continues on the next run")
                break
            r = client.fetch(sm_url, kind="sitemap", allow_url=adapter.allow_url)
            result.sitemaps_read += 1
            row: dict[str, Any] = {"url": sm_url, "final_url": r.final_url, "outcome": r.outcome, "status": r.status,
                                   "fetched_at_utc": r.fetched_at_utc}
            if r.ok:
                paths.sitemap_cache.put(sm_url, r.body)
                try:
                    sm = parse_sitemap(r.body)
                except SitemapTooLarge as e:
                    row.update(outcome="too_large", error=str(e))
                    sm = None
                if sm is not None:
                    row.update(kind=sm.kind, entries=len(sm.entries))
                    if sm.kind == "index":
                        row["children"] = [e.loc for e in sm.entries]
                        queue.extend(c for c in row["children"] if adapter.follow_sitemap(c))
                    elif sm.kind == "urlset":
                        url_lists += 1
                        matched = new = 0
                        for e in sm.entries:
                            url = adapter.normalize_url(e.loc)
                            if not adapter.keep_entry(e) or not adapter.is_article_url(url):
                                continue
                            matched += 1
                            if url not in found:
                                found[url] = None
                                url_log.write({"url": url, "lastmod": e.lastmod, "sitemap": sm_url})
                                new += 1
                        row.update(article_urls=matched, new_urls=new)
                        result.new_urls += new
            sitemap_log.write(row)
            log.info("sitemap %s -> %s %s (%s, %s entries, %s new article URLs)", sm_url, r.outcome, r.status,
                     row.get("kind"), row.get("entries"), row.get("new_urls"))
    finally:
        sitemap_log.close()
        url_log.close()
    result.urls = list(found)
    return result


def crawl_order(urls: Iterable[str], seed: int) -> list[str]:
    """Seeded pseudo-random order that does not depend on which other URLs are present."""
    return sorted(urls, key=lambda u: hashlib.sha1(f"{seed}:{u}".encode("utf-8")).hexdigest())


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

@dataclass
class CrawlSummary:
    stop_reason: str = ""
    kept: int = 0  # articles kept in this run
    skipped: Counter[str] = field(default_factory=Counter)
    fetched: Counter[str] = field(default_factory=Counter)  # outcome -> count, this run
    from_cache: int = 0
    text_bytes: int = 0  # all articles.jsonl, after this run
    articles_total: int = 0


def article_row(url: str, meta: dict[str, Any], e: Any, adapter: SiteAdapter) -> dict[str, Any]:
    return {"url": url, "canonical_url": e.canonical_url, "title": e.title, "text": e.text,
            "published": e.published, "fetched_at_utc": meta["fetched_at_utc"], "http_status": meta["status"],
            "html_sha256": meta["html_sha256"], "extractor_version": adapter.extractor_version}


def crawl(client: ScrapeClient, adapter: SiteAdapter, paths: SitePaths, order: Sequence[str], *,
          target_bytes: int, limit: int | None, log: logging.Logger, deadline: float | None = None) -> CrawlSummary:
    s = CrawlSummary()
    fetched = {r["url"]: r for r in read_rows(paths.fetched)}  # the last row for a URL wins
    articles = read_rows(paths.articles)
    done = {a["url"] for a in articles} | {r["url"] for r in read_rows(paths.skips)}
    canonicals = {a["canonical_url"] for a in articles if a.get("canonical_url")}
    s.text_bytes = sum(len(a["text"].encode("utf-8")) for a in articles)
    s.articles_total = len(articles)
    cache = paths.html
    fetched_log, article_log, skip_log = Appender(paths.fetched), Appender(paths.articles), Appender(paths.skips)
    consecutive_errors = 0
    started, requests_at_start = time.monotonic(), client.attempts
    log.info("crawl: %d URLs in order, %d articles (%.2f MB) already kept, target %.1f MB",
             len(order), len(articles), s.text_bytes / 1e6, target_bytes / 1e6)
    try:
        for url in order:
            if s.text_bytes >= target_bytes:
                s.stop_reason = f"target reached ({s.text_bytes / 1e6:.2f} MB >= {target_bytes / 1e6:.1f} MB)"
                break
            if limit is not None and s.kept >= limit:
                s.stop_reason = f"--limit {limit} articles reached"
                break
            if deadline is not None and time.monotonic() >= deadline:
                s.stop_reason = f"--max-hours reached ({s.text_bytes / 1e6:.2f} MB kept)"
                break
            if url in done:
                continue
            prev = fetched.get(url)
            if prev is not None and prev["outcome"] in PERMANENT_OUTCOMES:
                continue
            html = cache.get(url) if prev is not None and prev["outcome"] == "ok" else None
            if html is not None:
                s.from_cache += 1
            else:
                r = client.fetch(url, kind="article", allow_url=adapter.allow_url)
                s.fetched[r.outcome] += 1
                row = {"url": url, "final_url": r.final_url, "outcome": r.outcome, "status": r.status,
                       "fetched_at_utc": r.fetched_at_utc, "bytes": len(r.body),
                       "html_sha256": hashlib.sha256(r.body).hexdigest() if r.ok else None}
                if r.error:
                    row["error"] = r.error
                fetched_log.write(row)
                fetched[url] = row
                if not r.ok:
                    if r.outcome in PERMANENT_OUTCOMES:
                        consecutive_errors = 0
                        continue
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        s.stop_reason = f"{consecutive_errors} consecutive failed fetches (last: {r.outcome} {url})"
                        break
                    continue
                consecutive_errors = 0
                cache.put(url, r.body)
                html = r.body
            meta = fetched[url]
            try:
                e = adapter.extract(html, meta["final_url"])
            except NotArticle as x:
                reason = str(x)
                skip_log.write({"url": url, "reason": reason, "extractor_version": adapter.extractor_version})
                s.skipped[reason] += 1
                done.add(url)
                continue
            if e.canonical_url and e.canonical_url in canonicals:  # another URL already gave us this story
                skip_log.write({"url": url, "reason": "duplicate canonical URL",
                                "extractor_version": adapter.extractor_version})
                s.skipped["duplicate canonical URL"] += 1
                done.add(url)
                continue
            article_log.write(article_row(url, meta, e, adapter))
            done.add(url)
            if e.canonical_url:
                canonicals.add(e.canonical_url)
            s.kept += 1
            s.articles_total += 1
            s.text_bytes += len(e.text.encode("utf-8"))
            if s.kept % PROGRESS_EVERY == 0:
                _progress(s, started, client.attempts - requests_at_start, target_bytes, log)
        else:
            s.stop_reason = "no URLs left"
    finally:
        for f in (fetched_log, article_log, skip_log):
            f.close()
    return s


def _progress(s: CrawlSummary, started: float, requests: int, target_bytes: int, log: logging.Logger) -> None:
    elapsed = time.monotonic() - started
    rate = s.kept / elapsed if elapsed else 0.0  # articles per second, this run
    per_article = (s.text_bytes / s.articles_total) if s.articles_total else 0
    left = max(0, target_bytes - s.text_bytes)
    eta_h = (left / per_article / rate / 3600) if rate and per_article else float("nan")
    log.info("progress: %d kept this run (%d total), %.2f MB, %d requests in %.0f min, skipped %s; ETA %.1f h",
             s.kept, s.articles_total, s.text_bytes / 1e6, requests, elapsed / 60, dict(s.skipped), eta_h)


# ---------------------------------------------------------------------------
# Re-extraction (no network)
# ---------------------------------------------------------------------------

def reextract(adapter: SiteAdapter, paths: SitePaths, log: logging.Logger) -> tuple[int, Counter[str]]:
    """Rebuild articles.jsonl and skips.jsonl from fetched.jsonl and the HTML cache. Makes no requests."""
    fetched = {r["url"]: r for r in read_rows(paths.fetched)}
    cache = paths.html
    articles: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    canonicals: set[str] = set()
    for url, meta in fetched.items():  # first-fetch order
        if meta["outcome"] != "ok":
            continue
        html = cache.get(url)
        if html is None:
            log.warning("no cached HTML for %s; skipped (a crawl run will fetch it again)", url)
            continue
        try:
            e = adapter.extract(html, meta["final_url"])
        except NotArticle as x:
            reasons[str(x)] += 1
            skips.append({"url": url, "reason": str(x), "extractor_version": adapter.extractor_version})
            continue
        if e.canonical_url and e.canonical_url in canonicals:
            reasons["duplicate canonical URL"] += 1
            skips.append({"url": url, "reason": "duplicate canonical URL",
                          "extractor_version": adapter.extractor_version})
            continue
        if e.canonical_url:
            canonicals.add(e.canonical_url)
        articles.append(article_row(url, meta, e, adapter))
    write_jsonl(paths.articles, articles)
    write_jsonl(paths.skips, skips)
    return len(articles), reasons


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", required=True, choices=sorted(ADAPTERS), help="site adapter to use")
    p.add_argument("--target-mb", type=float, default=70.0, metavar="MB",
                   help="stop at this much extracted UTF-8 text in articles.jsonl (MB = 10^6 bytes; default 70)")
    p.add_argument("--max-hours", type=float, metavar="H",
                   help="stop this run after H hours of wall-clock time, discovery included (rerun to continue)")
    p.add_argument("--limit", type=int, metavar="N", help="stop after N articles kept in this run (smoke tests)")
    p.add_argument("--max-sitemaps", type=int, metavar="N",
                   help="read at most N URL-list sitemaps in this run (smoke tests); the next run continues")
    p.add_argument("--rediscover", action="store_true", help="read every sitemap again")
    p.add_argument("--reextract", action="store_true",
                   help="rebuild articles.jsonl and skips.jsonl from the HTML cache; makes no requests")
    p.add_argument("--seed", type=int, default=1337, help="crawl-order seed (default 1337)")
    p.add_argument("--min-interval", type=float, default=1.0, metavar="SEC",
                   help="minimum seconds between requests; robots.txt Crawl-delay wins if longer (default 1.0)")
    p.add_argument("--jitter", type=float, default=0.5, metavar="SEC", help="extra random delay, 0 to SEC (default 0.5)")
    p.add_argument("--data-dir", type=Path, default=OWN_DIR, help=argparse.SUPPRESS)
    p.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR, help=argparse.SUPPRESS)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR, help=argparse.SUPPRESS)
    p.add_argument("--user-agent", default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.min_interval < 1.0 and args.user_agent is None:
        p.error("--min-interval below 1.0 s is only for tests")
    if args.max_hours is not None and args.max_hours < 0:
        p.error("--max-hours must be >= 0")
    return args


def run(args: argparse.Namespace, client_kwargs: dict[str, Any] | None = None) -> int:
    adapter = ADAPTERS[args.site]()
    log = setup_logging(f"crawl_{args.site}", args.log_dir)
    paths = SitePaths(args.data_dir / args.site)
    if args.reextract:
        n, reasons = reextract(adapter, paths, log)
        log.info("re-extracted with %s: %d articles kept, skipped %s", adapter.extractor_version, n, dict(reasons))
        return 0
    try:
        ua = args.user_agent or default_user_agent()
    except MissingContact as e:
        log.error("%s", e)
        return 2
    deadline = time.monotonic() + args.max_hours * 3600 if args.max_hours is not None else None
    log.info("start %s at %s: target %.1f MB, max hours %s, limit %s, seed %d", adapter.site_id, utc_now_ms(),
             args.target_mb, args.max_hours, args.limit, args.seed)
    with EvidenceLog(args.evidence_dir / f"crawl_log_{adapter.site_id}.jsonl.gz", {"site": adapter.site_id}) as ev:
        client = ScrapeClient(ua, evidence=ev, min_interval=args.min_interval, jitter=args.jitter, logger=log,
                              **(client_kwargs or {}))
        try:
            d = discover(client, adapter, paths, max_sitemaps=args.max_sitemaps, rediscover=args.rediscover, log=log,
                         deadline=deadline)
            log.info("discovery %s: %d article URLs (%d new), %d sitemaps read this run",
                     "complete" if d.complete else "partial", len(d.urls), d.new_urls, d.sitemaps_read)
            s = crawl(client, adapter, paths, crawl_order(d.urls, args.seed),
                      target_bytes=int(args.target_mb * 1e6), limit=args.limit, log=log, deadline=deadline)
        except (BotChallenge, SiteBlocked) as e:
            log.error("STOPPED %s: %s. Nothing more is fetched from this site; see the evidence log.",
                      adapter.site_id, e)
            return 3
        except KeyboardInterrupt:
            log.warning("interrupted; run the same command again to resume")
            return 130
        finally:
            client.session.close()
    log.info("done: %s. This run: %d kept, %d from cache, fetch outcomes %s, skipped %s. Total: %d articles, "
             "%.2f MB of text, %d requests this run.", s.stop_reason, s.kept, s.from_cache, dict(s.fetched),
             dict(s.skipped), s.articles_total, s.text_bytes / 1e6, client.attempts)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raise_on_sigterm()
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
