"""Phase C: evidence report for our own collection, and its collect/sources.json entry.

    python collect/scrape/report.py                  # every site with crawl data
    python collect/scrape/report.py --sites voa_urdu
    python collect/scrape/report.py --no-sources     # do not touch collect/sources.json

Reads only local files (no requests): the crawl state in data/raw/own_scrape/<site>/,
the request log evidence/crawl_log_<site>.jsonl.gz and the robots/terms snapshots.
Writes evidence/scrape_report.md with, per site: crawl dates; robots.txt and terms
snapshot paths; URLs discovered / fetched / blocked by robots / errors; articles kept
and skipped (by reason); extracted MB (10^6 bytes) and MiB; article date range and
articles per year; 20 random sample URLs with titles (seed 1337); the SHA-256 of
articles.jsonl. Then it updates the own_scrape entry of collect/sources.json
(licence = each site's terms summary, collected_by_us, collection_date, raw_bytes
= size of the articles.jsonl files, sha256, text_bytes and per-site details).

Run it again after the full crawls; the numbers are as of the moment it runs.
"""

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect.download import _locked, combined_sha256  # noqa: E402
from collect.scrape.crawl import EVIDENCE_DIR, OWN_DIR, PERMANENT_OUTCOMES, SitePaths, read_rows  # noqa: E402
from collect.scrape.fetch import iter_log  # noqa: E402
from collect.scrape.sites.base import SiteAdapter  # noqa: E402
from collect.scrape.sites.registry import ADAPTERS  # noqa: E402
from common import LOG_DIR, setup_logging, sha256_file, write_json  # noqa: E402

SOURCES_PATH = ROOT / "collect" / "sources.json"
SAMPLE_SIZE = 20


@dataclass
class SiteStats:
    adapter: SiteAdapter
    first_utc: str | None = None
    last_utc: str | None = None
    requests: int = 0
    request_statuses: Counter[str] = field(default_factory=Counter)
    robots_blocked_requests: int = 0
    challenges: list[str] = field(default_factory=list)
    sitemaps_read: int = 0
    urls_discovered: int = 0
    fetch_outcomes: Counter[str] = field(default_factory=Counter)
    articles: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    text_bytes: int = 0
    extractor_versions: Counter[str] = field(default_factory=Counter)
    years: Counter[str] = field(default_factory=Counter)
    first_published: str | None = None
    last_published: str | None = None
    no_date: int = 0
    samples: list[tuple[str, str]] = field(default_factory=list)
    articles_path: str = ""
    articles_bytes: int = 0
    articles_sha256: str | None = None
    robots_snapshot: str | None = None
    terms_snapshot: str | None = None


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _latest(directory: Path, site_id: str) -> str | None:
    files = sorted(directory.glob(f"{site_id}_*.txt"))
    return _rel(files[-1]) if files else None


def collect_stats(adapter: SiteAdapter, data_dir: Path, evidence_dir: Path, seed: int) -> SiteStats:
    st = SiteStats(adapter)
    paths = SitePaths(data_dir / adapter.site_id)
    for row in iter_log(evidence_dir / f"crawl_log_{adapter.site_id}.jsonl.gz"):
        if row.get("robots_allowed") is False:
            st.robots_blocked_requests += 1
            continue
        st.requests += 1
        st.request_statuses[str(row.get("status"))] += 1
        if row.get("challenge"):
            st.challenges.append(f"{row['utc']} {row['url']}: {row['challenge']}")
        st.first_utc = min(filter(None, [st.first_utc, row["utc"]]))
        st.last_utc = max(filter(None, [st.last_utc, row["utc"]]))
    st.sitemaps_read = sum(1 for r in read_rows(paths.sitemaps) if r["outcome"] == "ok")
    st.urls_discovered = len({r["url"] for r in read_rows(paths.urls)})
    fetched = {r["url"]: r for r in read_rows(paths.fetched)}
    st.fetch_outcomes = Counter(r["outcome"] for r in fetched.values())
    st.skipped = Counter(r["reason"] for r in read_rows(paths.skips))
    articles = read_rows(paths.articles)
    st.articles = len(articles)
    dates = []
    for a in articles:
        st.text_bytes += len(a["text"].encode("utf-8"))
        st.extractor_versions[a["extractor_version"]] += 1
        day = (a.get("published") or "")[:10]
        if len(day) == 10 and day[:4].isdigit():
            dates.append(day)
            st.years[day[:4]] += 1
        else:
            st.no_date += 1
    if dates:
        st.first_published, st.last_published = min(dates), max(dates)
    rng = random.Random(seed)
    st.samples = [(a["url"], a["title"]) for a in rng.sample(articles, min(SAMPLE_SIZE, len(articles)))]
    if paths.articles.exists():
        st.articles_path = _rel(paths.articles)
        st.articles_bytes = paths.articles.stat().st_size
        st.articles_sha256 = sha256_file(paths.articles)
    st.robots_snapshot = _latest(evidence_dir / "robots", adapter.site_id)
    st.terms_snapshot = _latest(evidence_dir / "terms", adapter.site_id)
    return st


def _mb(n: int) -> str:
    return f"{n / 1e6:,.2f} MB ({n / 2**20:,.2f} MiB)"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render(stats: Sequence[SiteStats], generated_utc: str) -> str:
    total_text = sum(s.text_bytes for s in stats)
    lines = [
        "# Our own collection: scrape report",
        "",
        f"Generated by `collect/scrape/report.py` at {generated_utc} from local files only. "
        "MB = 10^6 bytes of UTF-8 text (title + body, before cleaning).",
        "",
        "| site | genre | crawl dates (UTC) | requests | articles | extracted text | published |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(f"| {s.adapter.name} (`{s.adapter.site_id}`) | {s.adapter.genre} | "
                     f"{(s.first_utc or '?')[:10]} to {(s.last_utc or '?')[:10]} | {s.requests:,} | {s.articles:,} | "
                     f"{_mb(s.text_bytes)} | {s.first_published or '?'} to {s.last_published or '?'} |")
    lines += ["", f"**Total extracted text: {_mb(total_text)}** in {sum(s.articles for s in stats):,} articles.", ""]
    lines += ["How it was collected: `collect/scrape/crawl.py`, one adapter per site in `collect/scrape/sites/`. "
              "Every request was checked against robots.txt by our RFC 9309 matcher and urllib.robotparser "
              "(both must allow), spaced by max(Crawl-delay, 1 s) plus 0–0.5 s jitter, one connection per host, "
              "no cookies, no login. Only article URLs from the sites' sitemaps were fetched; search, tag, author, "
              "comment, login and profile URLs never were. Comments, bylines, author boxes and lines holding an "
              "email address are dropped during extraction. The text stays inside the course.", ""]
    for s in stats:
        a = s.adapter
        lines += [f"## {a.name} (`{a.site_id}`)", "",
                  f"- Home: {a.home_url} · genre: {a.genre} · extractor: "
                  + ", ".join(f"`{v}` ({n:,})" for v, n in s.extractor_versions.items()),
                  f"- Crawl dates: {s.first_utc or '?'} to {s.last_utc or '?'} (UTC)",
                  f"- robots.txt snapshot: `{s.robots_snapshot}` · terms snapshot: `{s.terms_snapshot}`"
                  + (f" · terms page: {a.terms_url}" if a.terms_url else " · no terms page on the site"),
                  f"- Terms summary: {a.terms_summary}",
                  f"- Requests: {s.requests:,} (log: `evidence/crawl_log_{a.site_id}.jsonl.gz`), by HTTP status: "
                  + ", ".join(f"{k}: {v:,}" for k, v in sorted(s.request_statuses.items())),
                  f"- Requests refused by our robots check (never sent): {s.robots_blocked_requests:,}",
                  f"- Bot challenges: {len(s.challenges)}" + ("".join(f"\n  - {c}" for c in s.challenges)),
                  f"- Sitemaps read: {s.sitemaps_read:,}; article URLs discovered: {s.urls_discovered:,}",
                  f"- Article URLs fetched: {sum(s.fetch_outcomes.values()):,}: "
                  + ", ".join(f"{k} {v:,}" for k, v in sorted(s.fetch_outcomes.items())),
                  f"  - blocked by robots: {s.fetch_outcomes.get('robots', 0):,}; errors (not 200, not 404, "
                  f"not robots): {sum(v for k, v in s.fetch_outcomes.items() if k not in ('ok', 'robots', 'not_found')):,}",
                  f"- Articles kept: {s.articles:,}; skipped: {sum(s.skipped.values()):,}"
                  + (" (" + ", ".join(f"{k}: {v:,}" for k, v in s.skipped.most_common()) + ")" if s.skipped else ""),
                  f"- Extracted text: {_mb(s.text_bytes)}",
                  f"- Published: {s.first_published or '?'} to {s.last_published or '?'}; no date: {s.no_date:,}",
                  "- Articles per year: " + ", ".join(f"{y}: {n:,}" for y, n in sorted(s.years.items())),
                  f"- `{s.articles_path}`: {s.articles_bytes:,} bytes, SHA-256 `{s.articles_sha256}`",
                  "", f"Random sample of {len(s.samples)} articles (seed 1337):", "",
                  "| # | URL | title |", "|---|---|---|"]
        lines += [f"| {i} | {url} | {_md(title)} |" for i, (url, title) in enumerate(s.samples, 1)]
        lines.append("")
    return "\n".join(lines)


def update_sources(sources_path: Path, stats: Sequence[SiteStats], generated_utc: str) -> None:
    files = {f"{s.adapter.site_id}/articles.jsonl": {"sha256": s.articles_sha256} for s in stats if s.articles_sha256}
    with _locked(sources_path):
        with open(sources_path, encoding="utf-8") as f:
            data = json.load(f)
        entry = next(e for e in data["sources"] if e["id"] == "own_scrape")
        entry.update({
            "name": "Our own collection: " + " and ".join(s.adapter.name for s in stats),
            "genre": "; ".join(f"{s.adapter.genre} ({s.adapter.name})" for s in stats),
            "enabled": True,
            "landing_url": " ; ".join(s.adapter.home_url for s in stats),
            "download_urls": [],
            "licence": " | ".join(f"{s.adapter.name}: {s.adapter.terms_summary}" for s in stats),
            "licence_url": " ; ".join(s.adapter.terms_url or f"none ({s.adapter.name} has no terms page)"
                                      for s in stats),
            "terms_notes": ("Collected by us with collect/scrape/crawl.py: robots.txt obeyed (RFC 9309 matcher and "
                            "urllib.robotparser must both allow), Crawl-delay honoured, no login, no cookies, nothing "
                            "behind a login or paywall. No comments, bylines, author emails or user pages. Kept inside "
                            "the course and never redistributed. Evidence: evidence/scrape_report.md, "
                            "evidence/crawl_log_<site>.jsonl.gz, evidence/robots/, evidence/terms/, "
                            "evidence/probe_report.md."),
            "access_method": "scrape",
            "collected_by_us": True,
            "collection_date": max((s.last_utc or "")[:10] for s in stats) or None,
            "raw_bytes": sum(s.articles_bytes for s in stats),
            "sha256": combined_sha256(files) if files else None,
            "text_bytes": sum(s.text_bytes for s in stats),
            "sites": [{
                "site_id": s.adapter.site_id, "name": s.adapter.name, "landing_url": s.adapter.home_url,
                "genre": s.adapter.genre, "terms_url": s.adapter.terms_url, "terms_snapshot": s.terms_snapshot,
                "robots_snapshot": s.robots_snapshot, "crawl_first_utc": s.first_utc, "crawl_last_utc": s.last_utc,
                "requests": s.requests, "articles": s.articles, "text_bytes": s.text_bytes,
                "articles_jsonl": s.articles_path, "articles_jsonl_bytes": s.articles_bytes,
                "articles_jsonl_sha256": s.articles_sha256, "published_range": [s.first_published, s.last_published],
                "extractor_versions": dict(s.extractor_versions),
            } for s in stats],
            "notes": (f"Numbers as of {generated_utc} (collect/scrape/report.py). raw_bytes is the size of the "
                      "articles.jsonl files (JSON Lines of extracted articles); text_bytes is their UTF-8 title+body "
                      "text. sha256 is the SHA-256 of the sha256sum-style listing of the articles.jsonl files when "
                      "there are several. Rerun the report after the crawls finish."),
        })
        write_json(sources_path, data)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", nargs="+", choices=sorted(ADAPTERS), help="default: every site with crawl data")
    p.add_argument("--no-sources", action="store_true", help="do not update collect/sources.json")
    p.add_argument("--seed", type=int, default=1337, help="seed for the random sample of URLs")
    p.add_argument("--data-dir", type=Path, default=OWN_DIR, help=argparse.SUPPRESS)
    p.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR, help=argparse.SUPPRESS)
    p.add_argument("--sources", type=Path, default=SOURCES_PATH, help=argparse.SUPPRESS)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("scrape_report", args.log_dir)
    site_ids = args.sites or [s for s in ADAPTERS if (args.data_dir / s / "articles.jsonl").exists()]
    if not site_ids:
        log.error("no crawl data under %s", args.data_dir)
        return 1
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stats = [collect_stats(ADAPTERS[s](), args.data_dir, args.evidence_dir, args.seed) for s in site_ids]
    out = args.evidence_dir / "scrape_report.md"
    out.write_text(render(stats, generated), encoding="utf-8")
    log.info("wrote %s: %s", _rel(out), "; ".join(f"{s.adapter.site_id} {s.articles:,} articles, "
                                                 f"{s.text_bytes / 1e6:.2f} MB" for s in stats))
    if not args.no_sources:
        update_sources(args.sources, stats, generated)
        log.info("updated own_scrape in %s", _rel(args.sources))
    return 0


if __name__ == "__main__":
    sys.exit(main())
