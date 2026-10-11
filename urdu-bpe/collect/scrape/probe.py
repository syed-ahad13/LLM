"""Phase A probe: can each candidate site be used for our own collection?

    python collect/scrape/probe.py                          # every candidate
    python collect/scrape/probe.py --sites humsub dw_urdu   # some of them
    python collect/scrape/probe.py --max-requests 30 --articles 10

For each site, with at most --max-requests requests spaced by at least 1 s
plus jitter (collect/scrape/fetch.py):
1. robots.txt is saved to evidence/robots/<site>_<YYYYMMDD>.txt. Sample
   article URLs are tested for User-agent * and for our product token, with
   both our RFC 9309 matcher and urllib.robotparser. Crawl-delay is recorded.
2. Sitemaps come from robots.txt Sitemap lines, /sitemap.xml, /sitemap_index.xml
   and the candidate's extra list. Child sitemaps are grouped by name pattern
   (digits masked). Each group is counted in full when it is small; otherwise
   a seeded sample is counted and scaled to the group's size.
3. --articles article pages, sampled from the sitemaps with the seed, are
   fetched. For each: HTTP status, bot challenge, and whether the body text
   is in the raw HTML ("server"), only inside a script/JSON blob ("json"), or
   absent ("none": needs JavaScript).
4. The body is extracted with a site-independent heuristic (largest block of
   Arabic-script <p> elements) to get the median UTF-8 bytes per article.
5. The terms page (links named Terms, Privacy, شرائط, قواعد, ...) is saved to
   evidence/terms/<site>_<YYYYMMDD>.txt and searched for clauses on scraping,
   crawling, robots, automated access, data mining, AI/ML training and
   reproduction. Comment lines in robots.txt are searched the same way: some
   sites (BBC, DW) state their terms there.
6. Estimated MB = estimated article count x median bytes (MB = 10^6 bytes).

A bot challenge stops the site at once. Outputs: evidence/probe_report.md,
evidence/probe_<YYYYMMDD>.json and evidence/probe_log_<YYYYMMDD>.jsonl.gz
(one line per request). The sample pages are cached in
data/raw/own_scrape/_probe/<site>/ (gitignored) for writing the adapters.
"""

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collect.polite import MissingContact, default_user_agent  # noqa: E402
from collect.robots import RobotsTxt  # noqa: E402
from collect.scrape.extract import (  # noqa: E402
    arabic_letters, collapse_ws, drop_elements, largest_text_block, page_title, published_date, soup_of,
)
from collect.scrape.fetch import (  # noqa: E402
    BotChallenge, EvidenceLog, HtmlCache, ScrapeClient, SiteBlocked, same_site,
)
from collect.scrape.sitemaps import Sitemap, SitemapTooLarge, parse_sitemap, type_key  # noqa: E402
from common import LOG_DIR, setup_logging, write_json  # noqa: E402

EVIDENCE_DIR = ROOT / "evidence"
PROBE_DATA_DIR = ROOT / "data" / "raw" / "own_scrape" / "_probe"
SAMPLES_PER_GROUP = 3


@dataclass(frozen=True)
class Candidate:
    site_id: str
    name: str
    home_url: str
    domain: str  # registrable domain; redirects and sitemaps must stay on it
    article_re: str  # first guess at the article-URL pattern, checked against the sitemaps
    path_prefix: str = "/"  # for sections of shared hosts (bbc.com/urdu, dw.com/ur)
    extra_sitemaps: tuple[str, ...] = ()
    genre: str = "news"


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("humsub", "Hum Sub", "https://www.humsub.com.pk/", "humsub.com.pk",
              r"^https?://(?:www\.)?humsub\.com\.pk/\d+/[^/?#]+/?$", genre="opinion/blogs"),
    Candidate("independent_urdu", "Independent Urdu", "https://www.independenturdu.com/", "independenturdu.com",
              r"^https?://(?:www\.)?independenturdu\.com/node/\d+(?:/[^?#]*)?$"),
    Candidate("bbc_urdu", "BBC Urdu", "https://www.bbc.com/urdu", "bbc.com",
              r"^https?://www\.bbc\.com/urdu/(?:articles/[a-z0-9]{8,}|[a-z0-9-]+-\d{6,})$",
              path_prefix="/urdu", extra_sitemaps=("https://www.bbc.com/urdu/sitemap.xml",)),
    Candidate("dw_urdu", "DW Urdu", "https://www.dw.com/ur/", "dw.com",
              r"^https?://www\.dw\.com/ur/[^/?#]+/a-\d+$", path_prefix="/ur"),
    Candidate("urdunews", "Urdu News", "https://www.urdunews.com/", "urdunews.com",
              r"^https?://(?:www\.)?urdunews\.com/node/\d+(?:/[^?#]*)?$"),
    Candidate("voa_urdu", "VOA Urdu", "https://www.urduvoa.com/", "urduvoa.com",
              r"^https?://(?:www\.)?urduvoa\.com/a/(?:[^/?#]+/)?\d+\.html$"),
    Candidate("express", "Express News", "https://www.express.pk/", "express.pk",
              r"^https?://(?:www\.)?express\.pk/story/\d+(?:/[^?#]*)?$"),
)
CANDIDATE_IDS = tuple(c.site_id for c in CANDIDATES)

# Child-sitemap names that rarely hold articles: sampled last.
_UNLIKELY_SITEMAP = re.compile(r"tag|categor|author|user|topic|section|video|image|gallery|page-sitemap|static",
                               re.I)
_NEWS_SITEMAP = re.compile(r"news", re.I)
# Media sitemaps: their pages may match the article regex but hold little text.
_MEDIA_SITEMAP = re.compile(r"video|gallery|slideshow|photo|audio|podcast", re.I)
# Generic fallback when the guessed article regex matches nothing.
_NON_ARTICLE_PATH = re.compile(
    r"/(tag|tags|category|categories|author|authors|topic|topics|search|page|feed|video|videos|gallery|"
    r"live|login|user|users|profile|account)(/|$)", re.I)

# ---------------------------------------------------------------------------
# Terms of use
# ---------------------------------------------------------------------------

_POLICY_LINKS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (3, re.compile(r"terms|conditions|user agreement", re.I)),
    (3, re.compile(r"شرائط|قواعد|ضوابط")),
    (2, re.compile(r"legal|disclaimer|copyright|imprint|ڈس ?کلیمر", re.I)),
    (1, re.compile(r"privacy|پرائیویسی|پرائیوسی|رازداری", re.I)),
)

CLAUSE_PATTERNS: dict[str, re.Pattern[str]] = {
    "scraping/crawling": re.compile(r"scrap(?:e|es|ed|ing|er|ers)\b|crawl|spider|harvest|اسکریپ|سکریپ|کرال", re.I),
    "robots": re.compile(r"\brobots?\b|robots\.txt|روبوٹ", re.I),
    "automated access": re.compile(
        r"automat(?:ed|ic|ically|ion)|\bbots?\b|\bscripts?\b.{0,40}\b(?:access|collect|download)|خودکار|بوٹ", re.I),
    "data mining": re.compile(r"data[- ]min(?:e|es|ing)\b|\bTDM\b|data extraction|datasets?\b|ڈیٹا مائننگ", re.I),
    "AI/ML training": re.compile(
        r"artificial intelligence|\bA\.?I\.?\b|machine[- ]learning|\bLLMs?\b|language models?|"
        r"train(?:ing)?\s+(?:of\s+)?(?:an?\s+)?(?:AI|models?|algorithms?|software)|مصنوعی ذہانت", re.I),
    "reproduction": re.compile(
        r"reproduc|republish|re-?distribut|\bcopy(?:ing)?\b|commercial (?:use|purpose)|"
        r"نقل|کاپی|دوبارہ شائع|اجازت کے بغیر", re.I),
}
# Stock robots.txt comments shipped by CMSs (Drupal, WordPress plugins): describe robots.txt, restrict nothing.
_ROBOTS_BOILERPLATE = re.compile(
    r"This file is to prevent the crawling|By telling these \"?robots\"? where not to go|"
    r"This file will be ignored unless|For more information about the robots\.txt standard|"
    r"^(?:Used|Ignored):\s", re.I)
RESTRICTIVE_CATEGORIES = ("scraping/crawling", "robots", "automated access", "data mining", "AI/ML training")
_PROHIBITIVE = re.compile(
    r"\bnot\b|n't\b|prohibit|forbid|restrict|without (?:the )?(?:prior |express )?(?:written )?"
    r"(?:permission|consent|authori[sz]ation)|\bno\b|ممنوع|منع|نہیں|اجازت کے بغیر", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?۔؟])\s+")


_POLICY_PATH = re.compile(r"terms|conditions|privacy|policy|legal|disclaimer|imprint|copyright", re.I)
MAX_POLICY_LINK_TEXT = 40  # headlines are longer; "استعمال کی شرائط" or "Privacy Policy" are short


def find_policy_links(soup: Any, page_url: str) -> list[dict[str, Any]]:
    """Links that look like terms/privacy pages, best first: {score, kind, url, text, in_footer}.

    The link text must be short (headlines such as "شرائط اور ضد: ..." are not
    policy links) or the URL path must name a policy. Footer links rank first.
    """
    found: dict[str, dict[str, Any]] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue
        url = urljoin(page_url, href).split("#", 1)[0]
        if urlsplit(url).scheme not in ("http", "https"):
            continue
        text = collapse_ws(a.get_text())
        path = urlsplit(url).path
        text_ok = 0 < len(text) <= MAX_POLICY_LINK_TEXT
        haystack = " ".join(x for x, ok in ((text, text_ok), (path, bool(_POLICY_PATH.search(path)))) if ok)
        score = max((s for s, pat in _POLICY_LINKS if pat.search(haystack)), default=0) if haystack else 0
        if not score:
            continue
        in_footer = a.find_parent("footer") is not None
        if url not in found or found[url]["score"] < score:
            found[url] = {"score": score, "kind": "terms" if score >= 2 else "privacy", "url": url, "text": text,
                          "in_footer": in_footer}
    return sorted(found.values(), key=lambda d: (-d["score"], not d["in_footer"], len(d["url"])))


def robots_clauses(robots_text: str) -> list[dict[str, Any]]:
    """find_clauses over the comment lines of a robots.txt, tagged page="robots.txt"."""
    comments = [line.split("#", 1)[1].strip() for line in robots_text.splitlines() if "#" in line]
    joined, current = [], ""
    for c in comments:  # comment sentences are often wrapped over several lines
        if not c or c.startswith("-"):
            if current:
                joined.append(current)
            current = c.lstrip("- ").strip()
        else:
            current = f"{current} {c}".strip()
    if current:
        joined.append(current)
    return [{**c, "page": "robots.txt", "url": ""} for c in find_clauses("\n".join(joined))
            if not _ROBOTS_BOILERPLATE.search(c["quote"])]


def page_text(html: bytes) -> str:
    """Readable text of a policy page: main/article if present, else body; one line per block."""
    soup = soup_of(html)
    drop_elements(soup, ("script", "style", "noscript", "template", "svg"))
    scope = soup.find("main") or soup.find("article") or soup.body or soup
    lines = [collapse_ws(x) for x in scope.get_text("\n").split("\n")]
    return "\n".join(x for x in lines if x)


def find_clauses(text: str, max_quote: int = 500) -> list[dict[str, Any]]:
    """Sentences mentioning scraping, robots, automated access, data mining, AI training or reproduction."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in text.split("\n"):
        for sentence in _SENTENCE_SPLIT.split(line):
            sentence = sentence.strip()
            if len(sentence) < 15 or sentence in seen:
                continue
            categories = [c for c, pat in CLAUSE_PATTERNS.items() if pat.search(sentence)]
            if not categories:
                continue
            seen.add(sentence)
            quote = sentence if len(sentence) <= max_quote else sentence[: max_quote - 1] + "…"
            out.append({"categories": categories, "prohibitive": bool(_PROHIBITIVE.search(sentence)),
                        "quote": quote})
    return out


# ---------------------------------------------------------------------------
# Probe state
# ---------------------------------------------------------------------------

@dataclass
class SitemapGroup:
    key: str
    children: int
    sampled: int = 0
    article_counts: list[int] = field(default_factory=list)
    leaf_counts: list[int] = field(default_factory=list)  # leaf-like URLs, used if the regex matches nothing
    failed: int = 0  # sampled children that did not load
    estimate: float | None = None
    exact: bool = False
    news: bool = False


@dataclass
class SiteResult:
    site_id: str
    name: str
    home_url: str
    genre: str
    article_re: str
    probed_at_utc: str = ""
    requests: int = 0
    robots_url: str = ""
    robots_status: int | None = None
    robots_snapshot: str | None = None
    robots_sha256: str | None = None
    crawl_delay_ua: float | None = None
    crawl_delay_star: float | None = None
    robots_eval: dict[str, Any] = field(default_factory=dict)
    sitemaps: list[dict[str, Any]] = field(default_factory=list)  # every sitemap fetched
    sitemap_groups: list[dict[str, Any]] = field(default_factory=list)
    estimated_articles: int | None = None
    estimate_exact: bool = False
    estimate_notes: list[str] = field(default_factory=list)
    article_urls_seen: int = 0
    nonmatching_url_samples: list[str] = field(default_factory=list)
    regex_fallback: bool = False
    articles: list[dict[str, Any]] = field(default_factory=list)
    median_bytes: float | None = None
    estimated_mb: float | None = None
    server_rendered: str = "unknown"
    terms_pages: list[dict[str, Any]] = field(default_factory=list)
    terms_snapshot: str | None = None
    clauses: list[dict[str, Any]] = field(default_factory=list)
    challenge: str | None = None
    errors: list[str] = field(default_factory=list)
    verdict: str = ""


class Budget:
    def __init__(self, client: ScrapeClient, limit: int) -> None:
        self.client, self.limit, self.start = client, limit, client.attempts

    @property
    def used(self) -> int:
        return self.client.attempts - self.start

    def left(self) -> int:
        return self.limit - self.used


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_robots(client: ScrapeClient, cand: Candidate, res: SiteResult, evidence_dir: Path, date: str) -> RobotsTxt:
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(cand.home_url))
    res.robots_url = origin + "/robots.txt"
    client.kind = "robots"
    robots = client.robots(cand.home_url)  # raises BotChallenge if robots.txt is a challenge page
    res.robots_status = robots.status
    path = evidence_dir / "robots" / f"{cand.site_id}_{date}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if robots.status is not None and robots.status < 400:
        data = robots.text.encode("utf-8")
        path.write_bytes(data)
        res.robots_sha256 = _sha256(data)
    else:
        why = (f"HTTP {robots.status}: no robots.txt, so no restrictions (RFC 9309 2.3.1.3)"
               if robots.status else "unreachable: treated as disallow-all (RFC 9309 2.3.1.4)")
        path.write_text(f"# {res.robots_url} fetched {res.probed_at_utc} by collect/scrape/probe.py: {why}\n",
                        encoding="utf-8")
    res.robots_snapshot = _rel(path)
    res.crawl_delay_ua = robots.crawl_delay(client.agent_token)
    res.crawl_delay_star = robots.crawl_delay("*")
    res.clauses = [c for c in res.clauses if c.get("page") != "robots.txt"] + robots_clauses(robots.text)
    return robots


def _fetch_sitemap(client: ScrapeClient, url: str, cand: Candidate, res: SiteResult,
                   log: logging.Logger) -> Sitemap | None:
    r = client.fetch(url, kind="sitemap", allow_url=lambda u: same_site(u, cand.domain))
    info: dict[str, Any] = {"url": url, "final_url": r.final_url, "outcome": r.outcome, "status": r.status}
    sm: Sitemap | None = None
    if r.ok:
        try:
            sm = parse_sitemap(r.body)
        except SitemapTooLarge as e:
            info["error"] = str(e)
        if sm is not None:
            info["kind"] = sm.kind
            info["entries"] = len(sm.entries)
    res.sitemaps.append(info)
    log.info("[%s] sitemap %s -> %s %s (%s, %s entries)", cand.site_id, url, r.outcome, r.status,
             info.get("kind"), info.get("entries"))
    return sm if sm is not None and sm.kind != "unknown" else None


@dataclass
class SitemapFindings:
    article_urls: list[str]  # matched the guessed regex
    leaf_urls: list[str]  # leaf-like URLs that did not match
    estimate: float | None  # article count from regex matches
    exact: bool
    leaf_estimate: float | None  # count of leaf-like URLs, used only if nothing matches the regex
    leaf_exact: bool
    notes: list[str] = field(default_factory=list)


def _group_totals(groups: Sequence[SitemapGroup], exact_counts: dict[str, int], news_exact: set[str],
                  leaf: bool) -> tuple[float | None, bool, list[str]]:
    """Sum the exact seed counts and each group's estimate. Unknown groups make the total inexact."""
    notes: list[str] = []
    counts = (lambda g: g.leaf_counts) if leaf else (lambda g: g.article_counts)
    non_news_positive = (any(sum(counts(g)) > 0 and not g.news for g in groups)
                         or any(n > 0 for u, n in exact_counts.items() if u not in news_exact))
    total, known, exact = 0.0, False, True
    for url, n in exact_counts.items():
        if url in news_exact and non_news_positive:
            notes.append(f"news sitemap {url} ({n} recent URLs) left out: its URLs repeat the archive sitemaps")
            continue
        total += n
        known = True
    for g in groups:
        c = counts(g)
        if not c:
            exact = False
            why = f"{g.failed} sample(s) failed to load" if g.failed else "not sampled (budget)"
            notes.append(f"{g.children} child sitemaps matching {g.key}: {why}; count unknown")
            continue
        g_exact = len(c) >= g.children
        estimate = sum(c) if g_exact else statistics.mean(c) * g.children
        exact = exact and g_exact
        if g.news and non_news_positive and len(groups) > 1:
            notes.append(f"news sitemaps {g.key} left out of the count")
            continue
        if _MEDIA_SITEMAP.search(g.key) and estimate:
            notes.append(f"media sitemaps {g.key} (~{estimate:,.0f} URLs) left out of the count")
            continue
        total += estimate
        known = True
    return (total if known else None), (exact and known), notes


def step_sitemaps(client: ScrapeClient, cand: Candidate, robots: RobotsTxt, res: SiteResult, budget: Budget,
                  reserve: int, rng: random.Random, log: logging.Logger) -> SitemapFindings:
    """Fetch seed sitemaps and a seeded sample of child sitemaps; count article and leaf-like URLs."""
    article_re = re.compile(cand.article_re)
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(cand.home_url))
    on_site = [u for u in robots.sitemaps if same_site(u, cand.domain)]
    sectioned = cand.path_prefix != "/"
    prefixed = [u for u in on_site if sectioned and cand.path_prefix in urlsplit(u).path]
    # A section of a shared host (bbc.com/urdu) skips the host's other sections' sitemaps when it has its own.
    others = on_site if not (sectioned and prefixed) else []
    seeds = list(dict.fromkeys([*cand.extra_sitemaps, *prefixed, *others,
                                origin + "/sitemap.xml", origin + "/sitemap_index.xml"]))
    notes: list[str] = []
    if len(on_site) < len(robots.sitemaps):
        notes.append(f"{len(robots.sitemaps) - len(on_site)} robots.txt sitemaps on other hosts ignored")

    article_urls: set[str] = set()
    leaf_urls: set[str] = set()
    other_urls: list[str] = []
    exact_counts: dict[str, int] = {}  # URL-list seeds counted in full: url -> regex matches
    exact_leaf: dict[str, int] = {}
    news_exact: set[str] = set()
    children: dict[str, str | None] = {}
    fetched: set[str] = set()

    def count(sm: Sitemap) -> tuple[int, int]:
        """(URLs matching the guessed regex, leaf-like URLs) in one URL-list sitemap."""
        n = leaves = 0
        for e in sm.entries:
            if article_re.match(e.loc) and same_site(e.loc, cand.domain, cand.path_prefix):
                article_urls.add(e.loc)
                n += 1
                continue
            if len(other_urls) < 2000:
                other_urls.append(e.loc)
            if is_leaf_like(e.loc, cand):
                leaves += 1
                if len(leaf_urls) < 100_000:
                    leaf_urls.add(e.loc)
        return n, leaves

    seed_budget = max(2, (budget.left() - reserve) // 2)
    for url in seeds:
        if budget.left() <= reserve or seed_budget <= 0:
            notes.append("request budget reached before all seed sitemaps were fetched")
            break
        seed_budget -= 1
        fetched.add(url)
        sm = _fetch_sitemap(client, url, cand, res, log)
        if sm is None:
            continue
        if sm.kind == "index":
            for e in sm.entries:
                if same_site(e.loc, cand.domain):
                    children.setdefault(e.loc, e.lastmod)
        else:
            exact_counts[url], exact_leaf[url] = count(sm)
            if _NEWS_SITEMAP.search(url):
                news_exact.add(url)
    for url in fetched:
        children.pop(url, None)

    members: dict[str, list[str]] = {}
    for url in children:
        members.setdefault(type_key(url), []).append(url)
    groups = {key: SitemapGroup(key, len(urls), news=bool(_NEWS_SITEMAP.search(key))) for key, urls in members.items()}
    in_section = (lambda key: cand.path_prefix.rstrip("/") + "/" in key) if sectioned else (lambda key: True)
    order = sorted(groups.values(), key=lambda g: (not in_section(g.key), bool(_UNLIKELY_SITEMAP.search(g.key)),
                                                   g.news, -g.children))
    pools = {g.key: rng.sample(members[g.key], len(members[g.key])) for g in order}

    for round_no in range(SAMPLES_PER_GROUP):
        for g in order:
            if budget.left() <= reserve:
                break
            tried = g.sampled + g.failed
            if tried > round_no or tried >= g.children:
                continue
            url = pools[g.key][tried]
            sm = _fetch_sitemap(client, url, cand, res, log)
            if sm is None:
                g.failed += 1  # a sitemap that will not load says nothing about the count
            elif sm.kind == "urlset":
                n, leaves = count(sm)
                g.sampled += 1
                g.article_counts.append(n)
                g.leaf_counts.append(leaves)
            else:  # nested index: sample one grandchild and scale by the number of grandchildren
                grandchildren = [e.loc for e in sm.entries if same_site(e.loc, cand.domain)]
                gsm = (_fetch_sitemap(client, rng.choice(grandchildren), cand, res, log)
                       if grandchildren and budget.left() > reserve else None)
                if gsm is not None and gsm.kind == "urlset":
                    n, leaves = count(gsm)
                    g.sampled += 1
                    g.article_counts.append(n * len(grandchildren))
                    g.leaf_counts.append(leaves * len(grandchildren))
                    notes.append(f"{url} is a nested index of {len(grandchildren)}; one sampled")
                else:
                    g.failed += 1

    estimate, exact, regex_notes = _group_totals(order, exact_counts, news_exact, leaf=False)
    leaf_estimate, leaf_exact, leaf_notes = _group_totals(order, exact_leaf, news_exact, leaf=True)
    res.sitemap_groups = [asdict(g) for g in order]
    res.article_urls_seen = len(article_urls)
    res.nonmatching_url_samples = sorted(set(other_urls))[:: max(1, len(set(other_urls)) // 10)][:10]
    return SitemapFindings(sorted(article_urls), sorted(leaf_urls), estimate if article_urls else None, exact,
                           leaf_estimate, leaf_exact, notes + (regex_notes if article_urls else leaf_notes))


_NUMERIC_ID = re.compile(r"/(\d{4,})(?:/|\.html|$)")


def is_leaf_like(url: str, cand: Candidate) -> bool:
    """A same-site URL with at least two path segments that is not a listing, tag, author or media page."""
    path = urlsplit(url).path
    return (same_site(url, cand.domain, cand.path_prefix) and not _NON_ARTICLE_PATH.search(path)
            and path.strip("/").count("/") >= 1)


def home_article_links(home_html: bytes, cand: Candidate) -> tuple[list[str], bool]:
    """Article-like links on the home page: (urls, matched the guessed regex). Used when sitemaps fail."""
    soup = soup_of(home_html)
    links = {urljoin(cand.home_url, a["href"].strip()).split("#", 1)[0] for a in soup.find_all("a", href=True)}
    links = {u for u in links if same_site(u, cand.domain, cand.path_prefix)}
    article_re = re.compile(cand.article_re)
    matched = sorted(u for u in links if article_re.match(u))
    if matched:
        return matched, True
    return sorted(u for u in links if is_leaf_like(u, cand)), False


def step_robots_eval(client: ScrapeClient, urls: Sequence[str], res: SiteResult, rng: random.Random) -> None:
    sample = rng.sample(list(urls), min(len(urls), 1000))
    ours = client.agent_token
    counts = {"rfc_ua": 0, "rfc_star": 0, "stdlib_ua": 0, "stdlib_star": 0}
    disagreements: list[dict[str, Any]] = []
    for url in sample:
        rfc_ua, std_ua = client.robots_verdicts(url)
        rfc_star, std_star = client.robots_verdicts(url, "*")
        for k, v in (("rfc_ua", rfc_ua), ("rfc_star", rfc_star), ("stdlib_ua", std_ua), ("stdlib_star", std_star)):
            counts[k] += v
        if (rfc_ua, rfc_star) != (std_ua, std_star) and len(disagreements) < 5:
            disagreements.append({"url": url, "rfc_ua": rfc_ua, "rfc_star": rfc_star,
                                  "stdlib_ua": std_ua, "stdlib_star": std_star})
    res.robots_eval = {"urls_tested": len(sample), "agent_token": ours, **counts, "disagreements": disagreements}


def render_class(html: bytes, body: str) -> str:
    """'server' if the article text is in the HTML, 'json' if only in a script blob, else 'none'."""
    if arabic_letters(body) >= 200:
        return "server"
    soup = soup_of(html)
    script_text = " ".join(s.get_text() for s in soup.find_all("script"))
    escaped = len(re.findall(r"\\u06[0-9a-fA-F]{2}", script_text))
    return "json" if arabic_letters(script_text) + escaped >= 1000 else "none"


def step_articles(client: ScrapeClient, cand: Candidate, urls: Sequence[str], n: int, res: SiteResult,
                  cache: HtmlCache, rng: random.Random, log: logging.Logger) -> None:
    for url in rng.sample(list(urls), min(n, len(urls))):
        r = client.fetch(url, kind="article", allow_url=lambda u: same_site(u, cand.domain))
        row: dict[str, Any] = {"url": url, "final_url": r.final_url, "outcome": r.outcome, "status": r.status,
                               "html_bytes": len(r.body)}
        if r.ok:
            cache.put(url, r.body)
            soup = soup_of(r.body)
            body = largest_text_block(soup)
            row.update({
                "title": page_title(soup), "published": published_date(soup),
                "body_bytes": len(body.encode("utf-8")), "arabic_letters": arabic_letters(body),
                "render": render_class(r.body, body), "html_sha256": _sha256(r.body),
            })
        res.articles.append(row)
        log.info("[%s] article %s -> %s %s %s B", cand.site_id, url, r.outcome, r.status, row.get("body_bytes"))
    with (cache.dir / "index.jsonl").open("a", encoding="utf-8") as f:
        for row in res.articles:
            if row["outcome"] == "ok":
                f.write(json.dumps({k: row[k] for k in ("url", "final_url", "status", "html_sha256")},
                                   ensure_ascii=False) + "\n")
    summarize_articles(res)


def step_terms(client: ScrapeClient, cand: Candidate, res: SiteResult, budget: Budget, evidence_dir: Path,
               date: str, home_html: bytes | None, cache: HtmlCache, log: logging.Logger) -> None:
    links: list[dict[str, Any]] = []
    if home_html:
        links = find_policy_links(soup_of(home_html), cand.home_url)
    if not links:
        first_ok = next((a for a in res.articles if a["outcome"] == "ok"), None)
        cached = cache.get(first_ok["url"]) if first_ok else None
        if cached:
            links = find_policy_links(soup_of(cached), first_ok["final_url"])
    chosen = [next((x for x in links if x["kind"] == "terms"), None),
              next((x for x in links if x["kind"] == "privacy"), None)]
    res.terms_pages = []
    res.clauses = [c for c in res.clauses if c.get("page") == "robots.txt"]
    res.errors = [e for e in res.errors if "terms" not in e and "privacy" not in e]
    sections: list[str] = []
    for link in [x for x in chosen if x is not None]:
        if budget.left() <= 0:
            res.errors.append("request budget reached before the privacy page")
            break
        r = client.fetch(link["url"], kind="terms")
        page = {**link, "final_url": r.final_url, "outcome": r.outcome, "status": r.status}
        if r.ok:
            text = page_text(r.body)
            page.update({"html_sha256": _sha256(r.body), "text_chars": len(text)})
            sections.append(f"===== {link['kind']}: {r.final_url} (HTTP {r.status}, fetched {r.fetched_at_utc}, "
                            f"html sha256 {page['html_sha256']}) =====\n{text}\n")
            for clause in find_clauses(text):
                res.clauses.append({**clause, "page": link["kind"], "url": r.final_url})
        res.terms_pages.append(page)
        log.info("[%s] %s page %s -> %s %s", cand.site_id, link["kind"], link["url"], r.outcome, r.status)
    if not links:
        res.errors.append("no terms or privacy link found on the home page or an article page")
    path = evidence_dir / "terms" / f"{cand.site_id}_{date}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Terms snapshot for {cand.name} ({cand.site_id}), collect/scrape/probe.py, {res.probed_at_utc}\n"
    if not sections:
        sections = ["(no terms or privacy page could be fetched; see evidence/probe_report.md)\n"]
    path.write_text(header + "\n".join(sections), encoding="utf-8")
    res.terms_snapshot = _rel(path)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def robots_summary(res: SiteResult) -> str:
    ev = res.robots_eval
    n = ev.get("urls_tested", 0)
    if not n:
        return "unknown (no article URLs)"
    ua = min(ev["rfc_ua"], ev["stdlib_ua"]) / n
    star = min(ev["rfc_star"], ev["stdlib_star"]) / n
    if ua == 1 and star == 1:
        return "yes"
    if ua == 0:
        return "no"
    return f"partly (UA {ua:.0%}, * {star:.0%})"


def decide(res: SiteResult) -> str:
    if res.challenge:
        return f"NO: bot challenge ({res.challenge})"
    restrictive = [c for c in res.clauses
                   if c["prohibitive"] and any(cat in RESTRICTIVE_CATEGORIES for cat in c["categories"])]
    if restrictive:
        cats = sorted({cat for c in restrictive for cat in c["categories"] if cat in RESTRICTIVE_CATEGORIES})
        where = sorted({"robots.txt comments" if c["page"] == "robots.txt" else f"{c['page']} page"
                        for c in restrictive})
        return f"NO: {' and '.join(where)} restrict {', '.join(cats)} (quotes below)"
    ok = [a for a in res.articles if a["outcome"] == "ok"]
    if not ok:
        reasons = sorted({a["outcome"] for a in res.articles}) or ["no article URLs found"]
        return "NO: no article page fetched (" + ", ".join(reasons) + ")"
    robots = robots_summary(res)
    if robots == "no":
        return "NO: robots.txt disallows article URLs for our user agent"
    if res.server_rendered == "none":
        return "NO: article text needs JavaScript"
    notes = []
    if robots != "yes":
        notes.append(f"robots {robots}")
    if res.server_rendered == "json":
        notes.append("text only in embedded JSON")
    if any("reproduction" in c["categories"] for c in res.clauses):
        notes.append("terms: reproduction clause only; we do not republish")
    if not res.terms_pages or not any(p["outcome"] == "ok" for p in res.terms_pages):
        notes.append("terms page not found")
    if res.estimated_mb is not None and res.estimated_mb < 20:
        notes.append(f"small (~{res.estimated_mb:.0f} MB)")
    return "OK" + (f" ({'; '.join(notes)})" if notes else "")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STEPS = ("sitemaps", "articles", "terms")


def _id_note(urls: Sequence[str], where: str) -> str | None:
    ids = [int(m.group(1)) for u in urls if (m := _NUMERIC_ID.search(urlsplit(u).path))]
    return (f"highest numeric ID among {where}: {max(ids):,} (an upper bound on items if the site numbers "
            "them sequentially)") if ids else None


def choose_article_urls(cand: Candidate, found: SitemapFindings | None, home_html: bytes | None,
                        res: SiteResult) -> list[str]:
    """Sample source, in order: sitemap URLs matching the regex, home links matching it,
    leaf-like sitemap URLs, leaf-like home links. Sets the count fields to match."""
    res.regex_fallback = False
    if found is None:  # articles re-sampled without the sitemaps step: drop notes from the old sample choice
        stale = ("leaf-like", "used as samples", "highest numeric ID", "article count is unknown")
        res.estimate_notes = [n for n in res.estimate_notes if not any(x in n for x in stale)]
    home_matched, home_leaf = ([], [])
    if home_html:
        links, matched = home_article_links(home_html, cand)
        home_matched, home_leaf = (links, []) if matched else ([], links)
    if found is not None:
        res.estimate_notes.extend(found.notes)
    if found is not None and found.article_urls:
        res.estimated_articles = round(found.estimate) if found.estimate is not None else None
        res.estimate_exact = found.exact
        urls, where = found.article_urls, "sitemap article URLs"
    elif home_matched:
        res.estimated_articles, res.estimate_exact = None, False
        if found is not None or (res.sitemaps and not res.article_urls_seen):
            res.estimate_notes.append("sitemaps list no URL matching the article regex (they hold other pages), "
                                      "so the article count is unknown")
        res.estimate_notes.append(f"{len(home_matched)} home-page links matching the article regex used as samples")
        urls, where = home_matched, "home-page article links"
    elif found is not None and found.leaf_urls:
        res.regex_fallback = True
        res.estimated_articles = round(found.leaf_estimate) if found.leaf_estimate is not None else None
        res.estimate_exact = found.leaf_exact
        res.estimate_notes.append("guessed article regex matched nothing in sitemaps or on the home page: counts are "
                                  "leaf-like sitemap URLs (not tag/category/author/search/feed/video pages), an upper "
                                  "bound for articles")
        urls, where = found.leaf_urls, "leaf-like sitemap URLs"
    elif home_leaf:
        res.regex_fallback = True
        res.estimated_articles, res.estimate_exact = None, False
        res.estimate_notes.append("guessed article regex matched nothing: leaf-like home-page links used as samples; "
                                  "article count unknown")
        urls, where = home_leaf, "leaf-like home-page links"
    else:
        return []
    note = _id_note(urls, where)
    if note:
        res.estimate_notes.append(note)
    return urls


def probe_site(cand: Candidate, args: argparse.Namespace, evidence: EvidenceLog, date: str,
               log: logging.Logger, client_kwargs: dict[str, Any] | None = None,
               earlier: SiteResult | None = None) -> SiteResult:
    """Probe one site.

    args.steps (default: all of STEPS) chooses what to redo. robots.txt is
    always fetched. With fewer than all steps, the site's earlier result from
    the same day is the starting point and only the chosen steps' fields are
    replaced. Without the sitemaps step, article samples come from the home
    page (the cached copy if there is one).
    """
    steps = set(getattr(args, "steps", None) or STEPS)
    partial = steps != set(STEPS)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if partial:
        if earlier is None:
            raise ValueError(f"--steps {' '.join(sorted(steps))} needs an earlier probe of {cand.site_id} "
                             "from the same day")
        res = earlier
        res.estimate_notes.append(f"re-run {now} with --steps {' '.join(s for s in STEPS if s in steps)}")
    else:
        res = SiteResult(cand.site_id, cand.name, cand.home_url, cand.genre, cand.article_re, probed_at_utc=now)
    evidence.extra = {"site": cand.site_id}
    client = ScrapeClient(args.user_agent, evidence=evidence, min_interval=args.min_interval, jitter=args.jitter,
                          max_retries=2, retry_after_cap=120.0, logger=log, **(client_kwargs or {}))
    budget = Budget(client, args.max_requests)
    rng = random.Random(f"{args.seed}:{cand.site_id}")
    cache = HtmlCache(args.data_dir / cand.site_id)
    try:
        robots = step_robots(client, cand, res, args.evidence_dir, date)
        home_html = cache.get(cand.home_url) if partial else None
        if home_html is None and (steps & {"sitemaps", "terms"} or partial):
            home = client.fetch(cand.home_url, kind="home", allow_url=lambda u: same_site(u, cand.domain))
            if home.ok:
                home_html = home.body
                cache.put(cand.home_url, home.body)
            else:
                res.errors.append(f"home page {cand.home_url}: {home.outcome} {home.status}")
        if "articles" in steps:
            found = None
            if "sitemaps" in steps:
                res.sitemaps, res.sitemap_groups, res.estimate_notes = [], [], []
                found = step_sitemaps(client, cand, robots, res, budget, args.articles + 3, rng, log)
            urls = choose_article_urls(cand, found, home_html, res)
            res.articles = []
            res.errors = [e for e in res.errors if "article URLs" not in e]
            if urls:
                step_robots_eval(client, urls, res, rng)
                step_articles(client, cand, urls, args.articles, res, cache, rng, log)
            else:
                res.errors.append("no article URLs found in sitemaps or on the home page")
        if "terms" in steps:
            step_terms(client, cand, res, budget, args.evidence_dir, date, home_html, cache, log)
    except BotChallenge as e:
        res.challenge = e.reason
        log.warning("[%s] stopped: %s", cand.site_id, e)
    except SiteBlocked as e:
        res.errors.append(f"stopped: {e}")
        log.warning("[%s] stopped: %s", cand.site_id, e)
    except Exception as e:  # one broken site must not lose the others' results
        res.errors.append(f"probe error: {type(e).__name__}: {e}")
        log.exception("[%s] probe error", cand.site_id)
    finally:
        client.session.close()
        res.requests = (res.requests + budget.used) if partial else budget.used
        summarize_articles(res)
        res.verdict = decide(res)
        log.info("[%s] %d requests; verdict: %s", cand.site_id, budget.used, res.verdict)
    return res


def _fmt_num(x: float | None, digits: int = 0) -> str:
    return "?" if x is None else f"{x:,.{digits}f}"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def terms_cell(res: SiteResult) -> str:
    if not res.clauses:
        return "none found" if any(p["outcome"] == "ok" for p in res.terms_pages) else "terms page not found"
    ranked = sorted(res.clauses, key=lambda c: (
        not (c["prohibitive"] and any(cat in RESTRICTIVE_CATEGORIES for cat in c["categories"])),
        "reproduction" in c["categories"] and len(c["categories"]) == 1, len(c["quote"])))
    best = ranked[0]
    quote = best["quote"] if len(best["quote"]) <= 220 else best["quote"][:219] + "…"
    more = f" (+{len(res.clauses) - 1} more)" if len(res.clauses) > 1 else ""
    return f"[{', '.join(best['categories'])}] “{_md(quote)}”{more}"


def summarize_articles(res: SiteResult) -> None:
    ok = [a for a in res.articles if a["outcome"] == "ok"]
    res.median_bytes = statistics.median(a["body_bytes"] for a in ok) if ok else None
    if ok:
        renders = [a["render"] for a in ok]
        res.server_rendered = max(set(renders), key=renders.count)
    res.estimated_mb = (res.estimated_articles * res.median_bytes / 1e6
                        if res.estimated_articles is not None and res.median_bytes is not None else None)


def refresh_from_snapshots(res: SiteResult, data_dir: Path) -> SiteResult:
    """Re-derive what the saved evidence allows, with no requests.

    robots.txt comment clauses come from the robots snapshot; article sizes
    and render classes are re-extracted from the cached sample pages with the
    current heuristic; then the verdict is decided again.
    """
    if res.robots_snapshot and (ROOT / res.robots_snapshot).exists():
        res.clauses = [c for c in res.clauses if c.get("page") != "robots.txt"]
        res.clauses[:0] = robots_clauses((ROOT / res.robots_snapshot).read_text(encoding="utf-8"))
    if not res.article_urls_seen and res.estimated_articles == 0 and not res.regex_fallback:
        res.estimated_articles, res.estimate_exact = None, False  # nothing matched the regex: unknown, not zero
    if res.estimated_articles and not any(n.startswith("media sitemaps") for n in res.estimate_notes):
        for g in res.sitemap_groups:  # results from before media sitemaps were left out of the count
            if _MEDIA_SITEMAP.search(g["key"]) and g.get("estimate"):
                res.estimated_articles = round(res.estimated_articles - g["estimate"])
                res.estimate_notes.append(f"media sitemaps {g['key']} (~{g['estimate']:,.0f} URLs) left out of the count")
    if res.sitemaps and not any(s.get("kind") == "urlset" for s in res.sitemaps) and res.estimated_articles == 0:
        res.estimated_articles, res.estimate_exact = None, False  # no URL list ever loaded: unknown, not zero
        if not any("no sitemap URL list loaded" in n for n in res.estimate_notes):
            res.estimate_notes.append("no sitemap URL list loaded (every child sitemap failed), so the count is unknown")
    cache = HtmlCache(data_dir / res.site_id)
    for a in res.articles:
        html = cache.get(a["url"]) if a["outcome"] == "ok" else None
        if html is not None:
            body = largest_text_block(soup_of(html))
            a.update({"body_bytes": len(body.encode("utf-8")), "arabic_letters": arabic_letters(body),
                      "render": render_class(html, body)})
    summarize_articles(res)
    res.verdict = decide(res)
    return res


def render_report(results: Sequence[SiteResult], args: argparse.Namespace, date: str, ua: str) -> str:
    lines = [
        "# Phase A probe report",
        "",
        f"Generated by `collect/scrape/probe.py` on {date} (UTC). User-Agent: `{ua.split('(')[0].strip()} (...)`. "
        f"Up to {args.max_requests} requests per site, at least {args.min_interval:g} s apart plus 0–{args.jitter:g} s "
        f"jitter, seed {args.seed}. MB = 10^6 bytes; KB = 10^3 bytes.",
        "",
        "Robots: a URL counts as allowed only if both our RFC 9309 matcher and `urllib.robotparser` allow it, "
        "for User-agent `*` and for our product token. Median KB is the UTF-8 size of the heuristic body "
        "(the container whose child elements hold the most Arabic-script text: usually `<p>`, sometimes `<span>` "
        "or `<h5>` blocks), before cleaning. Estimated MB = estimated articles x median bytes.",
        "",
        "| site | robots allows articles | crawl-delay | sitemaps found | est. articles | median KB/article "
        "| est. MB | server-rendered | terms clause | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        delay = r.crawl_delay_ua if r.crawl_delay_ua is not None else r.crawl_delay_star
        sitemaps_ok = [s for s in r.sitemaps if s.get("kind") in ("index", "urlset")]
        est = _fmt_num(r.estimated_articles) + ("" if r.estimate_exact or r.estimated_articles is None else " (est.)")
        lines.append("| " + " | ".join([
            f"{r.name} (`{r.site_id}`)",
            robots_summary(r) if not r.challenge else "n/a",
            "none" if delay is None else f"{delay:g} s",
            str(len(sitemaps_ok)),
            est,
            _fmt_num(r.median_bytes / 1000 if r.median_bytes is not None else None, 1),
            _fmt_num(r.estimated_mb),
            {"server": "yes", "json": "JSON only", "none": "no"}.get(r.server_rendered, r.server_rendered),
            terms_cell(r),
            _md(r.verdict),
        ]) + " |")
    lines += ["", "## Per-site details", ""]
    for r in results:
        lines += [f"### {r.name} (`{r.site_id}`)", "",
                  f"- Home: {r.home_url} · genre: {r.genre} · probed {r.probed_at_utc} · {r.requests} requests",
                  f"- robots.txt: {r.robots_url} → HTTP {r.robots_status}; snapshot `{r.robots_snapshot}`"
                  + (f" (sha256 `{r.robots_sha256[:16]}…`)" if r.robots_sha256 else ""),
                  f"- Crawl-delay: our UA {r.crawl_delay_ua}, `*` {r.crawl_delay_star}"]
        if r.challenge:
            lines.append(f"- **Bot challenge:** {r.challenge}. Probe stopped for this site.")
        ev = r.robots_eval
        if ev.get("urls_tested"):
            n = ev["urls_tested"]
            lines.append(f"- Robots on {n} sampled article URLs: RFC 9309 UA {ev['rfc_ua']}/{n}, `*` {ev['rfc_star']}/{n}; "
                         f"urllib.robotparser UA {ev['stdlib_ua']}/{n}, `*` {ev['stdlib_star']}/{n}")
            for d in ev["disagreements"]:
                lines.append(f"  - parsers disagree on {d['url']}: {d}")
        lines.append(f"- Guessed article regex: `{r.article_re}`" + (" (matched nothing; fallback used)"
                                                                       if r.regex_fallback else ""))
        if r.nonmatching_url_samples and (r.regex_fallback or not r.article_urls_seen):
            lines.append("- Sitemap URLs that did not match: " + ", ".join(r.nonmatching_url_samples[:5]))
        lines.append(f"- Sitemaps fetched ({len(r.sitemaps)}):")
        for s in r.sitemaps:
            lines.append(f"  - {s['url']} → {s['outcome']} {s['status'] or ''} {s.get('kind', '')} "
                         f"{str(s.get('entries', '')) + ' entries' if 'entries' in s else ''}".rstrip())
        if r.sitemap_groups:
            lines.append("- Child sitemap groups (digits masked as #): children, sampled, article URLs per sample, estimate")
            for g in r.sitemap_groups:
                lines.append(f"  - `{g['key']}`: {g['children']}, {g['sampled']}, {g['article_counts']}, "
                             f"{_fmt_num(g['estimate'])}{' (exact)' if g['exact'] else ''}{' (news)' if g['news'] else ''}")
        lines.append(f"- Estimated articles: {_fmt_num(r.estimated_articles)}"
                     f"{' (exact count)' if r.estimate_exact else ''}; article URLs seen: {r.article_urls_seen:,}")
        for note in r.estimate_notes:
            lines.append(f"  - {note}")
        if r.articles:
            lines.append("- Sample articles:")
            lines.append("")
            lines.append("  | URL | outcome | status | HTML KB | body KB | render | published |")
            lines.append("  |---|---|---|---|---|---|---|")
            for a in r.articles:
                lines.append(f"  | {a['url']} | {a['outcome']} | {a['status']} | {a['html_bytes'] / 1000:.0f} | "
                             f"{_fmt_num(a.get('body_bytes', 0) / 1000, 1)} | {a.get('render', '')} | "
                             f"{a.get('published') or ''} |")
            lines.append("")
        lines.append(f"- Terms snapshot: `{r.terms_snapshot}`")
        for p in r.terms_pages:
            lines.append(f"  - {p['kind']}: [{_md(p['text']) or p['url']}]({p['url']}) → {p['outcome']} {p['status']}")
        if r.clauses:
            lines.append("- Clauses found (sentence quotes; ⚠ = sentence also contains a prohibitive word):")
            for c in r.clauses:
                flag = "⚠ " if c["prohibitive"] else ""
                lines.append(f"  - {flag}[{', '.join(c['categories'])}] ({c['page']}) “{_md(c['quote'])}”")
        else:
            lines.append("- Clauses found: none")
        for e in r.errors:
            lines.append(f"- Error: {e}")
        lines.append(f"- **Verdict:** {r.verdict}")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", nargs="+", choices=CANDIDATE_IDS, default=list(CANDIDATE_IDS), metavar="SITE",
                   help=f"candidates to probe (default: all): {', '.join(CANDIDATE_IDS)}")
    p.add_argument("--max-requests", type=int, default=30, metavar="N", help="request budget per site (default 30)")
    p.add_argument("--articles", type=int, default=10, metavar="N", help="article pages to fetch per site (default 10)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--min-interval", type=float, default=1.0, metavar="SEC", help="minimum seconds between requests")
    p.add_argument("--jitter", type=float, default=0.5, metavar="SEC", help="extra random delay, 0 to SEC seconds")
    p.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS),
                   help="steps to (re)run; robots.txt is always fetched. Fewer than all keep the rest of the "
                        "site's results from earlier the same day (default: all)")
    p.add_argument("--report-only", action="store_true",
                   help="no requests: rebuild the report from the day's JSON, robots snapshots and cached pages")
    p.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR, help=argparse.SUPPRESS)
    p.add_argument("--data-dir", type=Path, default=PROBE_DATA_DIR, help=argparse.SUPPRESS)
    p.add_argument("--log-dir", type=Path, default=LOG_DIR, help=argparse.SUPPRESS)
    p.add_argument("--user-agent", default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("probe", args.log_dir)
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    try:
        ua = args.user_agent or default_user_agent()
    except MissingContact as e:
        log.error("%s", e)
        return 2
    args.user_agent = ua
    by_id = {c.site_id: c for c in CANDIDATES}
    json_path = args.evidence_dir / f"probe_{date}.json"
    earlier: dict[str, SiteResult] = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            earlier = {r["site_id"]: SiteResult(**r) for r in json.load(f)["results"]}
    results: list[SiteResult] = []
    log_path = args.evidence_dir / f"probe_log_{date}.jsonl.gz"
    if not args.report_only:
        with EvidenceLog(log_path) as evidence:
            for site_id in args.sites:
                results.append(probe_site(by_id[site_id], args, evidence, date, log,
                                          earlier=earlier.get(site_id)))
    # Sites not probed in this run keep their results from earlier the same day, refreshed offline.
    rerun = {r.site_id for r in results}
    results += [refresh_from_snapshots(r, args.data_dir) for sid, r in earlier.items() if sid not in rerun]
    results.sort(key=lambda r: CANDIDATE_IDS.index(r.site_id) if r.site_id in CANDIDATE_IDS else len(CANDIDATE_IDS))
    write_json(json_path, {"date": date, "seed": args.seed, "max_requests": args.max_requests,
                           "results": [asdict(r) for r in results]})
    report = args.evidence_dir / "probe_report.md"
    report.write_text(render_report(results, args, date, ua), encoding="utf-8")
    log.info("wrote %s, %s and %s", _rel(report), _rel(args.evidence_dir / f"probe_{date}.json"), _rel(log_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
