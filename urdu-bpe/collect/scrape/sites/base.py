"""Per-site adapter interface for the crawler (collect/scrape/crawl.py).

An adapter is mostly settings: where the sitemaps are, which URLs are
articles, which element holds the story and what junk to drop. The base class
does the extraction:

1. skip_reason(): pages that are not articles (media pages, reprints from
   sources whose terms forbid our use) raise NotArticle.
2. Title, publication date (meta article:published_time, else JSON-LD
   datePublished) and canonical URL from the page head.
3. The first element matching container_selectors is the story. The generic
   junk list (collect/scrape/extract.py) and the site's drop_selectors are
   removed from it: scripts, styles, nav, aside, footer, figures and captions,
   related-article boxes, share widgets, read-more lists, newsletter, author
   and comment boxes.
4. Its <p>, <h2>, <h3>, <li> blocks are collected in document order. Lines
   with an email address or matching drop_line_res (bylines) are dropped.
5. Too little Arabic-script text left (min_body_letters) raises NotArticle.

The document text is title + "\\n" + body.
"""

import re
from dataclasses import dataclass
from typing import ClassVar

from bs4 import BeautifulSoup

from collect.scrape.extract import (
    BLOCK_TAGS, GENERIC_DROP_SELECTORS, arabic_letters, canonical_url, drop_elements, drop_lines, page_title,
    published_date, soup_of, text_blocks,
)
from collect.scrape.fetch import same_site
from collect.scrape.sitemaps import Entry


class NotArticle(Exception):
    """The page is not an article we keep. The message is the reason, counted in the crawl report."""


@dataclass(frozen=True)
class Extracted:
    title: str
    body: str  # blocks joined with "\n"
    published: str | None
    canonical_url: str | None
    dropped_lines: int = 0  # lines removed for holding an email address or matching a byline pattern

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.body}" if self.title else self.body


class SiteAdapter:
    # identity: paths, sources.json, report
    site_id: ClassVar[str]  # data/raw/own_scrape/<site_id>/, evidence/crawl_log_<site_id>.jsonl.gz
    name: ClassVar[str]
    home_url: ClassVar[str]
    domain: ClassVar[str]  # registrable domain; every URL we request must be on it
    path_prefix: ClassVar[str] = "/"
    genre: ClassVar[str]
    extractor_version: ClassVar[str]  # bump whenever extraction changes, then run crawl.py --reextract
    terms_url: ClassVar[str | None] = None
    terms_summary: ClassVar[str]  # goes into the licence field of sources.json

    # discovery
    sitemap_urls: ClassVar[tuple[str, ...]] = ()  # entry points besides robots.txt Sitemap lines
    article_url_re: ClassVar[re.Pattern[str]]  # only matching URLs are fetched as articles
    # Never fetched, whatever else matches: search, tag, author/user/profile, comment, login, feed endpoints.
    blocked_url_re: ClassVar[re.Pattern[str]] = re.compile(
        r"/(?:search|tag|tags|author|authors|user|users|profile|comments?|login|signin|register|account|feed)"
        r"(?:/|$|\?)|[?&](?:s|q|replytocom|share|print)=", re.I)

    # extraction
    container_selectors: ClassVar[tuple[str, ...]]
    drop_selectors: ClassVar[tuple[str, ...]] = ()
    drop_line_res: ClassVar[tuple[re.Pattern[str], ...]] = ()
    block_tags: ClassVar[tuple[str, ...]] = BLOCK_TAGS
    title_selector: ClassVar[str | None] = None  # default: the page's first <h1>, else og:title, else <title>
    min_body_letters: ClassVar[int] = 50

    # -- discovery -------------------------------------------------------------

    def follow_sitemap(self, url: str) -> bool:
        """Whether to fetch this child of a sitemap index (e.g. post sitemaps yes, tag sitemaps no)."""
        return True

    def keep_entry(self, entry: Entry) -> bool:
        """Whether a URL-list sitemap entry is worth fetching (e.g. not a video entry)."""
        return True

    def normalize_url(self, url: str) -> str:
        """The form of a sitemap URL to request and store (e.g. http -> https). Default: unchanged."""
        return url

    def allow_url(self, url: str) -> bool:
        """Any URL we request (sitemaps, articles, redirect targets) must pass this."""
        return same_site(url, self.domain) and not self.blocked_url_re.search(url)

    def is_article_url(self, url: str) -> bool:
        return (bool(self.article_url_re.match(url)) and self.allow_url(url)
                and same_site(url, self.domain, self.path_prefix))

    # -- extraction ------------------------------------------------------------

    def skip_reason(self, soup: BeautifulSoup, url: str) -> str | None:
        """A reason to skip this page (not an article, not ours to take), or None."""
        return None

    def title(self, soup: BeautifulSoup) -> str:
        if self.title_selector:
            el = soup.select_one(self.title_selector)
            if el is not None and el.get_text(strip=True):
                return " ".join(el.get_text().split())
        return page_title(soup)

    def extract(self, html: bytes | str, url: str) -> Extracted:
        soup = soup_of(html)
        reason = self.skip_reason(soup, url)
        if reason:
            raise NotArticle(reason)
        container = next((c for sel in self.container_selectors if (c := soup.select_one(sel)) is not None), None)
        if container is None:
            raise NotArticle("no article container")
        title = self.title(soup)
        published = published_date(soup)
        canonical = canonical_url(soup, url)
        drop_elements(container, GENERIC_DROP_SELECTORS + self.drop_selectors)
        lines = [line for block in text_blocks(container, self.block_tags) for line in block.split("\n")]
        lines, dropped = drop_lines(lines, self.drop_line_res)
        body = "\n".join(lines)
        if arabic_letters(body) < self.min_body_letters:
            raise NotArticle("too little text")
        return Extracted(title, body, published, canonical, dropped)
