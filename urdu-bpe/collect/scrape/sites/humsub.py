"""Hum Sub (humsub.com.pk): Urdu opinion, columns and blogs, plus a newsdesk; archive from 2016.

Page structure (WordPress, checked on the probe's sample pages, 2026-10-11):
- The story is article.post .entry-content; the title is h1.entry-title.
- Inside .entry-content: share buttons (heateor), AdSense slots, ad-inserter blocks,
  a comments widget (#wpdevar_comment_*, .fb-comments) and the page links of multi-page
  posts (.pagination, plus the "press the next-page button" line) are dropped. Only
  page 1 of a multi-page post is collected (about 1% of posts).
- Bylines (span.byline, meta author) sit outside .entry-content and are never read.
- Reprints are skipped: posts in the bbc, dw or rsstest categories (feeds imported from
  other outlets) or credited to BBC/DW as author. BBC's and DW's robots.txt forbid
  scraping, dataset creation and AI training on their content wherever it appears.
  2 of the probe's 10 sample posts were BBC reprints.
- robots.txt: Crawl-delay 20 for User-agent *. Discovery: sitemap_index.xml lists
  post-sitemap<n>.xml (1,000 posts each); author, tag, category and page sitemaps
  are not followed.
"""

import re
from typing import ClassVar

from bs4 import BeautifulSoup

from collect.scrape.sites.base import SiteAdapter

REPRINT_CATEGORY = re.compile(r"^category-(?:bbc|dw|rsstest)(?:-|$)")
REPRINT_AUTHORS = frozenset({"بی بی سی", "بی بی سی اردو", "ڈی ڈبلیو", "ڈوئچے ویلے"})


class HumSub(SiteAdapter):
    site_id: ClassVar[str] = "humsub"
    name: ClassVar[str] = "Hum Sub"
    home_url: ClassVar[str] = "https://www.humsub.com.pk/"
    domain: ClassVar[str] = "humsub.com.pk"
    genre: ClassVar[str] = "opinion/blogs"
    extractor_version: ClassVar[str] = "humsub/3"
    terms_url: ClassVar[str | None] = None
    terms_summary: ClassVar[str] = (
        "No terms of use or privacy page is linked from the home page or article pages (probe, 2026-10-11). "
        "robots.txt allows article URLs for User-agent * with Crawl-delay: 20. Posts are the authors' and Hum Sub's "
        "copyright; we keep the text inside the course and never redistribute it. Reprints from BBC and DW "
        "(whose robots.txt forbid scraping, dataset creation and AI training) and RSS-imported posts are skipped.")

    sitemap_urls: ClassVar[tuple[str, ...]] = ("https://www.humsub.com.pk/sitemap_index.xml",)
    article_url_re: ClassVar[re.Pattern[str]] = re.compile(
        r"^https://(?:www\.)?humsub\.com\.pk/(?:n/\d+/\d{4}/\d{2}/\d{2}/[^/?#]+|\d+/[^/?#]+)/$")

    container_selectors: ClassVar[tuple[str, ...]] = ("article.post .entry-content",)
    drop_selectors: ClassVar[tuple[str, ...]] = (
        ".heateor_sss_sharing_container", ".heateorSssClear", "ins.adsbygoogle", ".ai-viewports", "[class*=ai-insert]",
        ".fb-comments", "[id^=wpdevar_comment]", ".wp-caption", ".sharedaddy", ".jp-relatedposts", "[class*=code-block]",
        ".pagination", ".page-links",
    )
    # "Press the next-page button to read more" (دبائیں / دبائیے) above the page links of multi-page posts.
    drop_line_res: ClassVar[tuple[re.Pattern[str], ...]] = (re.compile(r"اگلا صفحہ کا بٹن دبائی"),)
    title_selector: ClassVar[str | None] = "h1.entry-title"

    _POST_SITEMAP = re.compile(r"/post-sitemap\d*\.xml$")

    def follow_sitemap(self, url: str) -> bool:
        return bool(self._POST_SITEMAP.search(url))

    def skip_reason(self, soup: BeautifulSoup, url: str) -> str | None:
        post = soup.select_one("article.post")
        if post is None:
            return "not a post page"
        reprint = [c for c in post.get("class", []) if REPRINT_CATEGORY.match(c)]
        if reprint:
            return f"reprint ({reprint[0]})"
        author = soup.find("meta", attrs={"name": "author"})
        if author is not None and author.get("content", "").strip() in REPRINT_AUTHORS:
            return "reprint (credited to BBC/DW)"
        return None
