"""VOA Urdu (urduvoa.com): news, 2008 to the 2025 shutdown. VOA's own text is public domain.

Page structure (Pangea CMS, checked on the probe's sample pages, 2026-10-11):
- Articles have <body class="pg-article ...">; the story is #article-content .wsw.
- Video, audio and listener-letter pages share the /a/<slug>/<id>.html URL form but have
  <body class="pg-media ...">, og:type video.other, and no .wsw: skipped.
- Older stories (about 2012-2014) have no <p> in .wsw: the text is <span>s separated by
  <br><br>, or a bare <div>. text_blocks() reads that loose text too (extractor v2).
- Bylines (ul.authors-highlights), share boxes and category links sit outside .wsw.
  Inside .wsw: embedded media (.wsw__embed), infographics, tweet snippets, "SEE ALSO" cards
  (.also-read in div[data-owner-ct]) and stand-alone datelines are dropped.
- Discovery: sitemap.xml lists sitemap_445_<n>.xml.gz (the archive, ~20,000 URLs each)
  plus videos, sections, latest and news sitemaps, which are not followed.
"""

import re
from typing import ClassVar

from bs4 import BeautifulSoup

from collect.scrape.sites.base import SiteAdapter


class VoaUrdu(SiteAdapter):
    site_id: ClassVar[str] = "voa_urdu"
    name: ClassVar[str] = "VOA Urdu"
    home_url: ClassVar[str] = "https://www.urduvoa.com/"
    domain: ClassVar[str] = "urduvoa.com"
    genre: ClassVar[str] = "news"
    extractor_version: ClassVar[str] = "voa_urdu/2"
    terms_url: ClassVar[str | None] = "https://www.urduvoa.com/p/7823.html"
    terms_summary: ClassVar[str] = (
        "VOA's terms (https://www.urduvoa.com/p/7823.html, copyright section): all text, audio and video produced "
        "only by VOA is in the public domain, with credit to VOA required. Third-party content (AFP, AP, Reuters) "
        "is licensed to VOA and may not be reproduced, republished or redistributed without the agencies' "
        "permission; storing it for personal or non-commercial purposes is allowed. robots.txt allows /a/ article "
        "URLs, no Crawl-delay. We keep the text inside the course and never redistribute it.")

    sitemap_urls: ClassVar[tuple[str, ...]] = ("https://www.urduvoa.com/sitemap.xml",)
    article_url_re: ClassVar[re.Pattern[str]] = re.compile(r"^https://www\.urduvoa\.com/a/(?:[^/?#]+/)?\d+\.html$")

    container_selectors: ClassVar[tuple[str, ...]] = ("#article-content .wsw",)
    drop_selectors: ClassVar[tuple[str, ...]] = (
        ".wsw__embed", ".infgraphicsAttach", ".twitterSnippet", ".snippetLoading", ".media-download", ".c-mmp",
        ".also-read", "div[data-owner-ct]",  # "SEE ALSO" cards linking other stories
        ":scope > .dateline",  # a dateline standing alone before the first paragraph ("واشنگٹن —")
    )
    title_selector: ClassVar[str | None] = "h1.pg-title"

    _ARCHIVE_SITEMAP = re.compile(r"/sitemap_\d+_\d+\.xml\.gz$")

    def follow_sitemap(self, url: str) -> bool:
        return bool(self._ARCHIVE_SITEMAP.search(url))

    def skip_reason(self, soup: BeautifulSoup, url: str) -> str | None:
        body_classes = soup.body.get("class", []) if soup.body else []
        if "pg-media" in body_classes:
            return "media page (video/audio)"
        og = soup.find("meta", property="og:type")
        if og is not None and og.get("content", "").startswith(("video", "music")):
            return "media page (video/audio)"
        if "pg-article" not in body_classes:
            return "not an article page"
        return None
