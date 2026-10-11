"""Sitemap parsing for the probe and the crawler.

parse_sitemap() reads a sitemap index (<sitemapindex>) or a URL list (<urlset>),
plain or gzip-compressed, and returns its <loc>/<lastmod> entries. Namespaces
are ignored, so Google News and image extensions parse like plain sitemaps.

Sitemaps come from the sites we crawl, so they are untrusted:
- gzip output is capped at MAX_SITEMAP_BYTES (the sitemap protocol allows 50 MB
  uncompressed), so a decompression bomb cannot fill memory;
- the XML parser never resolves entities and never touches the network;
- anything that is not a sitemap (an HTML error page served with status 200,
  broken XML) parses as kind "unknown" with no entries.
"""

import gzip
import io
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from lxml import etree

MAX_SITEMAP_BYTES = 60 * 1000 * 1000

_CORE_TAGS = frozenset({"loc", "lastmod", "changefreq", "priority"})
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, recover=True,
                          remove_comments=True, remove_pis=True)


@dataclass(frozen=True)
class Entry:
    loc: str
    lastmod: str | None = None
    extensions: frozenset[str] = frozenset()  # local names of extension children: news, image, video, ...


@dataclass
class Sitemap:
    kind: str  # "index", "urlset" or "unknown"
    entries: list[Entry] = field(default_factory=list)


class SitemapTooLarge(ValueError):
    pass


def _gunzip(data: bytes, limit: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        out = f.read(limit + 1)
    if len(out) > limit:
        raise SitemapTooLarge(f"sitemap decompresses to more than {limit} bytes")
    return out


def _local(tag: object) -> str:
    return etree.QName(tag).localname if isinstance(tag, str) else ""


def parse_sitemap(data: bytes, limit: int = MAX_SITEMAP_BYTES) -> Sitemap:
    """Parse sitemap bytes (gzip detected by its magic number, not the URL)."""
    if data[:2] == b"\x1f\x8b":
        data = _gunzip(data, limit)
    elif len(data) > limit:
        raise SitemapTooLarge(f"sitemap is larger than {limit} bytes")
    data = data.lstrip()
    if not data.startswith(b"<"):
        return Sitemap("unknown")
    try:
        root = etree.fromstring(data, parser=_PARSER)
    except etree.XMLSyntaxError:
        return Sitemap("unknown")
    if root is None:
        return Sitemap("unknown")
    kind = {"sitemapindex": "index", "urlset": "urlset"}.get(_local(root.tag))
    if kind is None:
        return Sitemap("unknown")
    child_tag = "sitemap" if kind == "index" else "url"
    entries = []
    for child in root:
        if _local(child.tag) != child_tag:
            continue
        loc = lastmod = None
        extensions = set()
        for el in child:
            name = _local(el.tag)
            if name == "loc" and el.text:
                loc = el.text.strip()
            elif name == "lastmod" and el.text:
                lastmod = el.text.strip()
            elif name and name not in _CORE_TAGS:
                extensions.add(name)
        if loc:
            entries.append(Entry(loc, lastmod, frozenset(extensions)))
    return Sitemap(kind, entries)


_DIGITS = re.compile(r"\d+")


def type_key(url: str) -> str:
    """Group key for child sitemaps: host + path with digit runs replaced by '#'.

    post-sitemap12.xml and post-sitemap13.xml share a key; tag-sitemap1.xml does not.
    The query string is kept (some sites page sitemaps with ?page=N), digits masked too.
    """
    parts = urlsplit(url)
    key = parts.netloc.lower() + _DIGITS.sub("#", parts.path)
    if parts.query:
        key += "?" + _DIGITS.sub("#", parts.query)
    return key
