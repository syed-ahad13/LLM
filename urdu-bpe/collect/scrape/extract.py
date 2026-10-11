"""HTML-to-text helpers shared by the probe and the site adapters.

Text is kept as the page has it. The only change made here is whitespace inside
a block: runs of HTML whitespace become one space, and <br> becomes a line break.
Unicode normalization belongs to normalize.py (CLAUDE.md, hard rule 4).

- soup_of: parse HTML with lxml through BeautifulSoup.
- block_text / text_blocks: the text of <p>, <h2>, <h3>, <li> elements in
  document order, plus loose text outside them (runs of text and inline
  elements, broken at <br> and block edges). A block nested inside another
  selected block is not counted twice. Inline tags are joined without a
  separator, so <p>اردو<b>زبان</b></p> stays one word as written.
- drop_elements: remove junk (GENERIC_DROP_SELECTORS plus site selectors).
- largest_text_block: the probe's site-independent body heuristic.
- published_date, canonical_url, page_title: metadata from meta tags and JSON-LD.
  JSON-LD is read only for dates. Its author fields are never used.
- drop_lines: remove lines holding an email address or matching a site's
  byline patterns (CLAUDE.md, hard rule 5).
"""

import json
import re
from collections.abc import Iterable, Iterator, Sequence
from typing import Any
from urllib.parse import urljoin

import regex
from bs4 import BeautifulSoup, Comment, Tag

BLOCK_TAGS: tuple[str, ...] = ("p", "h2", "h3", "li")

# Removed from every article container before blocks are collected. Sites add their own.
GENERIC_DROP_SELECTORS: tuple[str, ...] = (
    "script", "style", "noscript", "template", "nav", "aside", "footer", "form", "iframe",
    "figure", "figcaption", "button", "svg", "video", "audio", "picture", "object", "embed",
    "[class*=share]", "[class*=social]", "[class*=related]", "[class*=newsletter]", "[class*=subscribe]",
    "[class*=comment]", "[id*=comment]", "[class*=author]", "[class*=byline]", "[rel=author]",
    "[class*=read-more]", "[class*=readmore]", "[class*=also-read]", "[class*=read-also]", "[class*=advert]", "[id*=advert]", "[class*=ad-slot]",
)

ARABIC_LETTER = regex.compile(r"[\p{Arabic}&&\p{L}]", regex.V1)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_WS = re.compile(r"\s+")
_BR = "\x00"  # <br> marker; HTML parsers turn a literal NUL into U+FFFD, so it cannot occur in text


def soup_of(html: str | bytes) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def arabic_letters(text: str) -> int:
    """Number of Arabic-script letters (Urdu, Arabic, Persian) in text."""
    return len(ARABIC_LETTER.findall(text))


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _text_with_breaks(el: Tag) -> list[str]:
    """Text of el with <br> as line breaks: one string per non-empty line."""
    parts: list[str] = []
    for node in el.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, Tag):
            if node.name == "br":
                parts.append(_BR)
            continue
        if isinstance(node, str):
            parts.append(str(node))
    return [line for line in (collapse_ws(x) for x in "".join(parts).split(_BR)) if line]


def block_text(el: Tag) -> str:
    """Text of one block element: <br> splits lines, everything else joins as written."""
    return "\n".join(_text_with_breaks(el))


def _nested_in_block(el: Tag, container: Tag, tags: Sequence[str]) -> bool:
    parent = el.parent
    while parent is not None and parent is not container:
        if parent.name in tags:
            return True
        parent = parent.parent
    return False


# Inline elements: their text joins the surrounding run of loose text.
INLINE_TAGS = frozenset({"a", "abbr", "b", "bdi", "bdo", "cite", "code", "em", "font", "i", "kbd", "mark", "q", "s",
                         "samp", "small", "span", "strong", "sub", "sup", "time", "u", "var", "wbr"})


def _raw_text(el: Tag) -> str:
    """Text of an inline element as written, with <br> as the line marker (whitespace not collapsed)."""
    parts = []
    for node in el.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, Tag):
            if node.name == "br":
                parts.append(_BR)
        elif isinstance(node, str):
            parts.append(str(node))
    return "".join(parts)


def text_blocks(container: Tag, tags: Sequence[str] = BLOCK_TAGS) -> list[str]:
    """Text blocks of a container in document order.

    Each element in `tags` (outermost only) is one block. Text outside those
    elements also counts: runs of text nodes and inline elements (span, b, a,
    ...) become blocks, broken at <br> and at the edges of block-level elements.
    Some pages carry the story as <span>s separated by <br><br> or as bare text
    in a <div>, with no <p> at all.
    """
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        for line in "".join(run).split(_BR):
            line = collapse_ws(line)
            if line:
                out.append(line)
        run.clear()

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if not isinstance(child, Tag):
                if isinstance(child, str):
                    run.append(str(child))
                continue
            if child.name in tags:
                flush()
                text = block_text(child)
                if text:
                    out.append(text)
            elif child.name == "br":
                run.append(_BR)
            elif child.name in INLINE_TAGS and child.find(lambda t: t.name not in INLINE_TAGS and t.name != "br") is None:
                run.append(_raw_text(child))
            else:  # block-level element (div, ul, blockquote, table, h4, ...) or inline wrapping blocks
                flush()
                walk(child)
                flush()

    walk(container)
    flush()
    return out


def drop_elements(container: Tag, selectors: Iterable[str]) -> int:
    """Remove every element matching any CSS selector (and comments). Returns elements removed."""
    removed = 0
    for comment in container.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for selector in selectors:
        for el in container.select(selector):
            if el.decomposed:
                continue
            el.decompose()
            removed += 1
    return removed


def drop_lines(lines: Iterable[str], patterns: Sequence[re.Pattern[str]] = ()) -> tuple[list[str], int]:
    """Drop lines that contain an email address or match a pattern. Returns (kept, dropped count)."""
    kept, dropped = [], 0
    for line in lines:
        if EMAIL_RE.search(line) or any(p.search(line) for p in patterns):
            dropped += 1
        else:
            kept.append(line)
    return kept, dropped


# ---------------------------------------------------------------------------
# Probe heuristic
# ---------------------------------------------------------------------------

# Elements whose own text is not body text: link lists (related stories, menus) and form controls.
_NOT_BODY_TEXT = frozenset({"a", "button", "option", "label", "title", "html", "head", "body", "select"})


def largest_text_block(soup: BeautifulSoup) -> str:
    """Body text by a site-independent heuristic, for the probe.

    Every element holding Arabic-script text of its own (not inside a child
    element) scores its letter count for its parent and three quarters of it
    for its grandparent, so pages that wrap each paragraph in its own <div>
    still pick the story container. Links and form controls do not score, so
    related-story lists lose. The best-scoring element's text blocks are
    returned in document order, one per line (<br> also breaks lines).
    Sites use <p> mostly, but not always: <span class="story-text"> with <br>,
    or <h5> paragraphs.
    """
    work = soup_of(str(soup))  # never mutate the caller's tree
    drop_elements(work, ("script", "style", "noscript", "template"))
    holders: list[Tag] = []
    scores: dict[int, float] = {}
    elements: dict[int, Tag] = {}
    for el in work.find_all(True):
        if el.name in _NOT_BODY_TEXT:
            continue
        n = sum(arabic_letters(str(t)) for t in el.find_all(string=True, recursive=False)
                if not isinstance(t, Comment))
        if not n:
            continue
        holders.append(el)
        for ancestor, weight in ((el.parent, 1.0), (el.parent.parent if el.parent else None, 0.75)):
            if isinstance(ancestor, Tag):
                scores[id(ancestor)] = scores.get(id(ancestor), 0.0) + n * weight
                elements[id(ancestor)] = ancestor
    if not scores:
        return ""
    best = elements[max(scores, key=lambda k: scores[k])]
    inside = [el for el in holders if best in el.parents]
    chosen = {id(el) for el in inside}
    lines = []
    for el in inside:
        if any(id(a) in chosen for a in el.parents if a is not best and best in a.parents):
            continue  # its text is already part of an outer holder's block
        text = block_text(el)
        if arabic_letters(text):
            lines.append(text)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def meta_content(soup: BeautifulSoup, *keys: str) -> str | None:
    """content= of the first <meta> whose property, name or itemprop equals one of keys (in order)."""
    for key in keys:
        for attr in ("property", "name", "itemprop"):
            el = soup.find("meta", attrs={attr: key})
            if el is not None and el.get("content", "").strip():
                return el["content"].strip()
    return None


def _walk_json(obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_json(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_json(item)


def json_ld(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """Every JSON object (nested ones and @graph members included) from ld+json scripts."""
    out: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        try:
            data = json.loads(script.string or script.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        out.extend(_walk_json(data))
    return out


def published_date(soup: BeautifulSoup) -> str | None:
    """article:published_time (meta), else the first JSON-LD datePublished, as written by the site."""
    value = meta_content(soup, "article:published_time", "datePublished", "pubdate")
    if value:
        return value
    for obj in json_ld(soup):
        date = obj.get("datePublished")
        if isinstance(date, str) and date.strip():
            return date.strip()
    return None


def canonical_url(soup: BeautifulSoup, page_url: str) -> str | None:
    link = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    if link is not None and link.get("href", "").strip():
        return urljoin(page_url, link["href"].strip())
    og = meta_content(soup, "og:url")
    return urljoin(page_url, og) if og else None


def page_title(soup: BeautifulSoup, container: Tag | None = None) -> str:
    """The article's <h1> (inside container if given, else the page's first), else og:title, else <title>."""
    for scope in ([container] if container is not None else []) + [soup]:
        h1 = scope.find("h1")
        if h1 is not None:
            title = collapse_ws(h1.get_text())
            if title:
                return title
    og = meta_content(soup, "og:title")
    if og:
        return collapse_ws(og)
    return collapse_ws(soup.title.get_text()) if soup.title else ""
