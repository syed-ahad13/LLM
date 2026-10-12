"""Wikitext -> plain text, written from scratch (no wiki parser library).

strip_wikitext(text, ns) applies these rules in this order:
 1. Remove <!-- comments --> (an unclosed one runs to the end).
 2. Remove <ref .../>, <ref ...>...</ref> and <references/> / <references>...</references>.
 3. Drop the whole content of DROP_CONTENT_TAGS (math, chem, gallery, timeline, ...).
 4. Cut the article at the first back-matter heading (BACK_MATTER: references,
    notes, see also, external links, further reading, bibliography; Urdu and English).
 5. Remove {{templates}}, {{#parser functions}} and {| tables |} with one
    depth scanner, since each can nest inside the other. An opener that is
    never closed is treated as plain text (the rescan drops only its braces).
 6. [http://... label] -> label; bracketed links without a label and bare URLs go.
 7. Internal links, innermost first: file, category and interlanguage links are
    dropped whole (so a caption's own links go with them); [[target|label]] ->
    label; [[target]] -> target without #anchor. Letters glued after "]]" stay glued.
 8. Remove '' and ''' emphasis; "== Heading ==" -> the heading text on its own line.
 9. Remove magic words (__TOC__ ...), turn <br> into a line break, remove other
    HTML tags but keep their inner text, html.unescape entities.
10. Per line: strip list markers (* # : ;) at the start, drop table/template
    residue lines starting with "|", drop lines left empty or without a letter or digit.

Disambiguation pages are detected on the raw wikitext (disambiguation_reason).
"""

import html
import re
from dataclasses import dataclass, field

import regex

DROP_CONTENT_TAGS = (
    "math", "chem", "ce", "score", "gallery", "timeline", "syntaxhighlight", "source", "code", "pre",
    "imagemap", "graph", "mapframe", "maplink", "templatedata", "hiero", "categorytree", "inputbox", "indicator",
)

# Namespace names that are always valid, besides the localized ones in the dump's <siteinfo>.
FILE_ALIASES = ("file", "image", "media", "فائل", "تصویر", "ملف")
CATEGORY_ALIASES = ("category", "زمرہ")
TEMPLATE_ALIASES = ("template", "سانچہ")

BACK_MATTER_HEADINGS = (
    "حوالہ جات", "حوالے", "حواشی", "مزید دیکھیے", "مزید دیکھیں", "بیرونی روابط", "مزید پڑھیے", "مزید پڑھیں", "کتابیات",
    "references", "notes", "footnotes", "see also", "external links", "further reading", "bibliography",
)

DISAMBIG_TEMPLATES = ("ضد ابہام", "disambiguation", "disambig", "dab", "dmbox")
DISAMBIG_MARKER = "ضد ابہام"

# Code points that are often typed for one another; folded only to compare names, never in output text.
_FOLD = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ـ": None, "‌": None, "_": " "})
_SPACES = re.compile(r"\s+")


def fold(name: str) -> str:
    """Comparison key for namespace names, template names and headings."""
    return _SPACES.sub(" ", name.translate(_FOLD)).strip().casefold()


BACK_MATTER = frozenset(fold(h) for h in BACK_MATTER_HEADINGS)
_DISAMBIG = frozenset(fold(t) for t in DISAMBIG_TEMPLATES)


@dataclass(frozen=True)
class Namespaces:
    """Folded namespace names: file-like, category, template, and every known name."""

    file: frozenset[str] = frozenset(fold(n) for n in FILE_ALIASES)
    category: frozenset[str] = frozenset(fold(n) for n in CATEGORY_ALIASES)
    template: frozenset[str] = frozenset(fold(n) for n in TEMPLATE_ALIASES)
    names: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_siteinfo(cls, namespaces: dict[int, str]) -> "Namespaces":
        """Build from <siteinfo><namespaces> (key -> localized name) plus the built-in aliases."""
        file = {fold(n) for n in FILE_ALIASES} | {fold(namespaces[k]) for k in (6, -2) if namespaces.get(k)}
        category = {fold(n) for n in CATEGORY_ALIASES} | ({fold(namespaces[14])} if namespaces.get(14) else set())
        template = {fold(n) for n in TEMPLATE_ALIASES} | ({fold(namespaces[10])} if namespaces.get(10) else set())
        names = {fold(n) for n in namespaces.values() if n} | file | category | template
        return cls(frozenset(file), frozenset(category), frozenset(template), frozenset(names))

    def known(self, folded: str) -> bool:
        return folded in self.names or folded in self.file or folded in self.category or folded in self.template


DEFAULT_NAMESPACES = Namespaces.from_siteinfo({})

# ---------------------------------------------------------------------------
# Rules 1-3: comments, references, dropped tags
# ---------------------------------------------------------------------------

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.S)
_REF_SELF = re.compile(r"<ref\b[^>]*/\s*>", re.I)
_REF_PAIR = re.compile(r"<ref(?:\s[^>]*)?(?<!/)>.*?</ref\s*>", re.I | re.S)
_REFERENCES = re.compile(r"<references\b[^>]*/\s*>|<references(?:\s[^>]*)?(?<!/)>.*?</references\s*>", re.I | re.S)
_DROP_TAGS = "|".join(DROP_CONTENT_TAGS)
_DROP_PAIR = re.compile(rf"<({_DROP_TAGS})(?:\s[^>]*)?(?<!/)>.*?</\1\s*>", re.I | re.S)
_DROP_SELF = re.compile(rf"<(?:{_DROP_TAGS})\b[^>]*/\s*>", re.I)


def remove_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def remove_refs(text: str) -> str:
    text = _REF_SELF.sub("", text)
    text = _REF_PAIR.sub("", text)
    return _REFERENCES.sub("", text)


def remove_dropped_tags(text: str) -> str:
    return _DROP_SELF.sub("", _DROP_PAIR.sub("", text))


# ---------------------------------------------------------------------------
# Rule 4: back matter
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^[ \t]*(={1,6})[ \t]*(.+?)[ \t]*\1[ \t]*$", re.M)


def cut_back_matter(text: str) -> str:
    """Everything before the first heading that names back matter (references, see also, ...)."""
    for m in _HEADING.finditer(text):
        if fold(m.group(2).replace("'", "")) in BACK_MATTER:
            return text[: m.start()]
    return text


# ---------------------------------------------------------------------------
# Rule 5: templates and tables
# ---------------------------------------------------------------------------

_BRACE_TOKEN = re.compile(r"\{\{|\}\}|^[ \t]*\{\||^[ \t]*\|\}", re.M)


def _brace_spans(text: str, literal: set[int]) -> tuple[list[tuple[int, int]], int | None]:
    """(spans to delete, start of the outermost unclosed opener or None). Openers in `literal` are text."""
    spans: list[tuple[int, int]] = []
    stack: list[tuple[str, int]] = []  # (kind, start); kind is "t" (template) or "table"
    for m in _BRACE_TOKEN.finditer(text):
        tok = m.group().lstrip(" \t")
        if tok == "{{" or tok == "{|":
            if m.start() in literal:
                continue
            stack.append(("t" if tok == "{{" else "table", m.start()))
            continue
        kind = "t" if tok == "}}" else "table"
        if not any(k == kind for k, _ in stack):
            continue  # a stray closer is plain text
        while True:
            k, start = stack.pop()
            if k == kind:
                break
        if not stack:
            spans.append((start, m.end()))
    return spans, (stack[0][1] if stack else None)


def remove_templates(text: str) -> str:
    """Remove {{...}} and {| ... |} with nesting. Unclosed openers are kept as text."""
    if "{{" not in text and "{|" not in text:
        return text
    literal: set[int] = set()
    while True:
        spans, unclosed = _brace_spans(text, literal)
        if unclosed is None:
            break
        literal.add(unclosed)
    out, pos = [], 0
    for start, end in spans:
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Rules 6-7: links
# ---------------------------------------------------------------------------

_EXT_LINK = re.compile(r"\[(?:(?:https?|ftps?|irc|news|mailto|gopher):|//)[^\s\]]*(?:[ \t]+([^\]\n]*))?\]", re.I)
_BARE_URL = re.compile(r"\b(?:https?|ftps?)://[^\s<>\[\]{}|\"]+", re.I)
_INNER_LINK = re.compile(r"\[\[((?:(?!\[\[|\]\])[^\n])*)\]\]")
_INTERLANG = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]+)*")
_PIPE_TRICK = re.compile(r"\s*(?:\([^()]*\)|,.*)$")


def replace_external_links(text: str) -> str:
    text = _EXT_LINK.sub(lambda m: (m.group(1) or "").strip(), text)
    return _BARE_URL.sub("", text)


def link_text(inner: str, ns: Namespaces) -> str:
    """Visible text of one [[...]] link with no nested links left inside."""
    s = inner.strip()
    visible = s.startswith(":")  # [[:Category:X]] shows a link instead of categorizing
    if visible:
        s = s[1:].lstrip()
    target, has_label, label = s.partition("|")
    prefix, has_colon, rest = target.partition(":")
    if has_colon and not visible:
        p = fold(prefix)
        if p in ns.file or p in ns.category:
            return ""
        if not ns.known(p) and _INTERLANG.fullmatch(prefix.strip()):
            return ""
    if has_label:
        if label.strip():
            return label.strip()
        bare = rest if has_colon and ns.known(fold(prefix)) else target  # pipe trick: [[Help:Foo (bar)|]] -> Foo
        return _PIPE_TRICK.sub("", bare).strip()
    shown = target.split("#", 1)[0].strip()
    return shown or target.strip().lstrip("#").strip()


def replace_internal_links(text: str, ns: Namespaces) -> str:
    """Resolve [[links]] from the innermost out, so a caption's links are handled before its file link."""
    while "[[" in text:
        new = _INNER_LINK.sub(lambda m: link_text(m.group(1), ns), text)
        if new == text:
            break
        text = new
    return text.replace("[[", "").replace("]]", "")


# ---------------------------------------------------------------------------
# Rules 8-10: emphasis, headings, magic words, tags, lines
# ---------------------------------------------------------------------------

_EMPHASIS = re.compile(r"'{2,}")
_MAGIC = regex.compile(r"__[\p{L}\p{N}_]+?__")
_BR = re.compile(r"<br\s*/?\s*>", re.I)
_TAG = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>")
_LIST_MARKER = re.compile(r"^[*#:;]+[ \t]*")
_HAS_CONTENT = regex.compile(r"[\p{L}\p{N}]")


def clean_lines(text: str) -> str:
    lines = []
    for line in text.split("\n"):
        line = _LIST_MARKER.sub("", line.strip()).strip()
        if line.startswith("|") or not _HAS_CONTENT.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def strip_wikitext(text: str, ns: Namespaces = DEFAULT_NAMESPACES) -> str:
    """Plain text of an article's wikitext, following rules 1-10 in the module docstring."""
    text = remove_comments(text)
    text = remove_refs(text)
    text = remove_dropped_tags(text)
    text = cut_back_matter(text)
    text = remove_templates(text)
    text = replace_external_links(text)
    text = replace_internal_links(text, ns)
    text = _EMPHASIS.sub("", text)
    text = _HEADING.sub(lambda m: m.group(2), text)
    text = _MAGIC.sub("", text)
    text = _BR.sub("\n", text)
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return clean_lines(text)


# ---------------------------------------------------------------------------
# Page filters on raw wikitext
# ---------------------------------------------------------------------------

_REDIRECT = re.compile(r"^\s*#\s*(?:redirect|رجوع[ _]مکرر)", re.I)
_TEMPLATE_NAME = re.compile(r"\{\{\s*([^{}|\n]+?)\s*(?:\||\}\})")
_CATEGORY_LINK = re.compile(r"\[\[\s*([^\]|:\n]+?)\s*:\s*([^\]|\n]+)")


def is_redirect_text(text: str) -> bool:
    return bool(_REDIRECT.match(text.translate(_FOLD)))


def disambiguation_reason(title: str, text: str, ns: Namespaces = DEFAULT_NAMESPACES) -> str | None:
    """'template', 'category' or 'title' if the page is a disambiguation page, else None.

    Checked on the raw wikitext, before any stripping: a disambiguation template
    ({{ضد ابہام}}, {{Disambiguation}}, ...), a category whose name contains
    ضد ابہام or "disambiguation", or a title ending in "(ضد ابہام)".
    """
    marker = fold(DISAMBIG_MARKER)
    for m in _TEMPLATE_NAME.finditer(text):
        name = fold(m.group(1))
        prefix, colon, rest = name.partition(":")
        if colon and prefix in ns.template:
            name = rest.strip()
        if name in _DISAMBIG or marker in name or name.startswith("disambig"):
            return "template"
    for m in _CATEGORY_LINK.finditer(text):
        if fold(m.group(1)) in ns.category:
            name = fold(m.group(2))
            if marker in name or "disambiguation" in name:
                return "category"
    if marker in fold(title):
        return "title"
    return None
