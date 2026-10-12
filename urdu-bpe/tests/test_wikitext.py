"""Hand-written wikitext fixtures for clean/wikitext.py, one group per rule in its docstring."""

import pytest

from clean.wikitext import (
    Namespaces,
    cut_back_matter,
    disambiguation_reason,
    is_redirect_text,
    link_text,
    remove_templates,
    strip_wikitext,
)

NS = Namespaces.from_siteinfo({6: "فائل", 14: "زمرہ", 10: "سانچہ", 4: "ویکیپیڈیا", 2: "صارف"})
BODY = "پاکستان جنوبی ایشیا کا ایک ملک ہے۔"


def strip(text: str) -> str:
    return strip_wikitext(text, NS)


# ---------------------------------------------------------------------------
# a. comments and references
# ---------------------------------------------------------------------------

def test_comments_removed_including_multiline_and_unclosed() -> None:
    assert strip("الف <!-- چھپا\nہوا --> ب") == "الف  ب"
    assert strip("الف\n<!-- unclosed till the end\nلائن") == "الف"


@pytest.mark.parametrize("ref", [
    '<ref>{{cite web|url=http://x.org|title=T}}</ref>',
    '<ref name="a">کتاب، صفحہ 3</ref>',
    '<ref name=b />',
    '<ref name="c"/>',
    '<REF group="n">نوٹ</REF>',
])
def test_refs_removed(ref: str) -> None:
    assert strip(f"جملہ{ref} جاری۔") == "جملہ جاری۔"


def test_self_closing_ref_does_not_swallow_until_next_close() -> None:
    assert strip('ایک<ref name="a"/> دو <ref>حوالہ</ref> تین') == "ایک دو  تین"


def test_references_tag_removed() -> None:
    assert strip("متن\n<references/>\n<references>\n<ref name=a>x</ref>\n</references>") == "متن"


# ---------------------------------------------------------------------------
# b. tags whose content is dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["math", "chem", "score", "gallery", "timeline", "syntaxhighlight", "source", "code",
                                 "pre", "imagemap", "graph", "mapframe", "templatedata"])
def test_dropped_tag_content(tag: str) -> None:
    assert strip(f'الف <{tag} attr="1">x = y\n{{{{z}}}}</{tag}> ب') == "الف  ب"


def test_dropped_self_closing_tag_and_poem_kept() -> None:
    assert strip('الف <mapframe zoom=5 /> ب') == "الف  ب"
    assert strip("<poem>\nپہلا مصرع\nدوسرا مصرع\n</poem>") == "پہلا مصرع\nدوسرا مصرع"


# ---------------------------------------------------------------------------
# c. templates, parser functions, tables
# ---------------------------------------------------------------------------

def test_nested_templates() -> None:
    text = "{{Infobox ملک|نام={{lang|ur|پاکستان}}|آبادی={{#expr: 2 * {{x}} }}}}" + BODY
    assert strip(text) == BODY


def test_parser_function_removed() -> None:
    assert strip("{{#if: {{{1|}}} | ہاں | نہیں }}" + BODY) == BODY


def test_table_removed() -> None:
    text = f"{BODY}\n{{| class=\"wikitable\"\n|-\n! سال !! آبادی\n|-\n| 2017 || 207\n|}}\nآخری سطر۔"
    assert strip(text) == f"{BODY}\nآخری سطر۔"


def test_table_inside_template() -> None:
    text = "{{Navbox|list=\n{| class=x\n|-\n| [[لاہور]] || {{flag|PK}}\n|}\n}}\n" + BODY
    assert strip(text) == BODY


def test_template_inside_table() -> None:
    text = "{|\n|-\n| {{nowrap|{{small|x}}}} || y\n|}\n" + BODY
    assert strip(text) == BODY


def test_unclosed_template_keeps_following_text() -> None:
    # MediaWiki shows an unclosed "{{" as text; we must not lose the rest of the article.
    assert remove_templates("الف {{ب\nج {{د}} ہ") == "الف {{ب\nج  ہ"


def test_stray_closer_is_text() -> None:
    assert remove_templates("الف }} ب") == "الف }} ب"


# ---------------------------------------------------------------------------
# d. internal links
# ---------------------------------------------------------------------------

def test_piped_and_plain_links() -> None:
    assert strip("[[اسلام آباد|دارالحکومت]] اور [[لاہور]] شہر ہیں۔") == "دارالحکومت اور لاہور شہر ہیں۔"


def test_anchor_dropped_from_plain_link() -> None:
    assert strip("دیکھیے [[پاکستان#تاریخ]] اور [[#جغرافیہ]]۔") == "دیکھیے پاکستان اور جغرافیہ۔"


def test_letters_glued_after_link() -> None:
    assert strip("[[کتاب]]وں کی [[book]]s") == "کتابوں کی books"


@pytest.mark.parametrize("link", [
    "[[فائل:Flag of Pakistan.svg|thumb|پرچم]]",
    "[[File:Map.png|250px]]",
    "[[Image:x.jpg]]",
    "[[تصویر:منظر.jpg|left]]",
    "[[ملف:x.png]]",
    "[[زمرہ:ایشیائی ممالک]]",
    "[[Category:Countries|P]]",
    "[[en:Pakistan]]",
    "[[zh-yue:巴基斯坦]]",
])
def test_file_category_and_interlanguage_links_dropped(link: str) -> None:
    assert strip(f"{BODY}{link}") == BODY


def test_file_link_whose_caption_contains_links() -> None:
    text = f"[[فائل:Lahore.jpg|thumb|[[لاہور]] کا [[بادشاہی مسجد|مسجد]] منظر]]\n{BODY}"
    assert strip(text) == BODY


def test_localized_file_namespace_from_siteinfo() -> None:
    ns = Namespaces.from_siteinfo({6: "Bestand"})
    assert link_text("Bestand:x.jpg|thumb", ns) == ""


def test_namespace_name_is_not_interlanguage() -> None:
    assert link_text("ویکیپیڈیا:دیوان عام|دیوان", NS) == "دیوان"
    assert link_text("Help:Contents", NS) == "Help:Contents"  # unknown, but not a language code


def test_visible_category_link_and_pipe_trick() -> None:
    assert link_text(":زمرہ:شہر|شہروں کا زمرہ", NS) == "شہروں کا زمرہ"
    assert link_text("لاہور (شہر)|", NS) == "لاہور"


# ---------------------------------------------------------------------------
# e. external links and URLs
# ---------------------------------------------------------------------------

def test_external_links() -> None:
    assert strip("دیکھیے [http://example.org سرکاری ویب سائٹ] یہاں۔") == "دیکھیے سرکاری ویب سائٹ یہاں۔"
    assert strip("حوالہ [https://example.org/x?a=1] ختم۔") == "حوالہ  ختم۔"
    assert strip("سائٹ https://www.example.com/page ہے۔") == "سائٹ  ہے۔"
    assert strip("[//example.org/x پروٹوکول کے بغیر] ٹھیک۔") == "پروٹوکول کے بغیر ٹھیک۔"


# ---------------------------------------------------------------------------
# f. emphasis and headings
# ---------------------------------------------------------------------------

def test_emphasis_removed() -> None:
    assert strip("'''پاکستان''' ایک ''ملک'' ہے اور '''''یہ''''' اہم ہے۔") == "پاکستان ایک ملک ہے اور یہ اہم ہے۔"


def test_heading_becomes_its_own_line() -> None:
    assert strip(f"{BODY}\n== تاریخ ==\nپہلا جملہ۔\n=== قدیم دور===\nدوسرا۔") == f"{BODY}\nتاریخ\nپہلا جملہ۔\nقدیم دور\nدوسرا۔"


# ---------------------------------------------------------------------------
# g. back matter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("heading", ["حوالہ جات", "حوالے", "حواشی", "مزید دیکھیے", "بیرونی روابط", "مزید پڑھیے",
                                     "کتابیات", "References", "See also", "External links", "Further reading"])
def test_back_matter_cut(heading: str) -> None:
    text = f"{BODY}\n== تاریخ ==\nمتن۔\n== {heading} ==\n* [[کچھ]]\n{{{{reflist}}}}\n== بعد ==\nیہ بھی نہیں۔"
    assert strip(text) == f"{BODY}\nتاریخ\nمتن۔"


def test_back_matter_matches_arabic_letter_variants_and_spacing() -> None:
    # Arabic heh U+0647 typed for gol heh, extra spaces, bold markup: still the references heading.
    assert cut_back_matter("الف\n==  '''حواله  جات''' ==\nب") == "الف\n"


def test_heading_that_only_contains_a_back_matter_word_is_kept() -> None:
    assert strip("== حوالہ جات کی تاریخ ==\nمتن۔") == "حوالہ جات کی تاریخ\nمتن۔"


# ---------------------------------------------------------------------------
# h. list markers, HTML tags, entities, magic words
# ---------------------------------------------------------------------------

def test_list_markers_stripped() -> None:
    assert strip("* پہلا\n** دوسرا\n# تیسرا\n: چوتھا\n; پانچواں") == "پہلا\nدوسرا\nتیسرا\nچوتھا\nپانچواں"


def test_html_tags_keep_inner_text_and_br_breaks_line() -> None:
    assert strip('<span style="x">رنگین</span> <b>موٹا</b><br/>نئی سطر <div>بلاک</div>') == "رنگین موٹا\nنئی سطر بلاک"


def test_entities_unescaped() -> None:
    assert strip("A&amp;B &lt;x&gt; &quot;q&quot; &#1575;") == 'A&B <x> "q" ا'


def test_magic_words_removed() -> None:
    assert strip(f"__TOC__\n{BODY}__NOTOC__ __NOEDITSECTION__") == BODY


# ---------------------------------------------------------------------------
# i. empty and punctuation-only lines
# ---------------------------------------------------------------------------

def test_empty_and_punctuation_lines_dropped() -> None:
    assert strip(f"{BODY}\n\n\n----\n۔ ، ؟\n  \n2024\n| leftover = cell") == f"{BODY}\n2024"


# ---------------------------------------------------------------------------
# A whole article
# ---------------------------------------------------------------------------

ARTICLE = """{{Infobox شہر
| نام = لاہور
| تصویر = [[فائل:Lahore.jpg|250px]]
| آبادی = {{formatnum:11126285}}
}}
'''لاہور''' ([[انگریزی زبان|انگریزی]]: Lahore) [[پاکستان]] کے [[صوبہ]] [[پنجاب، پاکستان|پنجاب]] کا [[دارالحکومت]] ہے۔<ref>{{cite web|url=http://x|title=y}}</ref>
یہ [[کراچی]] کے بعد پاکستان کا دوسرا بڑا شہر ہے۔<!-- آبادی کا اندازہ -->

== تاریخ ==
[[فائل:Badshahi.jpg|thumb|[[بادشاہی مسجد]] ([[1673ء]])]]
لاہور کی تاریخ ہزاروں سال پرانی ہے۔
{| class="wikitable"
! سال !! آبادی
|-
| 1998 || 5,143,495
|}

== حوالہ جات ==
{{حوالہ جات}}
[[زمرہ:پاکستان کے شہر]]
[[en:Lahore]]"""


def test_whole_article() -> None:
    assert strip(ARTICLE) == (
        "لاہور (انگریزی: Lahore) پاکستان کے صوبہ پنجاب کا دارالحکومت ہے۔\n"
        "یہ کراچی کے بعد پاکستان کا دوسرا بڑا شہر ہے۔\n"
        "تاریخ\n"
        "لاہور کی تاریخ ہزاروں سال پرانی ہے۔"
    )


# ---------------------------------------------------------------------------
# Page filters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["#REDIRECT [[پاکستان]]", "#redirect[[x]]", "  #رجوع مکرر [[پاکستان]]", "#رجوع_مکرر [[x]]"])
def test_redirect_text(text: str) -> None:
    assert is_redirect_text(text)


def test_not_redirect() -> None:
    assert not is_redirect_text(BODY + "\n#REDIRECT")


@pytest.mark.parametrize("title, text, reason", [
    ("لاہور", "'''لاہور''' سے مراد:\n* [[لاہور]]\n{{ضد ابہام}}", "template"),
    ("لاہور", "* [[x]]\n{{ سانچہ:ضد ابہام | شہر }}", "template"),
    ("Mercury", "* [[x]]\n{{Disambiguation}}", "template"),
    ("Mercury", "* [[x]]\n{{disambig|geo}}", "template"),
    ("لاہور", "* [[x]]\n{{ضد_ابهام}}", "template"),  # underscore and Arabic heh
    ("لاہور", "* [[x]]\n[[زمرہ:ضد ابہام صفحات]]", "category"),
    ("لاہور (ضد ابہام)", "* [[x]]", "title"),
])
def test_disambiguation_detected(title: str, text: str, reason: str) -> None:
    assert disambiguation_reason(title, text, NS) == reason


def test_ordinary_article_is_not_disambiguation() -> None:
    assert disambiguation_reason("لاہور", ARTICLE, NS) is None
