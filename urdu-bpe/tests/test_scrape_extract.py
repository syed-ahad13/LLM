"""Generic HTML-to-text helpers used by every site adapter (synthetic HTML only)."""

import re

from collect.scrape.extract import (
    GENERIC_DROP_SELECTORS, arabic_letters, block_text, canonical_url, drop_elements, drop_lines, json_ld,
    page_title, published_date, soup_of, text_blocks,
)


def test_inline_tags_join_without_inventing_spaces() -> None:
    soup = soup_of("<p>اردو<b>زبان</b> اور <a href='#'>ادب</a>\n  کی   تاریخ</p>")
    assert block_text(soup.p) == "اردوزبان اور ادب کی تاریخ"


def test_br_splits_lines_for_poetry() -> None:
    soup = soup_of("<p>پہلا مصرع<br>دوسرا مصرع<br/><br>  </p>")
    assert block_text(soup.p) == "پہلا مصرع\nدوسرا مصرع"


def test_nbsp_and_zwnj() -> None:
    # NBSP is HTML whitespace and collapses; ZWNJ is a character Urdu uses and stays for normalize.py.
    soup = soup_of("<p>ایک  لفظ نہ‌صرف</p>")
    assert block_text(soup.p) == "ایک لفظ نہ‌صرف"


def test_text_blocks_in_order_without_double_counting() -> None:
    soup = soup_of("""<div id="c"><h2>سرخی</h2><p>پہلا</p>
        <ul><li><p>فہرست کا پیرا</p></li><li>دوسری شق</li></ul>
        <h3>ذیلی</h3><blockquote><p>اقتباس</p></blockquote><h4>skipped h4</h4><p>   </p></div>""")
    assert text_blocks(soup.find(id="c")) == ["سرخی", "پہلا", "فہرست کا پیرا", "دوسری شق", "ذیلی", "اقتباس"]


def test_generic_drop_selectors_remove_junk() -> None:
    soup = soup_of("""<article><p>متن</p><script>var x='اردو';</script><figure><figcaption>تصویر</figcaption></figure>
        <div class="share-buttons"><p>شیئر</p></div><div class="related-posts"><li>متعلقہ</li></div>
        <div class="author-box"><p>مصنف کا تعارف</p></div><div id="comments"><p>تبصرہ</p></div>
        <aside><p>سائیڈ</p></aside><!-- <p>comment</p> --><p>آخری</p></article>""")
    art = soup.article
    drop_elements(art, GENERIC_DROP_SELECTORS)
    assert text_blocks(art) == ["متن", "آخری"]


def test_drop_lines_removes_emails_and_bylines() -> None:
    lines = ["خبر کا متن", "رابطہ: someone@example.com", "تحریر: کوئی نام", "مزید متن"]
    kept, dropped = drop_lines(lines, [re.compile(r"^تحریر:")])
    assert kept == ["خبر کا متن", "مزید متن"] and dropped == 2


def test_published_date_meta_then_json_ld() -> None:
    meta = soup_of('<meta property="article:published_time" content="2024-05-01T08:00:00+05:00">')
    assert published_date(meta) == "2024-05-01T08:00:00+05:00"
    ld = soup_of("""<script type="application/ld+json">{"@graph":[{"@type":"WebPage"},
        {"@type":"NewsArticle","datePublished":"2023-02-03T04:05:06Z","author":{"name":"X"}}]}</script>
        <script type="application/ld+json">{broken</script>""")
    assert published_date(ld) == "2023-02-03T04:05:06Z"
    assert any(o.get("@type") == "NewsArticle" for o in json_ld(ld))
    assert published_date(soup_of("<p>none</p>")) is None


def test_canonical_and_title() -> None:
    soup = soup_of("""<head><link rel="canonical" href="/urdu/articles/abc"><meta property="og:title" content="OG">
        <title>Page | Site</title></head><body><h1> اصل   عنوان </h1></body>""")
    assert canonical_url(soup, "https://www.example.com/x?y=1") == "https://www.example.com/urdu/articles/abc"
    assert page_title(soup) == "اصل عنوان"
    assert page_title(soup_of("<head><title>Only title</title></head>")) == "Only title"
    og = soup_of('<meta property="og:url" content="https://e.com/a">')
    assert canonical_url(og, "https://e.com/b") == "https://e.com/a"


def test_arabic_letters_counts_letters_only() -> None:
    assert arabic_letters("اردو 123 ۱۲۳ ، abc") == 4
    assert arabic_letters("اُردُو") == 4  # diacritics are marks, not letters
