"""Site adapters on synthetic fixtures (tests/fixtures/scrape/<site>/, invented text, real page structure)."""

from pathlib import Path

import pytest

from collect.scrape.sitemaps import Entry
from collect.scrape.sites.base import NotArticle
from collect.scrape.sites.humsub import HumSub
from collect.scrape.sites.registry import ADAPTERS
from collect.scrape.sites.voa_urdu import VoaUrdu

FIXTURES = Path(__file__).parent / "fixtures" / "scrape"


def fixture(site: str, name: str) -> bytes:
    return (FIXTURES / site / name).read_bytes()


# -- VOA Urdu ------------------------------------------------------------------

def test_voa_article() -> None:
    url = "https://www.urduvoa.com/a/test-story/1234567.html"
    e = VoaUrdu().extract(fixture("voa_urdu", "article.html"), url)
    assert e.title == "تجرباتی خبر کا عنوان"
    assert e.published == "2019-03-04 10:20:30Z"
    assert e.canonical_url == url
    assert e.body.split("\n") == [
        "اسلام آباد — یہ پہلا تجرباتی پیراگراف ہے جو صرف جانچ کے لیے لکھا گیا ہے۔",
        "دوسرے پیراگراف میں کچھ الفاظ نمایاں ہیں اور ایک ربط بھی ہے۔",
        "آخری تجرباتی پیراگراف یہاں ختم ہوتا ہے۔",
    ]
    assert e.dropped_lines == 1  # the paragraph with an email address
    assert e.text == e.title + "\n" + e.body
    for junk in ("فرضی نامہ نگار", "شیئر", "پاکستان", "تصویر", "ٹویٹ", "انفوگرافک", "متعلقہ", "حقوق", "@"):
        assert junk not in e.text


def test_voa_media_page_is_skipped() -> None:
    with pytest.raises(NotArticle, match="media page"):
        VoaUrdu().extract(fixture("voa_urdu", "media.html"), "https://www.urduvoa.com/a/7581671.html")


@pytest.mark.parametrize("url, ok", [
    ("https://www.urduvoa.com/a/afghanistan-list/3732084.html", True),
    ("https://www.urduvoa.com/a/7581671.html", True),
    ("http://www.urduvoa.com/a/afghanistan-list/3732084.html", False),
    ("https://www.urduvoa.com/z/2257", False),
    ("https://www.urduvoa.com/comments/a3732084.html", False),
    ("https://www.urduvoa.com/s?k=pakistan", False),
    ("https://www.urduvoa.com/a/x/1.html?p=2", False),
    ("https://evil.example/a/x/1.html", False),
])
def test_voa_article_urls(url: str, ok: bool) -> None:
    assert VoaUrdu().is_article_url(url) is ok


def test_voa_follows_only_archive_sitemaps() -> None:
    a = VoaUrdu()
    assert a.follow_sitemap("https://www.urduvoa.com/sitemap_445_3.xml.gz")
    for other in ("videos", "sections", "latest", "news"):
        assert not a.follow_sitemap(f"https://www.urduvoa.com/sitemap_445_{other}.xml.gz")
    assert not a.follow_sitemap("https://www.urduvoa.com/Sitemap/Hreflang")


# -- Hum Sub ---------------------------------------------------------------------

def test_humsub_post() -> None:
    url = "https://www.humsub.com.pk/123456/farzi-kalam-nigar-7/"
    e = HumSub().extract(fixture("humsub", "post.html"), url)
    assert e.title == "تجرباتی کالم کا عنوان"
    assert e.published == "2018-07-09T12:00:00+00:00"
    assert e.canonical_url == url
    assert e.body.split("\n") == [
        "کالم کا پہلا تجرباتی پیراگراف جس میں رائے کا اظہار کیا گیا ہے۔",
        "دوسرا پیراگراف",
        "جس میں ایک شعر کا پہلا مصرع",
        "اور دوسرا مصرع ہے۔",
        "ذیلی سرخی",
        "پہلی شق",
        "دوسری شق",
        "پس نوشت: یہ آخری تجرباتی سطر ہے۔",
    ]
    assert e.dropped_lines == 1  # the "press the next-page button" line
    for junk in ("فرضی کالم نگار", "شیئر", "اشتہار", "کیپشن", "تبصرہ", "تعارف", "مزید پڑھیں", "صفحہ اول", "@",
                 "Pages", "اگلا صفحہ"):
        assert junk not in e.text


def test_humsub_bbc_reprint_is_skipped() -> None:
    with pytest.raises(NotArticle, match=r"reprint \(category-bbc\)"):
        HumSub().extract(fixture("humsub", "reprint.html"), "https://www.humsub.com.pk/207973/bbc-1715/")


def test_humsub_reprint_credited_to_bbc_without_category_is_skipped() -> None:
    html = fixture("humsub", "post.html").replace('content="فرضی کالم نگار"'.encode(), 'content="بی بی سی"'.encode())
    with pytest.raises(NotArticle, match="credited to BBC/DW"):
        HumSub().extract(html, "https://www.humsub.com.pk/1/x/")


@pytest.mark.parametrize("url, ok", [
    ("https://www.humsub.com.pk/206373/asma-tariq-23/", True),
    ("https://humsub.com.pk/n/60335/2026/10/11/dr-niaz-aazim/", True),
    ("https://www.humsub.com.pk/206373/asma-tariq-23/?replytocom=5", False),
    ("https://www.humsub.com.pk/author/asma-tariq/", False),
    ("https://humsub.com.pk/n/author/dr-niaz-aazim/", False),
    ("https://humsub.com.pk/n/page/2/", False),
    ("https://www.humsub.com.pk/tag/x/", False),
    ("https://www.humsub.com.pk/?s=urdu", False),
])
def test_humsub_article_urls(url: str, ok: bool) -> None:
    assert HumSub().is_article_url(url) is ok


def test_humsub_follows_only_post_sitemaps() -> None:
    a = HumSub()
    assert a.follow_sitemap("https://www.humsub.com.pk/post-sitemap38.xml")
    assert a.follow_sitemap("https://www.humsub.com.pk/post-sitemap.xml")
    for other in ("author-sitemap5.xml", "post_tag-sitemap.xml", "category-sitemap.xml", "page-sitemap.xml"):
        assert not a.follow_sitemap(f"https://www.humsub.com.pk/{other}")


# -- all adapters --------------------------------------------------------------------

@pytest.mark.parametrize("site_id", sorted(ADAPTERS))
def test_adapter_contract(site_id: str) -> None:
    a = ADAPTERS[site_id]()
    assert a.site_id == site_id and a.extractor_version.startswith(site_id + "/")
    assert a.is_article_url(a.home_url) is False
    assert a.terms_summary and a.genre and a.container_selectors
    assert all(a.allow_url(u) for u in a.sitemap_urls)
    assert a.keep_entry(Entry("https://x.test/1"))


def test_voa_old_layout_without_paragraphs() -> None:
    e = VoaUrdu().extract(fixture("voa_urdu", "old_layout.html"), "https://www.urduvoa.com/a/old/1755703.html")
    assert e.published == "2013-09-23 18:00:00Z"
    assert e.body.split("\n") == [
        "یہ پہلا جملہ ہے جو کسی پیراگراف کے بغیر لکھا گیا ہے اور اسی سطر میں جاری رہتا ہے۔",
        "دوسرا حصہ دو لائن بریک کے بعد آتا ہے۔",
        "ذیلی سرخی",
        "آخری حصہ ایک سادہ div میں ہے۔",
    ]
    assert "SEE ALSO" not in e.text and "کسی اور خبر" not in e.text and "واشنگٹن" not in e.text
