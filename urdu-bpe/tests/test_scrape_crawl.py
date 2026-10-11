"""crawl.py end to end against a scripted local site, with a test adapter and synthetic pages."""

import re
from pathlib import Path

import pytest

from collect.scrape import crawl
from collect.scrape.fetch import iter_log
from collect.scrape.sites.base import SiteAdapter

UA = "TestBot/1.0 (tests; contact: nobody@example.invalid)"
P = "یہ ایک مصنوعی جملہ ہے جو صرف جانچ کے لیے لکھا گیا ہے اور کسی اصل مضمون سے نہیں لیا گیا۔"


class FakeAdapter(SiteAdapter):
    site_id = "fake"
    name = "Fake"
    domain = "127.0.0.1"
    genre = "test"
    extractor_version = "fake/1"
    terms_summary = "synthetic test site"
    article_url_re = re.compile(r"^http://127\.0\.0\.1:\d+/(?:private/)?\d+/[a-z0-9-]+/$")
    container_selectors = ("article .body",)
    title_selector = "h1"

    def follow_sitemap(self, url: str) -> bool:
        return "post-sitemap" in url

    def skip_reason(self, soup, url):
        return "media page" if soup.find(class_="video-player") else None


def page(i: int, paragraphs: int = 3, canonical: str | None = None, media: bool = False) -> bytes:
    body = "".join(f"<p>{P} {i}</p>" for _ in range(paragraphs))
    return f"""<html><head><title>t</title>
      <meta property="article:published_time" content="2020-01-{1 + i % 28:02d}T00:00:00Z">
      {f'<link rel="canonical" href="{canonical}">' if canonical else ''}</head>
      <body><nav><p>{P}</p></nav><article><h1>عنوان {i}</h1>
      {'<div class="video-player"></div>' if media else ''}
      <div class="body"><div class="share-box"><p>شیئر کریں</p></div>{body}
      <p>رابطہ: writer{i}@example.com</p><div class="author-bio"><p>مصنف کا تعارف</p></div>
      <div id="comments"><p>ایک تبصرہ</p></div></div></article></body></html>""".encode()


@pytest.fixture
def fake_site(site, monkeypatch):
    base = site.base_url
    site.robots = f"User-agent: *\nDisallow: /private/\nSitemap: {base}/sitemap_index.xml\n"
    site.files["/sitemap_index.xml"] = (
        f"<sitemapindex><sitemap><loc>{base}/post-sitemap1.xml</loc></sitemap>"
        f"<sitemap><loc>{base}/post-sitemap2.xml</loc></sitemap>"
        f"<sitemap><loc>{base}/tag-sitemap.xml</loc></sitemap></sitemapindex>").encode()
    s1 = [f"{base}/{i}/story-{i}/" for i in range(1, 6)] + [f"{base}/private/99/secret/", f"{base}/tag/sport/"]
    s2 = [f"{base}/{i}/story-{i}/" for i in range(6, 11)]
    for name, urls in (("post-sitemap1.xml", s1), ("post-sitemap2.xml", s2)):
        site.files[f"/{name}"] = ("<urlset>" + "".join(f"<url><loc>{u}</loc></url>" for u in urls)
                                  + "</urlset>").encode()
    site.files["/tag-sitemap.xml"] = f"<urlset><url><loc>{base}/tag/x/</loc></url></urlset>".encode()
    for i in range(1, 11):
        site.files[f"/{i}/story-{i}/"] = page(i, canonical=f"{base}/{i}/story-{i}/")
    site.files["/private/99/secret/"] = page(99)
    site.files["/4/story-4/"] = page(4, media=True)  # a media page: skipped
    site.files["/5/story-5/"] = page(5, canonical=f"{base}/1/story-1/")  # same story as story-1
    adapter = type("Adapter", (FakeAdapter,), {"home_url": base + "/", "sitemap_urls": (base + "/sitemap_index.xml",)})
    monkeypatch.setitem(crawl.ADAPTERS, "fake", adapter)
    return site


def run(tmp_path: Path, *extra: str) -> int:
    args = crawl.parse_args(["--site", "fake", "--user-agent", UA, "--min-interval", "0", "--jitter", "0",
                             "--data-dir", str(tmp_path / "own"), "--evidence-dir", str(tmp_path / "evidence"),
                             "--log-dir", str(tmp_path / "logs"), *extra])
    return crawl.run(args, client_kwargs={"sleep": lambda s: None})


def paths_requested(site) -> list[str]:
    return [r["path"] for r in site.requests]


def own(tmp_path: Path) -> crawl.SitePaths:
    return crawl.SitePaths(tmp_path / "own" / "fake")


def test_end_to_end(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path) == 0
    p = own(tmp_path)
    articles = crawl.read_rows(p.articles)
    kept = sorted(int(a["url"].rstrip("/").rsplit("-", 1)[1]) for a in articles)
    # 4 is a media page; 1 and 5 share a canonical URL, so whichever comes first in the crawl order is kept.
    assert len(kept) == 8 and 4 not in kept and len({1, 5} & set(kept)) == 1
    a = articles[0]
    assert set(a) == {"url", "canonical_url", "title", "text", "published", "fetched_at_utc", "http_status",
                      "html_sha256", "extractor_version"}
    assert a["text"].split("\n")[0] == a["title"] and a["http_status"] == 200 and a["extractor_version"] == "fake/1"
    for art in articles:  # junk, emails, author boxes and comments never reach the text
        assert "شیئر" not in art["text"] and "@" not in art["text"]
        assert "مصنف کا تعارف" not in art["text"] and "تبصرہ" not in art["text"]
        assert len(art["text"].split("\n")) == 4  # title + 3 paragraphs
    reasons = sorted(s["reason"] for s in crawl.read_rows(p.skips))
    assert reasons == ["duplicate canonical URL", "media page"]
    requested = paths_requested(fake_site)
    assert "/private/99/secret/" not in requested  # robots.txt disallows it
    assert "/tag-sitemap.xml" not in requested and "/tag/sport/" not in requested
    log = list(iter_log(tmp_path / "evidence" / "crawl_log_fake.jsonl.gz"))
    assert len([r for r in log if r["status"] is not None]) == len(requested)


def test_disallowed_article_url_is_logged_and_never_requested(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path) == 0
    fetched = {r["url"].split("/", 3)[3]: r for r in crawl.read_rows(own(tmp_path).fetched)}
    assert fetched["private/99/secret/"]["outcome"] == "robots"
    assert "/private/99/secret/" not in paths_requested(fake_site)
    log = list(iter_log(tmp_path / "evidence" / "crawl_log_fake.jsonl.gz"))
    blocked = [r for r in log if r["robots_allowed"] is False]
    assert len(blocked) == 1 and blocked[0]["url"].endswith("/private/99/secret/") and blocked[0]["status"] is None


def test_resume_requests_nothing_twice(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path, "--limit", "3") == 0
    first = paths_requested(fake_site)
    assert len(crawl.read_rows(own(tmp_path).articles)) == 3
    fake_site.requests.clear()
    assert run(tmp_path) == 0
    second = paths_requested(fake_site)
    assert not set(second) & {p for p in first if p.endswith("/") and p != "/"}  # no article fetched again
    assert "/sitemap_index.xml" not in second and "/post-sitemap1.xml" not in second  # sitemaps read once
    assert len(crawl.read_rows(own(tmp_path).articles)) == 8


def test_resume_reuses_cached_html(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path) == 0
    p = own(tmp_path)
    before = crawl.read_rows(p.articles)
    p.articles.unlink()
    p.skips.unlink()
    fake_site.requests.clear()
    assert run(tmp_path) == 0
    assert [r for r in paths_requested(fake_site) if r != "/robots.txt"] == []
    assert [a["url"] for a in crawl.read_rows(p.articles)] == [a["url"] for a in before]


def test_404_is_not_retried(fake_site, tmp_path: Path) -> None:
    del fake_site.files["/7/story-7/"]
    assert run(tmp_path) == 0
    fake_site.requests.clear()
    assert run(tmp_path) == 0
    assert "/7/story-7/" not in paths_requested(fake_site)


def test_stops_at_target_mb(fake_site, tmp_path: Path) -> None:
    one = len(crawl.read_rows(own(tmp_path).articles))  # 0: nothing yet
    assert one == 0
    assert run(tmp_path, "--target-mb", "0.001") == 0  # 1,000 bytes: a few synthetic articles
    articles = crawl.read_rows(own(tmp_path).articles)
    total = sum(len(a["text"].encode()) for a in articles)
    assert 1 < len(articles) < 8 and total >= 1000
    assert total - len(articles[-1]["text"].encode()) < 1000  # it stopped as soon as the target was reached
    fake_site.requests.clear()
    assert run(tmp_path, "--target-mb", "0.001") == 0  # already there: no article requests
    assert [r for r in paths_requested(fake_site) if r != "/robots.txt"] == []


def test_limit(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path, "--limit", "2") == 0
    assert len(crawl.read_rows(own(tmp_path).articles)) == 2


def test_partial_discovery_continues_next_run(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path, "--max-sitemaps", "1", "--limit", "1") == 0
    assert "/post-sitemap2.xml" not in paths_requested(fake_site)
    assert run(tmp_path, "--limit", "0") == 0
    assert "/post-sitemap2.xml" in paths_requested(fake_site)
    assert len(crawl.read_rows(own(tmp_path).urls)) == 11  # 10 stories + the robots-blocked one


def test_reextract_makes_no_requests(fake_site, tmp_path: Path) -> None:
    assert run(tmp_path) == 0
    before = crawl.read_rows(own(tmp_path).articles)
    fake_site.requests.clear()
    assert run(tmp_path, "--reextract") == 0
    assert fake_site.requests == []
    assert crawl.read_rows(own(tmp_path).articles) == before


def test_redirect_to_a_blocked_url_is_not_followed(fake_site, tmp_path: Path) -> None:
    fake_site.redirects["/2/story-2/"] = (302, "/login?next=/2/story-2/")
    assert run(tmp_path) == 0
    fetched = {r["url"].split("/", 3)[3]: r for r in crawl.read_rows(own(tmp_path).fetched)}
    assert fetched["2/story-2/"]["outcome"] == "redirect_rejected"
    assert "/login" not in paths_requested(fake_site)


def test_crawl_order_is_seeded_and_independent_of_other_urls() -> None:
    urls = [f"https://x.test/{i}/" for i in range(100)]
    order = crawl.crawl_order(urls, 1337)
    assert order == crawl.crawl_order(list(reversed(urls)), 1337)
    assert order != crawl.crawl_order(urls, 1)
    subset = crawl.crawl_order(urls[:50], 1337)
    assert [u for u in order if u in set(urls[:50])] == subset


def test_partial_last_line_is_ignored_then_cut(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"url": "a"}\n{"url": "b"', encoding="utf-8")
    assert crawl.read_rows(path) == [{"url": "a"}]
    w = crawl.Appender(path)
    w.write({"url": "c"})
    w.close()
    assert crawl.read_rows(path) == [{"url": "a"}, {"url": "c"}]


def test_min_interval_below_one_second_is_refused_outside_tests() -> None:
    with pytest.raises(SystemExit):
        crawl.parse_args(["--site", "voa_urdu", "--min-interval", "0.2"])
