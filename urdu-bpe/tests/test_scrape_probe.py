"""Phase A probe helpers and an end-to-end probe of a scripted local site (synthetic pages only)."""

import argparse
import logging
from pathlib import Path

import pytest

from collect.scrape import probe
from collect.scrape.extract import largest_text_block, soup_of
from collect.scrape.fetch import EvidenceLog, iter_log
from collect.scrape.sitemaps import parse_sitemap, type_key

UA = "TestBot/1.0 (tests; contact: nobody@example.invalid)"
URDU_P = "یہ ایک مصنوعی جملہ ہے جو صرف جانچ کے لیے لکھا گیا ہے اور کسی اصل مضمون سے نہیں لیا گیا۔"


# -- sitemaps -------------------------------------------------------------------

def test_parse_sitemap_index_urlset_gzip_and_junk() -> None:
    index = (b'<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
             b"<sitemap><loc> https://x.test/post-sitemap1.xml </loc><lastmod>2026-01-01</lastmod></sitemap>"
             b"<sitemap><loc>https://x.test/post-sitemap2.xml</loc></sitemap></sitemapindex>")
    sm = parse_sitemap(index)
    assert sm.kind == "index" and [e.loc for e in sm.entries] == ["https://x.test/post-sitemap1.xml",
                                                                 "https://x.test/post-sitemap2.xml"]
    assert sm.entries[0].lastmod == "2026-01-01"
    import gzip
    urlset = (b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
              b'xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">'
              b"<url><loc>https://x.test/1/a/</loc><news:news><news:title>t</news:title></news:news></url></urlset>")
    assert [e.loc for e in parse_sitemap(gzip.compress(urlset)).entries] == ["https://x.test/1/a/"]
    assert parse_sitemap(b"<!doctype html><html><body>404</body></html>").kind == "unknown"
    assert parse_sitemap(b"not xml at all").kind == "unknown"


def test_parse_sitemap_does_not_expand_entities() -> None:
    bomb = (b'<?xml version="1.0"?><!DOCTYPE u [<!ENTITY a "AAAAAAAAAA"><!ENTITY b "&a;&a;&a;&a;&a;&a;">]>'
            b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://x.test/&b;</loc></url></urlset>')
    sm = parse_sitemap(bomb)
    assert all("AAAAAAAAAA" not in e.loc for e in sm.entries)


def test_parse_sitemap_size_cap() -> None:
    import gzip
    with pytest.raises(ValueError):
        parse_sitemap(gzip.compress(b"<urlset>" + b" " * 5000 + b"</urlset>"), limit=1000)


def test_type_key_masks_digits() -> None:
    assert type_key("https://x.test/post-sitemap12.xml") == type_key("https://x.test/post-sitemap3.xml")
    assert type_key("https://x.test/post-sitemap1.xml") != type_key("https://x.test/tag-sitemap1.xml")
    assert type_key("https://x.test/sitemap.xml?page=4") == "x.test/sitemap.xml?page=#"


# -- extraction heuristic, rendering, terms -------------------------------------------

def test_largest_text_block_handles_wrapped_paragraphs_and_ignores_sidebar() -> None:
    html = f"""<html><body>
      <div class="sidebar"><p>{URDU_P[:20]}</p></div>
      <div class="story"><div><p>{URDU_P}</p></div><div><p>{URDU_P}</p></div><div><p>English only</p></div></div>
    </body></html>"""
    body = largest_text_block(soup_of(html))
    assert body.split("\n") == [URDU_P, URDU_P]


def test_render_class() -> None:
    server = f"<p>{URDU_P * 5}</p>".encode()
    assert probe.render_class(server, URDU_P * 5) == "server"
    json_only = ('<div id="app"></div><script>window.__DATA__={"body":"' + "\\u0627\\u0631" * 600 + '"}</script>').encode()
    assert probe.render_class(json_only, "") == "json"
    assert probe.render_class(b'<div id="app"></div><script src="app.js"></script>', "") == "none"


def test_find_policy_links_prefers_terms_over_privacy() -> None:
    soup = soup_of("""<footer><a href="/privacy-policy">Privacy Policy</a> <a href="/page/qawaid">قواعد و ضوابط</a>
                      <a href="mailto:x@y.z">Contact</a> <a href="#top">Top</a></footer>""")
    links = probe.find_policy_links(soup, "https://x.test/")
    assert [(l["kind"], l["url"]) for l in links] == [("terms", "https://x.test/page/qawaid"),
                                                       ("privacy", "https://x.test/privacy-policy")]


def test_find_clauses_flags_categories_and_prohibitions() -> None:
    text = ("You must not use any robot, spider or scraper to access the site. "
            "We may update these terms.\n"
            "Content may not be used for training artificial intelligence models.\n"
            "ہماری اجازت کے بغیر مواد کی نقل ممنوع ہے۔\n"
            "We use cookies to improve your experience on our website.")
    clauses = probe.find_clauses(text)
    by_quote = {c["quote"][:20]: c for c in clauses}
    assert len(clauses) == 3
    first = by_quote["You must not use any"]
    assert {"scraping/crawling", "robots"} <= set(first["categories"]) and first["prohibitive"]
    assert "AI/ML training" in by_quote["Content may not be u"]["categories"]
    urdu = next(c for c in clauses if c["quote"].startswith("ہماری"))
    assert urdu["categories"] == ["reproduction"] and urdu["prohibitive"]


def test_decide_rules() -> None:
    def res(**kw):
        r = probe.SiteResult("s", "S", "https://x.test/", "news", "x")
        r.articles = [{"outcome": "ok"}]
        r.robots_eval = {"urls_tested": 10, "rfc_ua": 10, "rfc_star": 10, "stdlib_ua": 10, "stdlib_star": 10}
        r.server_rendered = "server"
        r.terms_pages = [{"outcome": "ok"}]
        r.estimated_mb = 500
        for k, v in kw.items():
            setattr(r, k, v)
        return r
    assert probe.decide(res()) == "OK"
    assert probe.decide(res(challenge="Cloudflare challenge")).startswith("NO: bot challenge")
    assert probe.decide(res(server_rendered="none")) == "NO: article text needs JavaScript"
    assert probe.decide(res(robots_eval={"urls_tested": 4, "rfc_ua": 0, "rfc_star": 4, "stdlib_ua": 0,
                                         "stdlib_star": 4})).startswith("NO: robots.txt")
    scraping = {"categories": ["scraping/crawling"], "prohibitive": True, "quote": "no scraping", "page": "terms"}
    assert probe.decide(res(clauses=[scraping])).startswith("NO: terms page restrict scraping/crawling")
    in_robots = {**scraping, "page": "robots.txt"}
    # A prohibition in robots.txt comments decides even when no article was fetched.
    assert probe.decide(res(clauses=[in_robots], articles=[])).startswith("NO: robots.txt comments restrict")
    reproduction = {"categories": ["reproduction"], "prohibitive": True, "quote": "no copying", "page": "terms"}
    assert "reproduction clause only" in probe.decide(res(clauses=[reproduction]))


# -- end to end ------------------------------------------------------------------------

def article_html(i: int) -> bytes:
    return f"""<html><head><title>Story {i}</title>
      <meta property="article:published_time" content="2025-0{1 + i % 9}-01T10:00:00+05:00"></head>
      <body><nav><p>{URDU_P[:15]}</p></nav><article><h1>عنوان {i}</h1>
      <div class="body"><p>{URDU_P * 2}</p><p>{URDU_P * 2}</p><p>{URDU_P * (i % 3 + 1)}</p></div></article>
      <footer><a href="/terms-of-use">شرائط</a></footer></body></html>""".encode()


def build_site(site, children: int, per_child: int) -> list[str]:
    base = site.base_url
    site.robots = f"User-agent: *\nDisallow: /search\nCrawl-delay: 2\nSitemap: {base}/sitemap_index.xml\n"
    idx = "".join(f"<sitemap><loc>{base}/post-sitemap{c}.xml</loc></sitemap>" for c in range(1, children + 1))
    idx += f"<sitemap><loc>{base}/tag-sitemap1.xml</loc></sitemap>"
    site.files["/sitemap_index.xml"] = f"<sitemapindex>{idx}</sitemapindex>".encode()
    urls = []
    n = 0
    for c in range(1, children + 1):
        locs = []
        for _ in range(per_child):
            n += 1
            url = f"{base}/{1000 + n}/slug-{n}/"
            urls.append(url)
            locs.append(f"<url><loc>{url}</loc></url>")
            site.files[f"/{1000 + n}/slug-{n}/"] = article_html(n)
        site.files[f"/post-sitemap{c}.xml"] = f"<urlset>{''.join(locs)}</urlset>".encode()
    site.files["/tag-sitemap1.xml"] = f"<urlset><url><loc>{base}/tag/x/</loc></url></urlset>".encode()
    site.files["/"] = b'<html><body><footer><a href="/terms-of-use">Terms of Use</a></footer></body></html>'
    site.files["/terms-of-use"] = (b"<html><body><main><h1>Terms</h1><p>Do not reproduce our content without "
                                   b"permission.</p><p>We may change these terms.</p></main></body></html>")
    return urls


def run_probe(site, tmp_path: Path, max_requests: int, articles: int) -> probe.SiteResult:
    cand = probe.Candidate("fake", "Fake Site", site.url("/"), "127.0.0.1", r"^https?://[^/]+/\d+/[^/]+/$")
    args = argparse.Namespace(user_agent=UA, min_interval=0.0, jitter=0.0, max_requests=max_requests,
                              articles=articles, seed=1337, evidence_dir=tmp_path / "evidence",
                              data_dir=tmp_path / "probe")
    with EvidenceLog(tmp_path / "evidence" / "probe_log.jsonl.gz") as evidence:
        return probe.probe_site(cand, args, evidence, "20261011", logging.getLogger("test"),
                                client_kwargs={"sleep": lambda s: None})


def test_probe_end_to_end_exact_count(site, tmp_path: Path) -> None:
    build_site(site, children=3, per_child=4)
    res = run_probe(site, tmp_path, max_requests=30, articles=5)
    assert res.challenge is None and not res.errors, res.errors
    assert res.estimated_articles == 12 and res.estimate_exact
    assert res.crawl_delay_star == 2.0
    assert res.robots_eval["urls_tested"] == 12 and res.robots_eval["rfc_ua"] == 12
    ok = [a for a in res.articles if a["outcome"] == "ok"]
    assert len(ok) == 5 and all(a["render"] == "server" for a in ok)
    assert res.median_bytes and res.estimated_mb == pytest.approx(12 * res.median_bytes / 1e6)
    assert res.server_rendered == "server"
    assert res.verdict.startswith("OK") and "reproduction clause only" in res.verdict
    robots_file = tmp_path / "evidence" / "robots" / "fake_20261011.txt"
    assert robots_file.read_text().startswith("User-agent: *")
    terms_file = tmp_path / "evidence" / "terms" / "fake_20261011.txt"
    assert "Do not reproduce our content" in terms_file.read_text()
    assert len(list((tmp_path / "probe" / "fake").glob("*.html.gz"))) == 6  # 5 articles + home page
    rows = list(iter_log(tmp_path / "evidence" / "probe_log.jsonl.gz"))
    assert len(rows) == res.requests <= 30 and all(r["site"] == "fake" for r in rows)
    assert "/search" not in {r["path"] for r in site.requests}


def test_probe_extrapolates_when_budget_is_small(site, tmp_path: Path) -> None:
    build_site(site, children=40, per_child=5)
    res = run_probe(site, tmp_path, max_requests=14, articles=3)
    group = next(g for g in res.sitemap_groups if "post-sitemap" in g["key"])
    assert group["children"] == 40 and 0 < group["sampled"] < 40 and not group["exact"]
    assert res.estimated_articles == 200 and not res.estimate_exact  # every child holds 5, so the mean is exact
    assert res.requests <= 14 + 1


def test_robots_comment_clauses() -> None:
    bbc_like = ("# In short: use our site like a human.\n# - No scraping, crawling, or systematic extraction\n"
                "# - No use of content for training AI models\nUser-agent: *\nDisallow: /search\n")
    dw_like = ("User-agent: *\n# Content is for personal use. Use of any tool designed to data mine or scrape\n"
               "# the content using automated means is prohibited without prior written permission.\n")
    drupal = ("#\n# This file is to prevent the crawling and indexing of certain parts\n# of your site by web crawlers"
              " and spiders. By telling these \"robots\" where not to go on your site,\n# you save bandwidth.\n"
              "User-agent: *\nCrawl-delay: 10\n")
    bbc = probe.robots_clauses(bbc_like)
    assert {"scraping/crawling", "AI/ML training"} <= {cat for c in bbc for cat in c["categories"]}
    assert all(c["page"] == "robots.txt" for c in bbc)
    dw = probe.robots_clauses(dw_like)
    assert len(dw) == 1 and dw[0]["prohibitive"] and "data mining" in dw[0]["categories"]
    assert probe.robots_clauses(drupal) == []


def test_home_article_links_fallback() -> None:
    cand = probe.Candidate("x", "X", "https://www.x.test/", "x.test", r"^https://www\.x\.test/node/\d+")
    home = b"""<a href="/node/12/story">a</a><a href="https://www.x.test/node/13">b</a>
               <a href="/tag/sport">t</a><a href="https://other.test/node/1">o</a>"""
    assert probe.home_article_links(home, cand) == (["https://www.x.test/node/12/story",
                                                     "https://www.x.test/node/13"], True)
    cand2 = probe.Candidate("x", "X", "https://www.x.test/", "x.test", r"^nothing$")
    urls, matched = probe.home_article_links(home, cand2)
    assert not matched and urls == ["https://www.x.test/node/12/story", "https://www.x.test/node/13"]


def test_largest_text_block_takes_a_story_span_without_p() -> None:
    html = f"""<html><body><div class="story"><span class="story-text"><strong>{URDU_P}</strong><br>{URDU_P}<br>
      {URDU_P}</span></div><footer><p>{URDU_P[:40]}</p></footer></body></html>"""
    assert largest_text_block(soup_of(html)).split("\n") == [URDU_P] * 3


def test_policy_links_ignore_headlines_and_prefer_footer() -> None:
    soup = soup_of(f"""<main><a href="/n/1/2026/10/09/x/">شرائط اور ضد: کیا سیاست دان بس یہی جانتے ہیں؟</a>
        <a href="/terms">Terms</a></main>
        <footer><a href="/page/terms-of-use">استعمال کی شرائط</a></footer>""")
    links = probe.find_policy_links(soup, "https://x.test/")
    assert [l["url"] for l in links] == ["https://x.test/page/terms-of-use", "https://x.test/terms"]


def test_probe_counts_leaf_urls_when_the_regex_guess_is_wrong(site, tmp_path: Path) -> None:
    build_site(site, children=3, per_child=4)
    cand = probe.Candidate("fake", "Fake Site", site.url("/"), "127.0.0.1", r"^https?://[^/]+/story/\d+$")
    args = argparse.Namespace(user_agent=UA, min_interval=0.0, jitter=0.0, max_requests=30, articles=3, seed=1337,
                              evidence_dir=tmp_path / "evidence", data_dir=tmp_path / "probe")
    with EvidenceLog(tmp_path / "evidence" / "log.jsonl.gz") as evidence:
        res = probe.probe_site(cand, args, evidence, "20261011", logging.getLogger("test"),
                               client_kwargs={"sleep": lambda s: None})
    assert res.regex_fallback and res.article_urls_seen == 0
    assert res.estimated_articles == 12  # leaf-like URLs; /tag/x/ is not counted
    assert len([a for a in res.articles if a["outcome"] == "ok"]) == 3
    assert any("highest numeric ID" in n for n in res.estimate_notes)


def test_largest_text_block_takes_h5_paragraphs_and_skips_link_lists() -> None:
    related = "".join(f'<li><a href="/n/{i}">{URDU_P}</a></li>' for i in range(8))
    html = f"""<html><body><div class="article"><h5>{URDU_P}</h5><h5>{URDU_P}</h5><h5>{URDU_P}</h5></div>
      <ul class="more">{related}</ul></body></html>"""
    assert largest_text_block(soup_of(html)).split("\n") == [URDU_P] * 3
