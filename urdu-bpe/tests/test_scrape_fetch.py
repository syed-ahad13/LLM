"""ScrapeClient: robots enforcement (both parsers), redirects, cookies, pacing, blocks, evidence log, cache."""

import gzip
from pathlib import Path

import pytest

from collect.scrape.fetch import (
    BotChallenge, EvidenceLog, HtmlCache, ScrapeClient, SiteBlocked, detect_challenge, iter_log, same_site,
)

UA = "TestBot/1.0 (tests; contact: nobody@example.invalid)"


class FrozenClock:
    """Time stands still, so every pacing sleep equals the full interval."""

    def __call__(self) -> float:
        return 1000.0


def make_client(log: EvidenceLog | None, sleeps: list[float], **kw) -> ScrapeClient:
    kw.setdefault("min_interval", 0)
    kw.setdefault("jitter", 0)
    c = ScrapeClient(UA, evidence=log, max_retries=2, backoff_base=0.01, sleep=sleeps.append, **kw)
    c.session.trust_env = False  # never route 127.0.0.1 through a proxy from the environment
    return c


@pytest.fixture
def evlog(tmp_path: Path):
    log = EvidenceLog(tmp_path / "crawl_log_test.jsonl.gz", {"site": "test"})
    yield log
    log.close()


def paths(site) -> list[str]:
    return [r["path"] for r in site.requests]


# -- robots.txt --------------------------------------------------------------

def test_disallowed_url_is_never_requested_and_is_logged(site, sleeps, evlog) -> None:
    site.files["/private/a"] = b"secret"
    c = make_client(evlog, sleeps)
    r = c.fetch(site.url("/private/a"), kind="article")
    assert r.outcome == "robots" and r.body == b""
    assert paths(site) == ["/robots.txt"]
    evlog.close()
    rows = list(iter_log(evlog.path))
    assert [row["kind"] for row in rows] == ["robots", "article"]
    blocked = rows[1]
    assert blocked["status"] is None and blocked["robots_allowed"] is False and blocked["bytes"] == 0
    assert blocked["site"] == "test" and blocked["robots_rfc9309"] is False


def test_wildcard_rule_blocks_although_stdlib_parser_would_allow(site, sleeps) -> None:
    # urllib.robotparser treats '*' literally, so only our RFC 9309 matcher catches this rule.
    site.robots = "User-agent: *\nDisallow: /*?print=\n"
    site.files["/news/1"] = b"<p>x</p>"
    c = make_client(None, sleeps)
    assert c.robots_verdicts(site.url("/news/1?print=1")) == (False, True)
    assert c.fetch(site.url("/news/1?print=1")).outcome == "robots"
    assert c.fetch(site.url("/news/1")).ok
    assert paths(site) == ["/robots.txt", "/news/1"]


def test_stdlib_first_match_blocks_although_rfc_would_allow(site, sleeps) -> None:
    # RFC 9309 picks the longest match (Allow /news/), urllib.robotparser the first (Disallow /).
    # Both must allow, so the stricter answer wins.
    site.robots = "User-agent: *\nDisallow: /\nAllow: /news/\n"
    site.files["/news/1"] = b"x"
    c = make_client(None, sleeps)
    assert c.robots_verdicts(site.url("/news/1")) == (True, False)
    assert c.fetch(site.url("/news/1")).outcome == "robots"
    assert paths(site) == ["/robots.txt"]


def test_group_for_our_token_applies(site, sleeps) -> None:
    site.robots = "User-agent: *\nDisallow:\n\nUser-agent: TestBot\nDisallow: /news/\n"
    site.files["/news/1"] = b"x"
    c = make_client(None, sleeps)
    assert c.fetch(site.url("/news/1")).outcome == "robots"
    assert c.robots_verdicts(site.url("/news/1"), "*") == (True, True)


def test_unreachable_robots_disallows_everything(site, sleeps) -> None:
    site.robots_status = 503
    site.files["/a"] = b"x"
    c = make_client(None, sleeps)
    assert c.fetch(site.url("/a")).outcome == "robots"
    assert "/a" not in paths(site)


def test_robots_404_means_no_restrictions(site, sleeps) -> None:
    site.robots = None
    site.files["/a"] = b"x"
    assert make_client(None, sleeps).fetch(site.url("/a")).ok


# -- redirects -----------------------------------------------------------------

def test_redirect_hop_is_robots_checked(site, sleeps) -> None:
    site.redirects["/a"] = (301, "/private/b")
    site.files["/private/b"] = b"secret"
    r = make_client(None, sleeps).fetch(site.url("/a"))
    assert r.outcome == "robots" and r.final_url == site.url("/private/b")
    assert "/private/b" not in paths(site)


def test_redirect_rejected_by_url_filter_is_not_followed(site, sleeps) -> None:
    site.redirects["/a"] = (302, "https://login.example.invalid/signin")
    r = make_client(None, sleeps).fetch(site.url("/a"), allow_url=lambda u: same_site(u, "127.0.0.1"))
    assert r.outcome == "redirect_rejected" and r.final_url == "https://login.example.invalid/signin"
    assert paths(site) == ["/robots.txt", "/a"]


def test_redirect_followed_when_allowed(site, sleeps) -> None:
    site.redirects["/old"] = (301, "/new")
    site.files["/new"] = b"<p>body</p>"
    r = make_client(None, sleeps).fetch(site.url("/old"))
    assert r.ok and r.body == b"<p>body</p>" and r.redirects == [site.url("/new")]


def test_redirect_loop_stops(site, sleeps) -> None:
    site.redirects["/a"] = (302, "/b")
    site.redirects["/b"] = (302, "/a")
    r = make_client(None, sleeps).fetch(site.url("/a"), max_redirects=3)
    assert r.outcome == "too_many_redirects"
    assert paths(site).count("/a") + paths(site).count("/b") == 4


# -- cookies, user agent, pacing ---------------------------------------------------

def test_cookies_are_never_stored_or_sent(site, sleeps) -> None:
    site.files["/a"] = b"x"
    site.files["/b"] = b"y"
    site.set_cookie = "session=abc; Path=/"
    c = make_client(None, sleeps)
    assert c.fetch(site.url("/a")).ok and c.fetch(site.url("/b")).ok
    assert len(c.session.cookies) == 0
    assert all(r["cookie"] is None for r in site.requests)
    assert all(r["user_agent"] == UA for r in site.requests)


def test_delay_is_min_interval_plus_jitter(site, sleeps) -> None:
    for i in range(6):
        site.files[f"/p{i}"] = b"x"
    c = make_client(None, sleeps, min_interval=1.0, jitter=0.5, clock=FrozenClock())
    for i in range(6):
        c.fetch(site.url(f"/p{i}"))
    assert len(sleeps) == 6  # robots.txt first, then one wait before each of the 6 pages
    assert all(1.0 <= s <= 1.5 for s in sleeps)
    assert len(set(sleeps)) > 1


def test_crawl_delay_raises_the_interval(site, sleeps) -> None:
    site.robots = "User-agent: *\nCrawl-delay: 4\n"
    site.files["/a"] = b"x"
    site.files["/b"] = b"y"
    c = make_client(None, sleeps, min_interval=1.0, jitter=0.5, clock=FrozenClock())
    c.fetch(site.url("/a"))
    c.fetch(site.url("/b"))
    assert sleeps and all(4.0 <= s <= 4.5 for s in sleeps)


# -- refusals, rate limits, challenges ---------------------------------------------

def test_429_retry_after_is_honoured(site, sleeps) -> None:
    site.files["/a"] = b"x"
    site.scripted["/a"] = [(429, {"Retry-After": "7"}, b"slow down")]
    r = make_client(None, sleeps).fetch(site.url("/a"))
    assert r.ok
    assert max(sleeps) >= 7


def test_retry_after_longer_than_cap_stops_the_site(site, sleeps) -> None:
    site.scripted["/a"] = [(503, {"Retry-After": "99999"}, b"maintenance")]
    with pytest.raises(SiteBlocked):
        make_client(None, sleeps, retry_after_cap=60).fetch(site.url("/a"))
    assert paths(site).count("/a") == 1


def test_consecutive_403s_stop_the_site_and_a_200_resets_the_count(site, sleeps) -> None:
    for i in range(5):
        site.scripted[f"/x{i}"] = [(403, {}, b"forbidden")]
    site.files["/ok"] = b"fine"
    c = make_client(None, sleeps, max_consecutive_403=3)
    assert c.fetch(site.url("/x0")).outcome == "forbidden"
    assert c.fetch(site.url("/x1")).outcome == "forbidden"
    assert c.fetch(site.url("/ok")).ok
    assert c.fetch(site.url("/x2")).outcome == "forbidden"
    assert c.fetch(site.url("/x3")).outcome == "forbidden"
    with pytest.raises(SiteBlocked):
        c.fetch(site.url("/x4"))


def test_bot_challenge_stops_immediately_without_retrying(site, sleeps, evlog) -> None:
    site.scripted["/a"] = [(503, {"Server": "cloudflare"}, b"<title>Just a moment...</title><script>cf_chl_opt</script>")]
    c = make_client(evlog, sleeps)
    with pytest.raises(BotChallenge) as e:
        c.fetch(site.url("/a"))
    assert "Cloudflare" in e.value.reason
    assert paths(site).count("/a") == 1
    evlog.close()
    assert list(iter_log(evlog.path))[-1]["challenge"].startswith("Cloudflare")


def test_challenge_on_robots_txt_stops_the_site(site, sleeps) -> None:
    site.robots = None
    site.scripted["/robots.txt"] = [(403, {"cf-mitigated": "challenge"}, b"challenge")]
    with pytest.raises(BotChallenge):
        make_client(None, sleeps).fetch(site.url("/a"))


def test_404_and_401_are_recorded_not_raised(site, sleeps) -> None:
    site.scripted["/members"] = [(401, {}, b"login")]
    c = make_client(None, sleeps)
    assert c.fetch(site.url("/missing")).outcome == "not_found"
    assert c.fetch(site.url("/members")).outcome == "login_required"


@pytest.mark.parametrize("status, headers, body, expected", [
    (403, {"cf-mitigated": "challenge"}, b"", "Cloudflare challenge (cf-mitigated header)"),
    (503, {}, b"<html><title>Just a moment...</title>", "Cloudflare challenge"),
    (403, {}, b"Attention Required! | Cloudflare", "Cloudflare block"),
    (403, {}, b"<p>Sucuri WebSite Firewall - Access Denied</p>", "Sucuri firewall block"),
    (403, {"X-DataDome": "protected"}, b"", "DataDome block"),
    (202, {"x-amzn-waf-action": "challenge"}, b"", "AWS WAF challenge"),
    # Cloudflare injects challenge-platform scripts into normal pages: not a challenge.
    (200, {"Server": "cloudflare"}, b"<p>article</p><script src='/cdn-cgi/challenge-platform/x.js'></script>" * 50, None),
    (403, {}, b"Forbidden", None),
    (200, {}, b"<title>Just a moment...</title>", "Cloudflare challenge (interstitial page)"),
])
def test_detect_challenge(status, headers, body, expected) -> None:
    assert detect_challenge(status, headers, body) == expected


# -- evidence log and cache ----------------------------------------------------------

def test_every_attempt_is_logged_with_required_fields(site, sleeps, evlog) -> None:
    site.files["/a"] = b"0123456789"
    site.status_queue["/a"] = [503]
    c = make_client(evlog, sleeps)
    c.fetch(site.url("/a"), kind="article")
    evlog.close()
    rows = list(iter_log(evlog.path))
    assert [(r["kind"], r["status"]) for r in rows] == [("robots", 200), ("article", 503), ("article", 200)]
    for r in rows:
        assert {"url", "utc", "status", "bytes", "robots_allowed", "elapsed_ms", "kind", "site"} <= r.keys()
        assert r["utc"].endswith("Z") and r["robots_allowed"] is True
    assert rows[-1]["bytes"] == 10


def test_log_survives_a_killed_run(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl.gz"
    log = EvidenceLog(path)
    for i in range(3):
        log.write({"i": i})
    log._fh.fileobj.flush()  # simulate SIGKILL: lines sync-flushed, gzip trailer never written
    data = path.read_bytes()
    log._fh = gzip.open(tmp_path / "discard.gz", "ab")  # detach without closing the real member
    log.close()
    path.write_bytes(data)
    assert [r["i"] for r in iter_log(path)] == [0, 1, 2]
    with EvidenceLog(path) as log2:  # repairs, then appends a new member
        log2.write({"i": 3})
    with gzip.open(path, "rt") as f:  # readable by any gzip tool again
        assert len(f.read().splitlines()) == 4
    assert [r["i"] for r in iter_log(path)] == [0, 1, 2, 3]


def test_html_cache_round_trip_and_damaged_entry(tmp_path: Path) -> None:
    cache = HtmlCache(tmp_path / "html")
    url = "https://example.invalid/a/1"
    assert cache.get(url) is None
    p = cache.put(url, "<p>اردو</p>".encode())
    assert p.name.endswith(".html.gz") and len(p.stem.split(".")[0]) == 40
    assert cache.get(url) == "<p>اردو</p>".encode()
    p.write_bytes(b"not gzip")
    assert cache.get(url) is None


def test_same_site() -> None:
    assert same_site("https://www.dw.com/ur/x/a-1", "dw.com", "/ur")
    assert not same_site("https://www.dw.com/en/x/a-1", "dw.com", "/ur")
    assert not same_site("https://evil-dw.com/ur/", "dw.com")
    assert not same_site("ftp://dw.com/ur/", "dw.com")
