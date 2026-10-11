import pytest

from collect.robots import RobotsTxt

BOT = "UrduCorpusCourseBot"

# Excerpts of real files fetched on 2026-10-11.
LINDAT = """\
User-agent: *
Disallow: /repository/*allzip$
User-agent: *
# DSpace 7
Disallow: /repository/handle
Disallow: /repository/admin/*
"""
MENDELEY = """\
User-Agent: *
Disallow: /admin/
Allow /api/docs
Disallow: /api/
Disallow: /*?*page=
"""
GITHUB = """\
User-agent: *
Allow: /*?tab=achievements&achievement=*
Disallow: /*/archive/
Disallow: /*.git$
"""
STATMT = "User-agent: *\nDisallow: /\n"


@pytest.mark.parametrize(
    "text, url, allowed",
    [
        (LINDAT, "https://lindat.mff.cuni.cz/repository/handle/11858/00-097C-0000-0023-65A9-5", False),
        (LINDAT, "https://lindat.mff.cuni.cz/repository/server/api/pid/find?id=hdl:11858/x", True),
        (LINDAT, "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/f0/content", True),
        (LINDAT, "https://lindat.mff.cuni.cz/repository/items/x/allzip", False),
        (LINDAT, "https://lindat.mff.cuni.cz/repository/items/x/allzip?a=1", True),  # $ anchors the end
        (LINDAT, "https://lindat.mff.cuni.cz/repository/admin/x", False),
        (MENDELEY, "https://data.mendeley.com/api/datasets", False),
        (MENDELEY, "https://data.mendeley.com/public-api/datasets/834vsxnb99?version=4", True),
        (MENDELEY, "https://data.mendeley.com/datasets/x?q=1&page=2", False),
        (MENDELEY, "https://data.mendeley.com/api/docs", False),  # "Allow /api/docs" lacks a colon: ignored
        (GITHUB, "https://github.com/zeerakahmed/makhzan/archive/refs/heads/master.zip", False),
        (GITHUB, "https://github.com/zeerakahmed/makhzan", True),
        (GITHUB, "https://github.com/zeerakahmed/makhzan.git", False),
        (GITHUB, "https://github.com/zeerakahmed/makhzan.github.io", True),
        (STATMT, "https://data.statmt.org/cc-100/ur.txt.xz", False),
        (STATMT, "https://data.statmt.org/robots.txt", True),  # robots.txt itself is always allowed
        ("", "https://example.org/anything", True),
    ],
)
def test_real_world_rules(text: str, url: str, allowed: bool) -> None:
    assert RobotsTxt(text).allowed(url, BOT) is allowed


def test_longest_match_wins_and_allow_wins_ties() -> None:
    robots = RobotsTxt("User-agent: *\nDisallow: /a\nAllow: /a/b\nDisallow: /x\nAllow: /x\n")
    assert not robots.allowed("http://h/a/c", BOT)
    assert robots.allowed("http://h/a/b/c", BOT)
    assert robots.allowed("http://h/x", BOT)


def test_specific_group_replaces_star_group() -> None:
    text = "User-agent: *\nDisallow: /\n\nUser-agent: urducorpuscoursebot\nUser-agent: other\nDisallow: /secret\n"
    robots = RobotsTxt(text)
    assert robots.allowed("http://h/page", BOT)
    assert not robots.allowed("http://h/secret", BOT)
    assert not robots.allowed("http://h/page", "SomeoneElse")


def test_empty_disallow_and_rules_before_any_agent() -> None:
    robots = RobotsTxt("Disallow: /ignored\nUser-agent: *\nDisallow:\n")
    assert robots.allowed("http://h/ignored", BOT)


def test_comments_and_crawl_delay() -> None:
    robots = RobotsTxt("User-agent: * # everyone\nCrawl-delay: 3.5\nDisallow: /tmp # scratch\n")
    assert robots.crawl_delay(BOT) == 3.5
    assert not robots.allowed("http://h/tmp/x", BOT)
    assert RobotsTxt("").crawl_delay(BOT) is None


def test_unreachable_disallows_everything_but_robots_txt() -> None:
    robots = RobotsTxt.unreachable()
    assert not robots.allowed("http://h/", BOT)
    assert robots.allowed("http://h/robots.txt", BOT)
