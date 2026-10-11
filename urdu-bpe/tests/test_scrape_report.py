"""report.py on the output of a crawl of the scripted local site (see test_scrape_crawl.py)."""

import json
import shutil
from pathlib import Path

from collect.scrape import report
from common import sha256_file
from tests.test_scrape_crawl import fake_site, run  # noqa: F401  (fixture reused)

ROOT = Path(__file__).resolve().parent.parent


def test_report_and_sources_entry(fake_site, tmp_path: Path) -> None:  # noqa: F811
    # fake_site registers the test adapter in the shared registry (collect/scrape/sites/registry.ADAPTERS).
    assert run(tmp_path) == 0
    (tmp_path / "evidence" / "robots").mkdir(parents=True)
    (tmp_path / "evidence" / "robots" / "fake_20261011.txt").write_text("User-agent: *\n")
    sources = tmp_path / "sources.json"
    shutil.copy(ROOT / "collect" / "sources.json", sources)
    assert report.main(["--sites", "fake", "--data-dir", str(tmp_path / "own"), "--evidence-dir",
                        str(tmp_path / "evidence"), "--sources", str(sources), "--log-dir", str(tmp_path / "logs")]) == 0
    md = (tmp_path / "evidence" / "scrape_report.md").read_text()
    articles = tmp_path / "own" / "fake" / "articles.jsonl"
    assert sha256_file(articles) in md
    assert "Articles kept: 8" in md and "media page: 1" in md and "duplicate canonical URL: 1" in md
    assert "blocked by robots: 1" in md and "Random sample of 8 articles" in md
    assert "fake_20261011.txt" in md
    entry = next(e for e in json.loads(sources.read_text())["sources"] if e["id"] == "own_scrape")
    assert entry["collected_by_us"] is True and entry["access_method"] == "scrape"
    assert entry["raw_bytes"] == articles.stat().st_size
    assert entry["sha256"] == sha256_file(articles)  # one site: the file's own SHA-256
    assert entry["licence"] == "Fake: synthetic test site"
    assert entry["sites"][0]["articles"] == 8 and entry["text_bytes"] > 0
    assert entry["collection_date"] and len(entry["collection_date"]) == 10
    other = {e["id"]: e for e in json.loads((ROOT / "collect" / "sources.json").read_text())["sources"]}
    after = {e["id"]: e for e in json.loads(sources.read_text())["sources"]}
    assert {k: v for k, v in after.items() if k != "own_scrape"} == {k: v for k, v in other.items() if k != "own_scrape"}
