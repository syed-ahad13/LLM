"""Run any subset of the Step 3 extractors in parallel and print a per-source table.

    python clean/extract_all.py --limit 2000                         # every enabled source with raw data
    python clean/extract_all.py --sources wiki_ur ufal_umc --wiki-jobs 6
    python clean/extract_all.py --table                              # only print the table from data/stats/extract.json

Each source runs as its own process (clean/extract_<x>.py), so the Wikipedia
extractor can still start its own worker pool. Their logs go to stderr and to
logs/extract_<x>.log. The table comes from data/stats/extract.json: records
read, docs written, MB written and drops by reason.
"""

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import RAW_DIR, SOURCES_JSON, STAGE  # noqa: E402
from common import STATS_DIR, setup_logging  # noqa: E402

SCRIPTS = {
    "wiki_ur": "extract_wiki.py",
    "ufal_umc": "extract_ufal.py",
    "urdu_news_1m": "extract_news1m.py",
    "fineweb2_urd": "extract_fineweb2.py",
    "cc100_ur": "extract_cc100.py",
    "makhzan": "extract_makhzan.py",
    "own_scrape": "extract_scrape.py",
}


def enabled_sources(path: Path = SOURCES_JSON) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [s["id"] for s in json.load(f)["sources"] if s.get("enabled") and s["id"] in SCRIPTS]


def command(source: str, args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve().parent / SCRIPTS[source])]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.stats_dir != STATS_DIR:
        cmd += ["--stats-dir", str(args.stats_dir)]
    if source == "wiki_ur" and args.wiki_jobs:
        cmd += ["--jobs", str(args.wiki_jobs)]
    if source == "fineweb2_urd" and args.fineweb_keep_per_mille is not None:
        cmd += ["--keep-per-mille", str(args.fineweb_keep_per_mille)]
    if source == "ufal_umc" and args.ufal_group:
        cmd += ["--group", args.ufal_group]
    return cmd


def format_table(stats: dict[str, dict[str, Any]], sources: Sequence[str] | None = None) -> str:
    rows = [("source", "records read", "docs written", "MB written", "drops by reason")]
    totals = [0, 0, 0.0]
    for source in sources or sorted(stats):
        s = stats.get(source)
        if s is None:
            rows.append((source, "-", "-", "-", "no stats"))
            continue
        drops = ", ".join(f"{k.removeprefix('drop_')}={v:,}" for k, v in sorted(s.items()) if k.startswith("drop_"))
        mb = s.get("bytes_written", 0) / 1e6
        limit = f" (limit {s['limit']:,})" if s.get("limit") else ""
        rows.append((source + limit, f"{s.get('records_read', 0):,}", f"{s.get('docs_written', 0):,}", f"{mb:,.2f}",
                     drops or "none"))
        totals[0] += s.get("records_read", 0)
        totals[1] += s.get("docs_written", 0)
        totals[2] += mb
    rows.append(("total", f"{totals[0]:,}", f"{totals[1]:,}", f"{totals[2]:,.2f}", ""))
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = []
    for i, r in enumerate(rows):
        lines.append("  ".join([r[0].ljust(widths[0])] + [r[j].rjust(widths[j]) for j in (1, 2, 3)] + [r[4]]))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths) + "  " + "-" * 15)
    return "\n".join(lines)


def load_stats(stats_dir: Path) -> dict[str, dict[str, Any]]:
    path = stats_dir / f"{STAGE}.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", choices=sorted(SCRIPTS), metavar="SOURCE",
                   help=f"sources to run (default: enabled sources with a data/raw folder). Choices: {', '.join(SCRIPTS)}")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="pass --limit N to every extractor")
    p.add_argument("--parallel", type=int, default=len(SCRIPTS), help="sources running at once (default: all)")
    p.add_argument("--wiki-jobs", type=int, default=None, help="--jobs for extract_wiki.py")
    p.add_argument("--fineweb-keep-per-mille", type=int, default=None, metavar="K")
    p.add_argument("--ufal-group", default=None, help="--group for extract_ufal.py (auto or N)")
    p.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    p.add_argument("--table", action="store_true", help="only print the table from existing stats")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log = setup_logging("extract_all")
    sources = args.sources or [s for s in enabled_sources() if (RAW_DIR / s).is_dir()]
    if args.table:
        print(format_table(load_stats(args.stats_dir), sources))
        return 0
    if not sources:
        log.error("nothing to run: no enabled source has a folder in %s", RAW_DIR)
        return 1
    log.info("running %s", ", ".join(sources))
    queue = list(sources)
    running: dict[str, tuple[subprocess.Popen[bytes], float]] = {}
    failed: list[str] = []
    while queue or running:
        while queue and len(running) < args.parallel:
            source = queue.pop(0)
            cmd = command(source, args)
            log.info("start %s: %s", source, " ".join(cmd))
            running[source] = (subprocess.Popen(cmd, cwd=ROOT), time.monotonic())
        for source, (proc, started) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[source]
            elapsed = time.monotonic() - started
            if proc.returncode == 0:
                log.info("done %s in %.0f s", source, elapsed)
            else:
                log.error("FAILED %s (exit %d) after %.0f s", source, proc.returncode, elapsed)
                failed.append(source)
        time.sleep(0.2)
    print(format_table(load_stats(args.stats_dir), sources))
    if failed:
        log.error("failed: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
