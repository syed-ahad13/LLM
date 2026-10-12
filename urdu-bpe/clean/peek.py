"""Print random documents from data/interim/<source>.jsonl.gz in full (Step 3 acceptance).

    python clean/peek.py --source ufal_umc --n 2
    python clean/peek.py --source wiki_ur --n 1 --raw      # also the page's raw wikitext from the dump

The pick is a uniform reservoir sample with a fixed seed, so it is repeatable.
With --raw (wiki_ur only), the dump is scanned again for the picked page ids
and each raw wikitext is printed above its extracted text.
"""

import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clean.extract_common import INTERIM_DIR, RAW_DIR  # noqa: E402
from collect.inspect_raw import iter_wiki_pages  # noqa: E402
from common import iter_jsonl  # noqa: E402


def pick(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for i, doc in enumerate(iter_jsonl(path), start=1):
        if len(sample) < n:
            sample.append(doc)
        elif (j := rng.randrange(i)) < n:
            sample[j] = doc
    return sample


def raw_wikitext(dump: Path, page_ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for page in iter_wiki_pages(dump):
        if page["id"] in page_ids:
            found[page["id"]] = page["text"]
            if len(found) == len(page_ids):
                break
    return found


def show(doc: dict[str, Any]) -> None:
    head = {k: v for k, v in doc.items() if k != "text"}
    print(f"--- {doc['doc_id']}  {head}")
    print(doc["text"])
    print()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--raw", action="store_true", help="wiki_ur only: also print the raw wikitext")
    p.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    docs = pick(args.interim_dir / f"{args.source}.jsonl.gz", args.n, args.seed)
    raws: dict[str, str] = {}
    if args.raw and args.source == "wiki_ur":
        dumps = sorted((RAW_DIR / "wiki_ur").glob("*.xml.bz2"))
        raws = raw_wikitext(dumps[-1], {d["meta"]["page_id"] for d in docs}) if dumps else {}
    for doc in docs:
        if raws:
            print(f"=== RAW WIKITEXT, page {doc['meta']['page_id']} ===")
            print(raws.get(doc["meta"]["page_id"], "(not found)"))
            print("=== EXTRACTED ===")
        show(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
